"""
FDA-MIMO 雷达参数估计对比: Legacy Methods (Matlab复现) vs CVNN
说明:
1. Legacy MUSIC: 固定粗网格 (1度, 100米), 无细搜索 -> 会出现误差平台 (Error Floor)
2. Legacy ESPRIT: 无相位解模糊 -> 容易出现距离模糊 (Ambiguity)
3. Legacy OMP: 固定粗网格 -> 同样受限于量化误差
4. CVNN: 深度学习方法
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
import glob
import argparse
from tqdm import tqdm

# 加载项目依赖
import config as cfg
from model import FDA_CVNN, FDA_CVNN_Attention
from utils_physics import generate_covariance_matrix

# 屏蔽警告
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 0. CRB 理论下界计算 (保持标准)
# ==========================================
def compute_crb_full(snr_db, r_true, theta_true, L=None):
    L = L or cfg.L_snapshots
    M, N = cfg.M, cfg.N
    snr_linear = 10 ** (snr_db / 10.0)
    c, delta_f, d, wavelength = cfg.c, cfg.delta_f, cfg.d, cfg.wavelength
    theta_rad = np.deg2rad(theta_true)
    m, n = np.arange(M), np.arange(N)
    
    phi_tx = -4 * np.pi * delta_f * m * r_true / c + 2 * np.pi * d * m * np.sin(theta_rad) / wavelength
    a_tx = np.exp(1j * phi_tx)
    phi_rx = 2 * np.pi * d * n * np.sin(theta_rad) / wavelength
    a_rx = np.exp(1j * phi_rx)

    dphi_tx_dr = -4 * np.pi * delta_f * m / c
    da_tx_dr = 1j * dphi_tx_dr * a_tx
    da_dr = np.kron(da_tx_dr, a_rx)

    cos_theta = np.cos(theta_rad)
    dphi_tx_dtheta = 2 * np.pi * d * m * cos_theta / wavelength
    dphi_rx_dtheta = 2 * np.pi * d * n * cos_theta / wavelength
    da_tx_dtheta = 1j * dphi_tx_dtheta * a_tx
    da_rx_dtheta = 1j * dphi_rx_dtheta * a_rx
    da_dtheta = np.kron(da_tx_dtheta, a_rx) + np.kron(a_tx, da_rx_dtheta)

    D = np.column_stack([da_dr, da_dtheta * np.pi / 180])
    FIM = 2 * L * snr_linear * np.real(D.conj().T @ D)
    try:
        CRB = np.linalg.inv(FIM)
        return np.sqrt(CRB[0, 0]), np.sqrt(CRB[1, 1])
    except:
        return np.nan, np.nan

def compute_crb_average(snr_db, L=None, num_samples=200):
    crb_r_list, crb_theta_list = [], []
    for _ in range(num_samples):
        r = np.random.uniform(0, cfg.r_max)
        t = np.random.uniform(cfg.theta_min, cfg.theta_max)
        cr_r, cr_t = compute_crb_full(snr_db, r, t, L)
        if np.isfinite(cr_r) and np.isfinite(cr_t) and cr_r < cfg.r_max and cr_t < 180:
            crb_r_list.append(cr_r); crb_theta_list.append(cr_t)
    return np.mean(crb_r_list), np.mean(crb_theta_list)

# ==========================================
# 1. 传统算法 (Legacy / Matlab Implementations)
# ==========================================

def music_2d_legacy(R):
    """
    [MATLAB text_MUSIC_RMSE.m 复现]
    - 硬编码粗网格: Angle step=1 deg, Range step=100m
    - 无细搜索 (Refine)
    """
    M, N = cfg.M, cfg.N
    # 严格按照 Matlab 代码中的网格密度
    # Grid_theta = -50:1:50 (这里根据config范围适配，但步长保持1)
    theta_grid = np.arange(cfg.theta_min, cfg.theta_max + 1, 1) 
    # Grid_r = 0:100:5000 (步长保持100)
    r_grid = np.arange(0, cfg.r_max + 100, 100) 

    # 1. 噪声子空间
    w, v = np.linalg.eigh(R)
    Un = v[:, :-1] # 假设 K=1

    # 2. 构建字典 (向量化计算谱，代替Matlab的双重for循环，逻辑一致但速度快)
    R_mat, T_mat = np.meshgrid(r_grid, theta_grid, indexing='ij')
    R_flat, T_flat = R_mat.flatten(), T_mat.flatten()
    
    m = np.arange(M).reshape(-1, 1)
    n = np.arange(N).reshape(-1, 1)
    T_rad = np.deg2rad(T_flat)
    
    phi_tx = -4 * np.pi * cfg.delta_f * m * R_flat / cfg.c + 2 * np.pi * cfg.d * m * np.sin(T_rad) / cfg.wavelength
    a_tx = np.exp(1j * phi_tx)
    phi_rx = 2 * np.pi * cfg.d * n * np.sin(T_rad) / cfg.wavelength
    a_rx = np.exp(1j * phi_rx)
    A = (a_tx[:, np.newaxis, :] * a_rx[np.newaxis, :, :]).reshape(M*N, -1)
    
    # 3. 谱峰搜索
    # P = 1 / (a' * Un * Un' * a)
    denom = np.sum(np.abs(Un.conj().T @ A)**2, axis=0)
    spectrum = 1.0 / (denom + 1e-12)
    
    idx = np.argmax(spectrum)
    return R_flat[idx], T_flat[idx]

def esprit_2d_legacy(R):
    """
    [MATLAB text_ESPRIT_RMSE.m 复现]
    - 直接相位计算
    - 严重缺陷: 无相位解模糊 (Phase Unwrapping)，距离容易折叠
    """
    M, N = cfg.M, cfg.N
    K = 1
    w, v = np.linalg.eigh(R)
    Es = v[:, -K:] # 信号子空间

    # 1. 角度估计
    J1 = np.kron(np.eye(M), np.hstack([np.eye(N-1), np.zeros((N-1, 1))]))
    J2 = np.kron(np.eye(M), np.hstack([np.zeros((N-1, 1)), np.eye(N-1)]))
    Psi_theta = np.linalg.pinv(J1 @ Es) @ (J2 @ Es)
    
    # Matlab: theta_est = asin(angle(phi_theta) * lambda / (2*pi*d));
    theta_est = np.degrees(np.arcsin(np.angle(np.linalg.eigvals(Psi_theta)) * cfg.wavelength / (2 * np.pi * cfg.d)))

    # 2. 距离估计
    J3 = np.kron(np.hstack([np.eye(M-1), np.zeros((M-1, 1))]), np.eye(N))
    J4 = np.kron(np.hstack([np.zeros((M-1, 1)), np.eye(M-1)]), np.eye(N))
    Psi_r = np.linalg.pinv(J3 @ Es) @ (J4 @ Es)
    
    # Matlab: r_est = -(angle(phi_r) * c0) / (4*pi*Delta_f);
    # 缺陷: 当 4*pi*df*r/c > 2*pi 时发生模糊，Matlab代码未处理此情况
    r_est = -(np.angle(np.linalg.eigvals(Psi_r)) * cfg.c) / (4 * np.pi * cfg.delta_f)
    
    return float(np.real(r_est)), float(np.real(theta_est))

def omp_2d_legacy(R):
    """
    [MATLAB text_OMP_RMSE.m 复现]
    - 硬编码粗网格
    - 无细搜索
    """
    M, N = cfg.M, cfg.N
    w, v = np.linalg.eigh(R)
    y = v[:, -1] # 信号代理
    
    # 粗网格 (同 MUSIC)
    r_grid = np.arange(0, cfg.r_max + 100, 100)
    theta_grid = np.arange(cfg.theta_min, cfg.theta_max + 1, 1)
    
    R_mat, T_mat = np.meshgrid(r_grid, theta_grid, indexing='ij')
    R_flat, T_flat = R_mat.flatten(), T_mat.flatten()
    
    # 构建字典 A
    m = np.arange(M).reshape(-1, 1); n = np.arange(N).reshape(-1, 1)
    T_rad = np.deg2rad(T_flat)
    phi_tx = -4 * np.pi * cfg.delta_f * m * R_flat / cfg.c + 2 * np.pi * cfg.d * m * np.sin(T_rad) / cfg.wavelength
    a_tx = np.exp(1j * phi_tx)
    phi_rx = 2 * np.pi * cfg.d * n * np.sin(T_rad) / cfg.wavelength
    a_rx = np.exp(1j * phi_rx)
    A = (a_tx[:, np.newaxis, :] * a_rx[np.newaxis, :, :]).reshape(M*N, -1)
    
    # 列归一化
    A = A / np.sqrt(M*N)
    
    # 匹配
    correlations = np.abs(A.conj().T @ y)
    idx = np.argmax(correlations)
    
    return R_flat[idx], T_flat[idx]


# ==========================================
# 2. Benchmark 主程序
# ==========================================

def find_best_model_path(L_snapshots=None):
    """自动查找最佳模型权重文件"""
    L = L_snapshots or cfg.L_snapshots
    checkpoint_dir = cfg.checkpoint_dir
    candidates = []
    
    # 优先匹配特定 L 的模型
    pattern = f"{checkpoint_dir}/fda_cvnn_*_L{L}_best.pth"
    if glob.glob(pattern): 
        candidates.extend(glob.glob(pattern))
    candidates.append(f"{checkpoint_dir}/fda_cvnn_L{L}_best.pth")
    
    # 通用模型
    pattern_random = f"{checkpoint_dir}/fda_cvnn_*_Lrandom_best.pth"
    if glob.glob(pattern_random): 
        candidates.extend(glob.glob(pattern_random))
    candidates.append(f"{checkpoint_dir}/fda_cvnn_best.pth")
    
    for path in candidates:
        if os.path.exists(path): 
            return path
    return None


def load_cvnn_model(device, L_snapshots=None):
    """智能加载 CVNN 模型"""
    model_path = find_best_model_path(L_snapshots)
    
    if model_path is None or not os.path.exists(model_path):
        print(f"⚠️  未找到模型文件，使用默认初始化")
        return FDA_CVNN().to(device), None
    
    print(f"🔍 自动选择模型: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # 简易特征检测
        keys = list(state_dict.keys())
        has_dual = any('global_attn' in k for k in keys)
        has_far = any('attn' in k and 'conv_rr' in k for k in keys)
        has_se = any('attn' in k and '.fc.' in k for k in keys)
        
        if has_dual:
            model = FDA_CVNN_Attention(attention_type='dual').to(device)
        elif has_far:
            model = FDA_CVNN_Attention(attention_type='far').to(device)
        elif has_se:
            model = FDA_CVNN_Attention(attention_type='se').to(device)
        else:
            model = FDA_CVNN().to(device)
        
        # 修复 module. 前缀
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Loaded CVNN: {model_path}")
        return model, model_path
    except Exception as e:
        print(f"⚠️  加载模型失败: {e}")
        return FDA_CVNN().to(device), None


def run_legacy_benchmark(num_samples=200, L_snapshots=None):
    """运行 Legacy vs CVNN 对比实验
    
    Args:
        num_samples: 每个 SNR 点的测试样本数
        L_snapshots: 快拍数 (None 则使用 config 默认值)
    """
    # 设置快拍数
    L = L_snapshots if L_snapshots is not None else cfg.L_snapshots
    original_L = cfg.L_snapshots
    cfg.L_snapshots = L
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 运行 Benchmark: Legacy Methods vs CVNN")
    print(f"   Device: {device} | Snapshots L={L} | Samples={num_samples}")
    
    # 智能加载 CVNN 模型
    cvnn, model_path = load_cvnn_model(device, L_snapshots=L)
    cvnn.eval()
    
    snr_list = [-15, -10, -5, 0, 5, 10, 15]
    methods = ['MUSIC (Legacy)', 'ESPRIT (Legacy)', 'OMP (Legacy)', 'CVNN']
    results = {m: {'rmse_r': [], 'rmse_theta': []} for m in methods}
    results['CRB'] = {'rmse_r': [], 'rmse_theta': []}
    
    for snr in snr_list:
        errors = {m: {'r': [], 'theta': []} for m in methods}
        
        for _ in tqdm(range(num_samples), desc=f"SNR {snr}dB", leave=False):
            # 随机生成目标 (Off-grid, 非网格点)
            r_true = np.random.uniform(0, cfg.r_max)
            theta_true = np.random.uniform(cfg.theta_min, cfg.theta_max)
            
            # 生成数据
            R = generate_covariance_matrix(r_true, theta_true, snr)
            R_complex = R[0] + 1j * R[1]
            R_tensor = torch.FloatTensor(R).unsqueeze(0).to(device)
            
            # --- 1. MUSIC (Legacy) ---
            try:
                r_est, t_est = music_2d_legacy(R_complex)
                errors['MUSIC (Legacy)']['r'].append((r_est - r_true)**2)
                errors['MUSIC (Legacy)']['theta'].append((t_est - theta_true)**2)
            except: pass

            # --- 2. ESPRIT (Legacy) ---
            try:
                r_est, t_est = esprit_2d_legacy(R_complex)
                # ESPRIT Legacy 极其不稳定，如果 r_est < 0 (Matlab代码没处理)，算作大误差
                if r_est < 0: r_est = 0 
                # 防止极大值破坏绘图
                if abs(r_est - r_true) < 3000: 
                    errors['ESPRIT (Legacy)']['r'].append((r_est - r_true)**2)
                errors['ESPRIT (Legacy)']['theta'].append((t_est - theta_true)**2)
            except: pass

            # --- 3. OMP (Legacy) ---
            try:
                r_est, t_est = omp_2d_legacy(R_complex)
                errors['OMP (Legacy)']['r'].append((r_est - r_true)**2)
                errors['OMP (Legacy)']['theta'].append((t_est - theta_true)**2)
            except: pass
            
            # --- 4. CVNN ---
            with torch.no_grad():
                pred = cvnn(R_tensor).cpu().numpy()[0]
            # 反归一化
            r_est = pred[0] * cfg.r_max
            t_est = pred[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
            errors['CVNN']['r'].append((r_est - r_true)**2)
            errors['CVNN']['theta'].append((t_est - theta_true)**2)
            
        # 统计本 SNR 下的 RMSE
        for m in methods:
            if len(errors[m]['r']) > 0:
                results[m]['rmse_r'].append(np.sqrt(np.mean(errors[m]['r'])))
                results[m]['rmse_theta'].append(np.sqrt(np.mean(errors[m]['theta'])))
            else:
                results[m]['rmse_r'].append(np.nan)
                results[m]['rmse_theta'].append(np.nan)
                
        # 计算 CRB
        crb_r, crb_theta = compute_crb_average(snr, L=L)
        results['CRB']['rmse_r'].append(crb_r)
        results['CRB']['rmse_theta'].append(crb_theta)
    
    # 恢复原始配置
    cfg.L_snapshots = original_L
        
    return snr_list, results, L

# ==========================================
# 3. 绘图函数
# ==========================================
def plot_results(snr_list, results, L_snapshots=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    L = L_snapshots or cfg.L_snapshots
    
    # 颜色配置
    colors = {'MUSIC (Legacy)': 'red', 'ESPRIT (Legacy)': 'orange', 'OMP (Legacy)': 'green', 'CVNN': 'blue', 'CRB': 'black'}
    markers = {'MUSIC (Legacy)': 's', 'ESPRIT (Legacy)': 'x', 'OMP (Legacy)': 'd', 'CVNN': 'o', 'CRB': ''}
    styles = {'MUSIC (Legacy)': '--', 'ESPRIT (Legacy)': '--', 'OMP (Legacy)': '--', 'CVNN': '-', 'CRB': ':'}
    
    # 1. 距离 RMSE
    ax = axes[0]
    for m in results:
        if m == 'CRB':
            ax.plot(snr_list, results[m]['rmse_r'], color=colors[m], linestyle=styles[m], label='CRB', linewidth=2)
        else:
            ax.plot(snr_list, results[m]['rmse_r'], color=colors[m], marker=markers[m], linestyle=styles[m], label=m)
    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('RMSE Range (m)')
    ax.set_title(f'Range Estimation: CVNN vs Legacy (L={L})')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    
    # 2. 角度 RMSE
    ax = axes[1]
    for m in results:
        if m == 'CRB':
            ax.plot(snr_list, results[m]['rmse_theta'], color=colors[m], linestyle=styles[m], label='CRB', linewidth=2)
        else:
            ax.plot(snr_list, results[m]['rmse_theta'], color=colors[m], marker=markers[m], linestyle=styles[m], label=m)
    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('RMSE Angle (deg)')
    ax.set_title(f'Angle Estimation: CVNN vs Legacy (L={L})')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    save_path = f'results/benchmark_legacy_L{L}.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 图表已保存至: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='FDA-CVNN vs Legacy Methods Benchmark')
    parser.add_argument('--snapshots', '-L', type=int, default=None,
                        help='快拍数 L (默认使用 config 中的值)')
    parser.add_argument('--num-samples', '-n', type=int, default=200,
                        help='每个 SNR 点的测试样本数 (默认 200)')
    parser.add_argument('--no-plot', action='store_true',
                        help='不显示绘图窗口')
    args = parser.parse_args()
    
    snr_list, results, L = run_legacy_benchmark(
        num_samples=args.num_samples,
        L_snapshots=args.snapshots
    )
    
    if not args.no_plot:
        plot_results(snr_list, results, L_snapshots=L)
    
    return snr_list, results, L


if __name__ == "__main__":
    main()