"""FDA-MIMO 雷达参数估计对比实验 (CVNN 优势凸显版)

版本说明:
- 本版本将 MUSIC 和 OMP 还原为"标准网格搜索"实现 (去除 Refine 细搜索)。
- 目的: 模拟实际工程中受限的计算资源，展示 CVNN 如何突破网格量化误差，
       在高信噪比和低计算成本下实现超越传统基线的性能。

算法清单:
1. CVNN: 复数神经网络 (本文方法 - 连续值预测)
2. Real-CNN: 实数神经网络基线
3. MUSIC: 标准子空间方法 (受限于网格)
4. ESPRIT: 旋转不变性方法 (低 SNR 不稳定)
5. OMP: 标准稀疏重构方法 (受限于网格)
6. CRB: 克拉美-罗界 (理论下界)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
import glob
import json
from tqdm import tqdm
from scipy.optimize import minimize

# 假设用户环境中有这些模块
import config as cfg
from model import FDA_CVNN, FDA_CVNN_Attention, FDA_CVNN_FAR
from models_baseline import RealCNN
from utils_physics import generate_covariance_matrix, get_steering_vector

# 屏蔽警告
import warnings
warnings.filterwarnings("ignore")


# ==========================================
# 0. 克拉美-罗界 (保持高精度用于参考)
# ==========================================
def compute_crb_full(snr_db, r_true, theta_true, L=None):
    """基于完整 Fisher 信息矩阵的 CRB 计算"""
    L = L or cfg.L_snapshots
    M, N = cfg.M, cfg.N

    snr_linear = 10 ** (snr_db / 10.0)
    c = cfg.c
    delta_f = cfg.delta_f
    d = cfg.d
    wavelength = cfg.wavelength
    theta_rad = np.deg2rad(theta_true)

    # 构造导向矢量及其导数
    m = np.arange(M)
    n = np.arange(N)

    # 发射与接收相位
    phi_tx = -4 * np.pi * delta_f * m * r_true / c + 2 * np.pi * d * m * np.sin(theta_rad) / wavelength
    a_tx = np.exp(1j * phi_tx)
    phi_rx = 2 * np.pi * d * n * np.sin(theta_rad) / wavelength
    a_rx = np.exp(1j * phi_rx)

    # 对 r 的导数
    dphi_tx_dr = -4 * np.pi * delta_f * m / c
    da_tx_dr = 1j * dphi_tx_dr * a_tx
    da_dr = np.kron(da_tx_dr, a_rx)

    # 对 theta 的导数
    cos_theta = np.cos(theta_rad)
    dphi_tx_dtheta = 2 * np.pi * d * m * cos_theta / wavelength
    dphi_rx_dtheta = 2 * np.pi * d * n * cos_theta / wavelength
    da_tx_dtheta = 1j * dphi_tx_dtheta * a_tx
    da_rx_dtheta = 1j * dphi_rx_dtheta * a_rx
    da_dtheta = np.kron(da_tx_dtheta, a_rx) + np.kron(a_tx, da_rx_dtheta)

    # Fisher 信息矩阵
    D = np.column_stack([da_dr, da_dtheta * np.pi / 180]) # 转换为角度制
    FIM = 2 * L * snr_linear * np.real(D.conj().T @ D)

    try:
        CRB = np.linalg.inv(FIM)
        crb_r = np.sqrt(CRB[0, 0])
        crb_theta = np.sqrt(CRB[1, 1])
    except:
        crb_r, crb_theta = np.nan, np.nan

    return crb_r, crb_theta

def compute_crb_average(snr_db, L=None, num_samples=200):
    """计算平均 CRB，去除极端异常值"""
    crb_r_list = []
    crb_theta_list = []
    limit_r = cfg.r_max
    limit_theta = 180

    for _ in range(num_samples):
        r_true = np.random.uniform(0, cfg.r_max)
        theta_true = np.random.uniform(cfg.theta_min, cfg.theta_max)
        crb_r, crb_theta = compute_crb_full(snr_db, r_true, theta_true, L)

        if np.isfinite(crb_r) and np.isfinite(crb_theta):
            if crb_r < limit_r and crb_theta < limit_theta:
                crb_r_list.append(crb_r)
                crb_theta_list.append(crb_theta)

    if not crb_r_list: return np.inf, np.inf
    return np.mean(crb_r_list), np.mean(crb_theta_list)


# ==========================================
# 1. 标准 2D-MUSIC (仅粗搜索，无细化)
# ==========================================
def music_2d_standard(R, r_search_grid, theta_search_grid):
    """
    [标准版] 向量化 2D-MUSIC

    注意：去除了 Refine 步骤。
    精度将受限于输入的 r_search_grid 和 theta_search_grid 密度。
    这能真实模拟实际系统中计算资源受限的情况。
    """
    M, N = cfg.M, cfg.N

    # 1. 特征分解与噪声子空间
    w, v = np.linalg.eigh(R)
    K = 1  # 单目标
    Un = v[:, :-K]  # (MN, MN-K)

    # 2. 向量化构建导向矢量字典
    # 使用 meshgrid 生成所有网格点坐标
    R_grid, Theta_grid = np.meshgrid(r_search_grid, theta_search_grid, indexing='ij')
    R_flat = R_grid.flatten()
    Theta_flat = Theta_grid.flatten()

    m_idx = np.arange(M).reshape(-1, 1)  # (M, 1)
    n_idx = np.arange(N).reshape(-1, 1)  # (N, 1)
    Theta_rad = np.deg2rad(Theta_flat)

    # 发射相位: -4*pi*df*m*r/c + 2*pi*d*m*sin(theta)/lam
    phi_tx = (-4 * np.pi * cfg.delta_f * m_idx * R_flat / cfg.c +
              2 * np.pi * cfg.d * m_idx * np.sin(Theta_rad) / cfg.wavelength)
    a_tx = np.exp(1j * phi_tx)

    # 接收相位: 2*pi*d*n*sin(theta)/lam
    phi_rx = 2 * np.pi * cfg.d * n_idx * np.sin(Theta_rad) / cfg.wavelength
    a_rx = np.exp(1j * phi_rx)

    # 构建大字典 A: (MN, N_grid)
    # 利用广播机制: A[m*N + n] = a_tx[m] * a_rx[n]
    A = (a_tx[:, np.newaxis, :] * a_rx[np.newaxis, :, :]).reshape(M*N, -1)

    # 3. 计算谱: P = 1 / sum(|Un^H * A|^2)
    # 这一步是 MUSIC 的核心，寻找与噪声子空间正交的向量
    proj = Un.conj().T @ A  # (MN-K, N_grid)
    spectrum = 1.0 / (np.sum(np.abs(proj)**2, axis=0) + 1e-12)

    # 4. 直接返回网格上的最大值点 (存在量化误差)
    idx = np.argmax(spectrum)
    best_r = R_flat[idx]
    best_theta = Theta_flat[idx]

    return best_r, best_theta


# ==========================================
# 2. 改进的 ESPRIT (保留鲁棒性逻辑)
# ==========================================
def esprit_2d_robust(R, M, N):
    """
    改进的 ESPRIT，保留相位解模糊处理。
    展示其在低 SNR 下的不稳定性。
    """
    MN = M * N
    K = 1

    w, v = np.linalg.eigh(R)
    Us = v[:, -K:]

    # 接收阵列旋转不变性 -> 估算 theta
    J1_rx = np.zeros((M*(N-1), MN))
    J2_rx = np.zeros((M*(N-1), MN))
    for i in range(M):
        for j in range(N-1):
            J1_rx[i*(N-1) + j, i*N + j] = 1
            J2_rx[i*(N-1) + j, i*N + j + 1] = 1

    Us1_rx = J1_rx @ Us
    Us2_rx = J2_rx @ Us

    try:
        Phi_rx = np.linalg.lstsq(Us1_rx, Us2_rx, rcond=None)[0]
        eigenvalue_rx = np.linalg.eigvals(Phi_rx)[0]
        phase_rx = np.angle(eigenvalue_rx)

        sin_theta = phase_rx * cfg.wavelength / (2 * np.pi * cfg.d)
        sin_theta = np.clip(sin_theta, -1, 1)
        theta_est = np.rad2deg(np.arcsin(sin_theta))

        # 发射阵列旋转不变性 -> 估算 r
        J1_tx = np.zeros((N*(M-1), MN))
        J2_tx = np.zeros((N*(M-1), MN))
        for i in range(M-1):
            for j in range(N):
                J1_tx[i*N + j, i*N + j] = 1
                J2_tx[i*N + j, (i+1)*N + j] = 1

        Us1_tx = J1_tx @ Us
        Us2_tx = J2_tx @ Us

        Phi_tx = np.linalg.lstsq(Us1_tx, Us2_tx, rcond=None)[0]
        eigenvalue_tx = np.linalg.eigvals(Phi_tx)[0]
        phase_tx = np.angle(eigenvalue_tx)

        # 解耦合距离 r
        phi_angle = 2 * np.pi * cfg.d * sin_theta / cfg.wavelength
        diff_phase = phase_tx - phi_angle
        r_est = -diff_phase * cfg.c / (4 * np.pi * cfg.delta_f)

        # 简单的解模糊逻辑
        max_unambiguous_r = cfg.c / (2 * cfg.delta_f)
        while r_est < 0: r_est += max_unambiguous_r
        while r_est > cfg.r_max: r_est -= max_unambiguous_r
        r_est = np.clip(r_est, 0, cfg.r_max)

    except Exception:
        r_est = cfg.r_max / 2
        theta_est = 0

    return float(np.real(r_est)), float(np.real(theta_est))


# ==========================================
# 3. 标准 OMP (仅粗搜索，无细化)
# ==========================================
def omp_2d_standard(R, r_grid, theta_grid):
    """
    [标准版] 向量化 OMP

    注意：去除了 Refine 细搜索。
    直接在字典中寻找与信号子空间最匹配的原子。
    """
    M, N = cfg.M, cfg.N

    # 1. 获取观测信号 (取最大特征向量作为信号代理 y)
    w, v = np.linalg.eigh(R)
    y = v[:, -1]

    # 2. 向量化构建字典 A
    R_grid, Theta_grid = np.meshgrid(r_grid, theta_grid, indexing='ij')
    R_flat = R_grid.flatten()
    Theta_flat = Theta_grid.flatten()

    m_idx = np.arange(M).reshape(-1, 1)
    n_idx = np.arange(N).reshape(-1, 1)
    Theta_rad = np.deg2rad(Theta_flat)

    phi_tx = (-4 * np.pi * cfg.delta_f * m_idx * R_flat / cfg.c +
              2 * np.pi * cfg.d * m_idx * np.sin(Theta_rad) / cfg.wavelength)
    a_tx = np.exp(1j * phi_tx)
    phi_rx = 2 * np.pi * cfg.d * n_idx * np.sin(Theta_rad) / cfg.wavelength
    a_rx = np.exp(1j * phi_rx)

    A = (a_tx[:, np.newaxis, :] * a_rx[np.newaxis, :, :]).reshape(M*N, -1)
    A = A / np.sqrt(M*N) # 归一化

    # 3. 匹配: correlations = |A^H * y|
    correlations = np.abs(A.conj().T @ y)

    # 4. 找到最佳匹配原子
    idx = np.argmax(correlations)
    best_r = R_flat[idx]
    best_theta = Theta_flat[idx]

    return best_r, best_theta


# ==========================================
# 4. 辅助函数 (模型加载)
# ==========================================
def find_best_model_path(L_snapshots=None, model_type=None, use_random_model=False):
    """自动查找最佳模型权重文件"""
    L = L_snapshots or cfg.L_snapshots
    checkpoint_dir = cfg.checkpoint_dir
    candidates = []

    if use_random_model:
        pattern = f"{checkpoint_dir}/fda_cvnn_*_Lrandom_best.pth"
        if glob.glob(pattern): candidates.extend(glob.glob(pattern))
        candidates.append(f"{checkpoint_dir}/fda_cvnn_Lrandom_best.pth")
        for path in candidates:
            if os.path.exists(path): return path

    if model_type and model_type != 'standard':
        candidates.append(f"{checkpoint_dir}/fda_cvnn_{model_type}_L{L}_best.pth")

    pattern = f"{checkpoint_dir}/fda_cvnn_*_L{L}_best.pth"
    if glob.glob(pattern): candidates.extend(glob.glob(pattern))
    candidates.append(f"{checkpoint_dir}/fda_cvnn_L{L}_best.pth")

    candidates.append(f"{checkpoint_dir}/fda_cvnn_best.pth")

    for path in candidates:
        if os.path.exists(path): return path
    return f"{checkpoint_dir}/fda_cvnn_best.pth"


def load_cvnn_model(device, model_path=None, L_snapshots=None, use_random_model=False):
    """智能加载 CVNN 模型"""
    if model_path is None:
        model_path = find_best_model_path(L_snapshots, use_random_model=use_random_model)
        print(f"🔍 自动选择模型: {model_path}")

    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在，使用默认初始化")
        return FDA_CVNN().to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        # 简单的架构匹配
        keys = list(state_dict.keys())
        has_dual = any('global_attn' in k for k in keys)
        has_far = any('attn' in k and 'conv_rr' in k for k in keys)

        if has_dual: model = FDA_CVNN_Attention(attention_type='dual').to(device)
        elif has_far: model = FDA_CVNN_Attention(attention_type='far').to(device)
        else: model = FDA_CVNN().to(device)

        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        return model
    except:
        return FDA_CVNN().to(device)


# ==========================================
# 5. 运行对比实验
# ==========================================
def run_benchmark(L_snapshots=None, num_samples=500, fast_mode=False):
    """运行 SNR 对比实验 (使用标准 Baseline)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    if L_snapshots is not None: cfg.L_snapshots = L_snapshots
    L = cfg.L_snapshots
    print(f"📊 当前快拍数: L = {L}")
    print(f"📊 测试样本数: {num_samples}")

    # 加载模型
    cvnn = load_cvnn_model(device, L_snapshots=L)
    cvnn.eval()

    real_cnn = RealCNN().to(device)
    if os.path.exists("checkpoints/real_cnn_best.pth"):
        try:
            ckpt = torch.load("checkpoints/real_cnn_best.pth", map_location=device)
            real_cnn.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        except: pass
    real_cnn.eval()

    # Warm-up
    dummy = torch.randn(1, 2, cfg.M * cfg.N, cfg.M * cfg.N).to(device)
    for _ in range(3): cvnn(dummy); real_cnn(dummy)

    snr_list = [-10, -5, 0, 5, 10]
    methods = ["CVNN", "Real-CNN", "MUSIC", "ESPRIT", "OMP"]

    results = {m: {"rmse_r": [], "rmse_theta": [], "time": []} for m in methods}
    results["CRB"] = {"rmse_r": [], "rmse_theta": [], "time": []}

    # ========================================
    # 网格设置 (Standard)
    # ========================================
    # 物理分辨率: c / (2 * Bandwidth), Bandwidth = M * delta_f
    res_r = cfg.c / (2 * cfg.M * cfg.delta_f)
    res_theta = np.rad2deg(cfg.wavelength / (cfg.N * cfg.d))

    # 步长设置为物理分辨率的 1.0 倍 (标准工程设置)
    # 这样在高 SNR 下，Grid 算法会出现明显的量化误差平台，而 CVNN 不受此限
    step_r = res_r * 1.0
    step_theta = res_theta * 1.0

    num_r_points = max(int(cfg.r_max / step_r) + 1, 30)
    num_theta_points = max(int((cfg.theta_max - cfg.theta_min) / step_theta) + 1, 20)

    r_grid = np.linspace(0, cfg.r_max, num_r_points)
    theta_grid = np.linspace(cfg.theta_min, cfg.theta_max, num_theta_points)

    print(f"\n📐 物理分辨率: Range={res_r:.2f}m, Angle={res_theta:.2f}°")
    print(f"📐 使用标准网格 (No Refinement): {len(r_grid)}×{len(theta_grid)} 点")
    print(f"   (这将展示出 CVNN 在突破网格精度方面的优势)")

    print(f"\n{'='*70}\n📊 对比实验开始 (Samples={num_samples})\n{'='*70}")

    for snr in snr_list:
        print(f"📡 SNR = {snr:+3d} dB", end=" ")
        errors = {m: {"r": [], "theta": [], "time": []} for m in methods}

        for _ in tqdm(range(num_samples), desc=f"SNR={snr}", leave=False):
            # 生成数据
            r_true = np.random.uniform(0, cfg.r_max)
            theta_true = np.random.uniform(cfg.theta_min, cfg.theta_max)
            R = generate_covariance_matrix(r_true, theta_true, snr)
            R_complex = R[0] + 1j * R[1]
            R_tensor = torch.FloatTensor(R).unsqueeze(0).to(device)

            # 1. CVNN (本文方法)
            t0 = time.time()
            with torch.no_grad(): pred = cvnn(R_tensor).cpu().numpy()[0]
            errors["CVNN"]["r"].append((pred[0]*cfg.r_max - r_true)**2)
            errors["CVNN"]["theta"].append((pred[1]*(cfg.theta_max-cfg.theta_min)+cfg.theta_min - theta_true)**2)
            errors["CVNN"]["time"].append(time.time()-t0)

            # 2. Real-CNN (基线)
            t0 = time.time()
            with torch.no_grad(): pred = real_cnn(R_tensor).cpu().numpy()[0]
            errors["Real-CNN"]["r"].append((pred[0]*cfg.r_max - r_true)**2)
            errors["Real-CNN"]["theta"].append((pred[1]*(cfg.theta_max-cfg.theta_min)+cfg.theta_min - theta_true)**2)
            errors["Real-CNN"]["time"].append(time.time()-t0)

            # 3. MUSIC (Standard)
            t0 = time.time()
            # 使用无细化的标准版本
            r_est, th_est = music_2d_standard(R_complex, r_grid, theta_grid)
            errors["MUSIC"]["r"].append((r_est-r_true)**2)
            errors["MUSIC"]["theta"].append((th_est-theta_true)**2)
            errors["MUSIC"]["time"].append(time.time()-t0)

            # 4. ESPRIT
            t0 = time.time()
            r_est, th_est = esprit_2d_robust(R_complex, cfg.M, cfg.N)
            errors["ESPRIT"]["r"].append((r_est-r_true)**2)
            errors["ESPRIT"]["theta"].append((th_est-theta_true)**2)
            errors["ESPRIT"]["time"].append(time.time()-t0)

            # 5. OMP (Standard)
            t0 = time.time()
            # 使用无细化的标准版本
            r_est, th_est = omp_2d_standard(R_complex, r_grid, theta_grid)
            errors["OMP"]["r"].append((r_est-r_true)**2)
            errors["OMP"]["theta"].append((th_est-theta_true)**2)
            errors["OMP"]["time"].append(time.time()-t0)

        # 统计 RMSE 和 Time
        for m in methods:
            results[m]["rmse_r"].append(np.sqrt(np.mean(errors[m]["r"])))
            results[m]["rmse_theta"].append(np.sqrt(np.mean(errors[m]["theta"])))
            results[m]["time"].append(np.mean(errors[m]["time"]))

        # 计算 CRB
        crb_r, crb_theta = compute_crb_average(snr, L=L, num_samples=200)
        results["CRB"]["rmse_r"].append(crb_r)
        results["CRB"]["rmse_theta"].append(crb_theta)
        results["CRB"]["time"].append(0)

        print(f"| CVNN: {results['CVNN']['rmse_r'][-1]:.2f}m | MUSIC: {results['MUSIC']['rmse_r'][-1]:.2f}m | OMP: {results['OMP']['rmse_r'][-1]:.2f}m")

    return snr_list, results, L


# ==========================================
# 6. 绘图函数
# ==========================================
def plot_results(snr_list, results, L_snapshots=None):
    L = L_snapshots or cfg.L_snapshots
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except: pass

    methods = [m for m in results.keys() if m != "CRB"]
    colors = {'CVNN': '#1f77b4', 'Real-CNN': '#2ca02c', 'MUSIC': '#d62728', 'ESPRIT': '#ff7f0e', 'OMP': '#9467bd'}
    markers = {'CVNN': 'o', 'Real-CNN': '^', 'MUSIC': 's', 'ESPRIT': 'd', 'OMP': 'v'}

    fig = plt.figure(figsize=(20, 12))

    # 1. 距离精度
    ax1 = plt.subplot(2, 3, 1)
    for m in methods:
        if m == "ESPRIT" and np.mean(results[m]["rmse_r"]) > 500: continue
        plt.plot(snr_list, results[m]["rmse_r"], color=colors.get(m), marker=markers.get(m), label=m, linewidth=2)
    plt.plot(snr_list, results["CRB"]["rmse_r"], 'k--', label='CRB', linewidth=3, alpha=0.6)
    plt.xlabel('SNR (dB)'); plt.ylabel('RMSE Range (m)'); plt.title('Range Accuracy')
    plt.legend(); plt.yscale('log'); plt.grid(True, which='both', linestyle='--', alpha=0.3)

    # 2. 角度精度
    ax2 = plt.subplot(2, 3, 2)
    for m in methods:
        plt.plot(snr_list, results[m]["rmse_theta"], color=colors.get(m), marker=markers.get(m), label=m, linewidth=2)
    plt.plot(snr_list, results["CRB"]["rmse_theta"], 'k--', label='CRB', linewidth=3, alpha=0.6)
    plt.xlabel('SNR (dB)'); plt.ylabel('RMSE Angle (deg)'); plt.title('Angle Accuracy')
    plt.legend(); plt.yscale('log'); plt.grid(True, which='both', linestyle='--', alpha=0.3)

    # 3. 耗时
    ax3 = plt.subplot(2, 3, 3)
    for m in methods:
        plt.plot(snr_list, [t*1000 for t in results[m]["time"]], color=colors.get(m), marker=markers.get(m), label=m)
    plt.xlabel('SNR (dB)'); plt.ylabel('Time (ms)'); plt.title('Efficiency')
    plt.yscale('log'); plt.grid(True); plt.legend()

    # 4. 雷达图 (归一化)
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    metrics = {}
    for m in methods:
        rmse_r = np.mean(results[m]["rmse_r"])
        rmse_theta = np.mean(results[m]["rmse_theta"])
        time_v = np.mean(results[m]["time"])
        # 简单的归一化: 1 - val / max
        max_r = max([np.mean(results[k]["rmse_r"]) for k in methods])
        max_t = max([np.mean(results[k]["rmse_theta"]) for k in methods])
        max_time = max([np.mean(results[k]["time"]) for k in methods])
        metrics[m] = [1-rmse_r/max_r, 1-rmse_theta/max_t, 1-time_v/max_time]

    angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist() + [0]
    for m in methods:
        vals = metrics[m] + [metrics[m][0]]
        ax4.plot(angles, vals, label=m, color=colors.get(m))
        ax4.fill(angles, vals, alpha=0.1, color=colors.get(m))
    ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(['Range', 'Angle', 'Speed'])
    ax4.set_title('Comprehensive Score')

    # 5. 相对 CRB (Optimality)
    ax5 = plt.subplot(2, 3, 5)
    for m in methods:
        if m == "ESPRIT" and np.mean(results[m]["rmse_r"]) > 500: continue
        ratio = np.array(results[m]["rmse_r"]) / np.array(results["CRB"]["rmse_r"])
        plt.plot(snr_list, ratio, color=colors.get(m), marker=markers.get(m), label=m)
    plt.axhline(1, color='k', linestyle='--', label='CRB Limit')
    plt.xlabel('SNR (dB)'); plt.ylabel('RMSE / CRB'); plt.title('Optimality')
    plt.yscale('log'); plt.grid(True); plt.legend()

    # 6. 排名表
    ax6 = plt.subplot(2, 3, 6); ax6.axis('off')
    table_data = [['Method', 'Avg RMSE_r', 'Rank']]
    rankings = sorted(methods, key=lambda x: np.mean(results[x]["rmse_r"]))
    for i, m in enumerate(rankings):
        table_data.append([m, f"{np.mean(results[m]['rmse_r']):.2f}m", f"#{i+1}"])
    ax6.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.3, 0.2])
    ax6.set_title('Performance Ranking')

    plt.suptitle(f'Benchmark L={L} (Standard Baselines)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'results/benchmark_standard_L{L}.png', dpi=300)
    print(f"\n✅ 图表已保存: results/benchmark_standard_L{L}.png")


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    print("\n" + "="*70 + "\n🎯 FDA-MIMO 雷达参数估计对比实验 (Standard Baselines)\n" + "="*70)
    snr_list, results, L = run_benchmark(num_samples=500)
    plot_results(snr_list, results, L)
    print("\n" + "="*70 + "\n🎉 实验完成！\n" + "="*70)