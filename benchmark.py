"""FDA-MIMO 雷达参数估计对比实验
算法清单:
1. CVNN: 复数神经网络 (本文方法)
2. Real-CNN: 实数神经网络基线
3. MUSIC: 子空间方法 (两级搜索)
4. ESPRIT: 旋转不变性方法
5. OMP: 稀疏重构方法
6. CRB: 克拉美-罗界 (理论下界)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
from tqdm import tqdm

import config as cfg
from model import FDA_CVNN, FDA_CVNN_Attention, FDA_CVNN_FAR
from models_baseline import RealCNN
from utils_physics import generate_covariance_matrix, get_steering_vector


# ==========================================
# 0. 克拉美-罗界 (完整 FIM 版本)
# ==========================================
def compute_crb_full(snr_db, r_true, theta_true, L=None):
    """
    基于完整 Fisher 信息矩阵的 CRB 计算
    考虑距离-角度耦合效应
    """
    L = L or cfg.L_snapshots
    M = cfg.M
    N = cfg.N
    MN = M * N

    snr_linear = 10 ** (snr_db / 10.0)
    sigma2 = 1.0 / snr_linear

    c = cfg.c
    delta_f = cfg.delta_f
    d = cfg.d
    wavelength = cfg.wavelength

    theta_rad = np.deg2rad(theta_true)

    # 构造导向矢量及其导数
    m = np.arange(M)
    n = np.arange(N)

    phi_tx = -4 * np.pi * delta_f * m * r_true / c + 2 * np.pi * d * m * np.sin(theta_rad) / wavelength
    a_tx = np.exp(1j * phi_tx)

    phi_rx = 2 * np.pi * d * n * np.sin(theta_rad) / wavelength
    a_rx = np.exp(1j * phi_rx)

    a = np.kron(a_tx, a_rx)

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
    D = np.column_stack([da_dr, da_dtheta * np.pi / 180])
    FIM = 2 * L * snr_linear * np.real(D.conj().T @ D)

    try:
        CRB = np.linalg.inv(FIM)
        crb_r = np.sqrt(CRB[0, 0])
        crb_theta = np.sqrt(CRB[1, 1])
    except:
        crb_r = np.inf
        crb_theta = np.inf

    return crb_r, crb_theta


def compute_crb_average(snr_db, L=None, num_samples=30):
    """计算多个随机目标位置的平均 CRB"""
    crb_r_list = []
    crb_theta_list = []

    for _ in range(num_samples):
        r_true = np.random.uniform(0, cfg.r_max)
        theta_true = np.random.uniform(cfg.theta_min, cfg.theta_max)
        crb_r, crb_theta = compute_crb_full(snr_db, r_true, theta_true, L)
        if np.isfinite(crb_r) and np.isfinite(crb_theta):
            crb_r_list.append(crb_r)
            crb_theta_list.append(crb_theta)

    return np.mean(crb_r_list), np.mean(crb_theta_list)


# ==========================================
# 1. 改进的 2D-MUSIC (两级搜索)
# ==========================================
def music_2d_refined(R, r_search_coarse, theta_search_coarse, refine=True):
    """
    两级 MUSIC 算法
    1. 粗网格搜索
    2. 局部细化搜索
    """
    # 特征分解
    w, v = np.linalg.eigh(R)
    idx = np.argsort(w)
    v = v[:, idx]

    Un = v[:, :-1]

    def compute_music_spectrum(r, theta):
        """计算 MUSIC 谱值"""
        a = get_steering_vector(r, theta)
        proj = Un.conj().T @ a
        denom = np.sum(np.abs(proj)**2)
        return 1.0 / (denom + 1e-12)

    # === 第一步：粗搜索 ===
    max_p = -1
    best_r = 0
    best_theta = 0

    for r in r_search_coarse:
        for theta in theta_search_coarse:
            spectrum = compute_music_spectrum(r, theta)
            if spectrum > max_p:
                max_p = spectrum
                best_r = r
                best_theta = theta

    if not refine:
        return best_r, best_theta

    # === 第二步：细搜索 ===
    r_step = r_search_coarse[1] - r_search_coarse[0] if len(r_search_coarse) > 1 else 50
    theta_step = theta_search_coarse[1] - theta_search_coarse[0] if len(theta_search_coarse) > 1 else 2

    # 在粗估计结果附近生成细网格 (10倍精度)
    r_fine = np.linspace(max(0, best_r - r_step),
                         min(cfg.r_max, best_r + r_step), 41)
    theta_fine = np.linspace(max(cfg.theta_min, best_theta - theta_step),
                             min(cfg.theta_max, best_theta + theta_step), 41)

    max_p = -1
    for r in r_fine:
        for theta in theta_fine:
            spectrum = compute_music_spectrum(r, theta)
            if spectrum > max_p:
                max_p = spectrum
                best_r = r
                best_theta = theta

    return best_r, best_theta


# ==========================================
# 2. 改进的 ESPRIT (相位解模糊)
# ==========================================
def esprit_2d_robust(R, M, N):
    """
    改进的 ESPRIT，添加相位解模糊处理
    """
    MN = M * N
    K = 1

    w, v = np.linalg.eigh(R)
    Us = v[:, -K:]

    # 接收维度选择矩阵
    J1_rx = np.zeros((M*(N-1), MN))
    J2_rx = np.zeros((M*(N-1), MN))
    for i in range(M):
        for j in range(N-1):
            J1_rx[i*(N-1) + j, i*N + j] = 1
            J2_rx[i*(N-1) + j, i*N + j + 1] = 1

    Us1_rx = J1_rx @ Us
    Us2_rx = J2_rx @ Us

    try:
        # === Step 1: 从接收维度估计角度 ===
        Phi_rx = np.linalg.lstsq(Us1_rx, Us2_rx, rcond=None)[0]
        eigenvalue_rx = np.linalg.eigvals(Phi_rx)[0]
        phase_rx = np.angle(eigenvalue_rx)

        sin_theta = phase_rx * cfg.wavelength / (2 * np.pi * cfg.d)
        sin_theta = np.clip(sin_theta, -1, 1)
        theta_est = np.rad2deg(np.arcsin(sin_theta))

        # === Step 2: 从发射维度估计距离 ===
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

        # 从发射相位中扣除角度贡献
        phi_angle = 2 * np.pi * cfg.d * sin_theta / cfg.wavelength
        diff_phase = phase_tx - phi_angle

        # 计算距离
        r_est = -diff_phase * cfg.c / (4 * np.pi * cfg.delta_f)

        # === 相位解模糊 ===
        max_unambiguous_r = cfg.c / (2 * cfg.delta_f)

        # 周期性调整
        while r_est < 0:
            r_est += max_unambiguous_r
        while r_est > cfg.r_max:
            r_est -= max_unambiguous_r

        r_est = np.clip(r_est, 0, cfg.r_max)

    except Exception as e:
        # 如果失败，返回中间值
        r_est = cfg.r_max / 2
        theta_est = 0

    return float(np.real(r_est)), float(np.real(theta_est))


# ==========================================
# 3. OMP (归一化字典)
# ==========================================
def omp_2d(R, r_grid, theta_grid, K=1):
    """
    正交匹配追踪，字典原子已归一化
    """
    MN = cfg.M * cfg.N

    w, v = np.linalg.eigh(R)
    y = v[:, -1]
    y = y / (np.linalg.norm(y) + 1e-12)

    num_r = len(r_grid)
    num_theta = len(theta_grid)
    A = np.zeros((MN, num_r * num_theta), dtype=complex)

    # 构造归一化字典
    for i, r in enumerate(r_grid):
        for j, theta in enumerate(theta_grid):
            a = get_steering_vector(r, theta)
            A[:, i * num_theta + j] = a / (np.linalg.norm(a) + 1e-12)

    residual = y.copy()
    support = []

    for _ in range(K):
        correlations = np.abs(A.conj().T @ residual)
        best_idx = np.argmax(correlations)
        support.append(best_idx)

        A_s = A[:, support]
        x_s = np.linalg.lstsq(A_s, y, rcond=None)[0]
        residual = y - A_s @ x_s

    best_idx = support[0]
    r_idx = best_idx // num_theta
    theta_idx = best_idx % num_theta

    return r_grid[r_idx], theta_grid[theta_idx]


# ==========================================
# 5. 运行对比实验
# ==========================================
def load_cvnn_model(device, model_path="checkpoints/fda_cvnn_best.pth"):
    """
    智能加载 CVNN 模型，自动检测模型类型
    
    支持的模型类型:
    - FDA_CVNN: 标准 CVNN (无注意力模块)
    - FDA_CVNN_Attention (SE): 有 attn*.fc.* 层
    - FDA_CVNN_Attention (CBAM): 有 channel_attn 层
    - FDA_CVNN_FAR: 有 attn*.conv1.conv_rr.* 层 (复数卷积做注意力)
    """
    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        return FDA_CVNN().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 获取 state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            model_type = checkpoint.get('model_type', None)
        else:
            state_dict = checkpoint
            model_type = None
        
        # 如果有保存的 model_type，直接使用
        if model_type:
            print(f"🔍 检测到保存的模型类型: {model_type}")
            if model_type == 'far':
                model = FDA_CVNN_FAR().to(device)
            elif model_type == 'cbam':
                model = FDA_CVNN_Attention(use_cbam=True).to(device)
            elif model_type in ['attention', 'se']:
                model = FDA_CVNN_Attention(use_cbam=False).to(device)
            else:
                model = FDA_CVNN().to(device)
        else:
            # 通过 state_dict 的 key 推断模型类型
            keys = list(state_dict.keys())
            
            # FAR 特征: attn*.conv1.conv_rr (复数卷积层做注意力)
            has_far = any('attn' in k and 'conv1.conv_rr' in k for k in keys)
            # SE 特征: attn*.fc.* (全连接层做注意力)
            has_se = any('attn' in k and '.fc.' in k for k in keys)
            # CBAM 特征: channel_attn (SE + 空间注意力)
            has_cbam = any('channel_attn' in k for k in keys)
            
            if has_far:
                model = FDA_CVNN_FAR().to(device)
                print("🔍 检测到 FAR 模型结构 (局部池化注意力)")
            elif has_cbam:
                model = FDA_CVNN_Attention(use_cbam=True).to(device)
                print("🔍 检测到 CBAM 注意力模型结构")
            elif has_se:
                model = FDA_CVNN_Attention(use_cbam=False).to(device)
                print("🔍 检测到 SE 注意力模型结构 (通道注意力)")
            else:
                model = FDA_CVNN().to(device)
                print("🔍 检测到标准 CVNN 模型结构")
        
        # 加载权重
        model.load_state_dict(state_dict)
        print(f"✅ CVNN 模型加载成功 (参数量: {model.count_parameters():,})")
        return model
        
    except Exception as e:
        print(f"⚠️  CVNN 加载失败: {e}")
        print("   使用默认 FDA_CVNN 模型")
        return FDA_CVNN().to(device)


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 智能加载 CVNN 模型
    cvnn = load_cvnn_model(device)
    cvnn.eval()

    real_cnn = RealCNN().to(device)
    real_cnn_path = "checkpoints/real_cnn_best.pth"
    if os.path.exists(real_cnn_path):
        try:
            checkpoint = torch.load(real_cnn_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                real_cnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                real_cnn.load_state_dict(checkpoint)
            print(f"✅ Real-CNN 模型加载成功")
        except:
            print(f"⚠️  Real-CNN 使用随机权重")
    real_cnn.eval()

    # ========== GPU 预热 (Warm-up) ==========
    print("🔥 正在预热 GPU (Warm-up)...")
    # 生成 dummy input，形状与真实数据一致
    dummy_input = torch.randn(1, 2, cfg.M * cfg.N, cfg.M * cfg.N).to(device)
    
    # 强制让两个网络都空跑几次，消除冷启动开销
    with torch.no_grad():
        for _ in range(10):
            _ = cvnn(dummy_input)
            _ = real_cnn(dummy_input)
    
    # 强制同步 GPU，确保预热完成
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print("✅ 预热完成，开始正式测试...")

    # 参数设置
    snr_list = [-10, -5, 0, 5, 10]
    num_samples = 50

    methods = ["CVNN", "Real-CNN", "MUSIC", "ESPRIT", "OMP"]
    results = {m: {"rmse_r": [], "rmse_theta": [], "time": []} for m in methods}
    results["CRB"] = {"rmse_r": [], "rmse_theta": [], "time": []}

    # 搜索网格
    r_grid = np.linspace(0, cfg.r_max, 100)
    theta_grid = np.linspace(cfg.theta_min, cfg.theta_max, 60)

    r_grid_omp = np.linspace(0, cfg.r_max, 100)
    theta_grid_omp = np.linspace(cfg.theta_min, cfg.theta_max, 40)

    print(f"\n{'='*70}")
    print(f"📊 FDA-MIMO 雷达参数估计对比实验")
    print(f"{'='*70}")
    print(f"  样本数: {num_samples}")
    print(f"  MUSIC: {len(r_grid)}×{len(theta_grid)} 粗网格 + 自动细化")
    print(f"  OMP: {len(r_grid_omp)}×{len(theta_grid_omp)} 字典原子")
    print(f"{'='*70}\n")

    for snr in snr_list:
        print(f"📡 SNR = {snr:+3d} dB", end=" ")

        errors = {m: {"r": [], "theta": [], "time": []} for m in methods}

        for sample_idx in tqdm(range(num_samples), desc=f"SNR={snr:+3d}dB", leave=False):
            r_true = np.random.uniform(0, cfg.r_max)
            theta_true = np.random.uniform(cfg.theta_min, cfg.theta_max)
            R = generate_covariance_matrix(r_true, theta_true, snr)
            R_complex = R[0] + 1j * R[1]

            # CVNN
            t0 = time.time()
            R_tensor = torch.FloatTensor(R).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = cvnn(R_tensor).cpu().numpy()[0]
            r_pred = pred[0] * cfg.r_max
            theta_pred = pred[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
            t1 = time.time()
            errors["CVNN"]["r"].append((r_pred - r_true)**2)
            errors["CVNN"]["theta"].append((theta_pred - theta_true)**2)
            errors["CVNN"]["time"].append(t1 - t0)

            # Real-CNN
            t0 = time.time()
            with torch.no_grad():
                pred = real_cnn(R_tensor).cpu().numpy()[0]
            r_pred = pred[0] * cfg.r_max
            theta_pred = pred[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
            t1 = time.time()
            errors["Real-CNN"]["r"].append((r_pred - r_true)**2)
            errors["Real-CNN"]["theta"].append((theta_pred - theta_true)**2)
            errors["Real-CNN"]["time"].append(t1 - t0)

            # MUSIC
            t0 = time.time()
            r_pred, theta_pred = music_2d_refined(R_complex, r_grid, theta_grid, refine=True)
            t1 = time.time()
            errors["MUSIC"]["r"].append((r_pred - r_true)**2)
            errors["MUSIC"]["theta"].append((theta_pred - theta_true)**2)
            errors["MUSIC"]["time"].append(t1 - t0)

            # ESPRIT
            t0 = time.time()
            r_pred, theta_pred = esprit_2d_robust(R_complex, cfg.M, cfg.N)
            t1 = time.time()
            errors["ESPRIT"]["r"].append((r_pred - r_true)**2)
            errors["ESPRIT"]["theta"].append((theta_pred - theta_true)**2)
            errors["ESPRIT"]["time"].append(t1 - t0)

            # OMP
            t0 = time.time()
            r_pred, theta_pred = omp_2d(R_complex, r_grid_omp, theta_grid_omp)
            t1 = time.time()
            errors["OMP"]["r"].append((r_pred - r_true)**2)
            errors["OMP"]["theta"].append((theta_pred - theta_true)**2)
            errors["OMP"]["time"].append(t1 - t0)

        # 计算 RMSE
        for m in methods:
            rmse_r = np.sqrt(np.mean(errors[m]["r"]))
            rmse_theta = np.sqrt(np.mean(errors[m]["theta"]))
            avg_time = np.mean(errors[m]["time"])

            results[m]["rmse_r"].append(rmse_r)
            results[m]["rmse_theta"].append(rmse_theta)
            results[m]["time"].append(avg_time)

        # 计算 CRB
        crb_r, crb_theta = compute_crb_average(snr, L=cfg.L_snapshots, num_samples=20)
        results["CRB"]["rmse_r"].append(crb_r)
        results["CRB"]["rmse_theta"].append(crb_theta)
        results["CRB"]["time"].append(0)

        # 打印结果
        print(f"\n  {'Method':<12} {'RMSE_r (m)':>14} {'RMSE_θ (°)':>14} {'Time (ms)':>14}")
        print(f"  {'-'*56}")
        for m in methods:
            rmse_r = results[m]["rmse_r"][-1]
            rmse_theta = results[m]["rmse_theta"][-1]
            avg_time = results[m]["time"][-1] * 1000

            # 高亮最佳结果
            if rmse_r == min([results[mm]["rmse_r"][-1] for mm in methods]):
                r_marker = "🥇"
            else:
                r_marker = "  "
            if rmse_theta == min([results[mm]["rmse_theta"][-1] for mm in methods]):
                theta_marker = "🥇"
            else:
                theta_marker = "  "

            print(f"  {m:<12} {rmse_r:>14.3f}{r_marker} {rmse_theta:>14.3f}{theta_marker} {avg_time:>14.2f}")

        print(f"  {'CRB':<12} {crb_r:>14.3f}   {crb_theta:>14.3f}   {'(bound)':>14}")
        print()

    return snr_list, results


# ==========================================
# 6. 绘图函数
# ==========================================
def plot_results(snr_list, results):
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass

    methods = [m for m in results.keys() if m != "CRB"]
    colors = {
        'CVNN': '#1f77b4',
        'Real-CNN': '#2ca02c',
        'MUSIC': '#d62728',
        'ESPRIT': '#ff7f0e',
        'OMP': '#9467bd'
    }
    markers = {
        'CVNN': 'o',
        'Real-CNN': '^',
        'MUSIC': 's',
        'ESPRIT': 'd',
        'OMP': 'v'
    }

    fig = plt.figure(figsize=(20, 12))

    # 图1: 距离精度
    ax1 = plt.subplot(2, 3, 1)
    for m in methods:
        # 跳过失效的 ESPRIT
        if m == "ESPRIT" and np.mean(results[m]["rmse_r"]) > 500:
            continue
        plt.plot(snr_list, results[m]["rmse_r"],
                 color=colors.get(m, 'gray'),
                 marker=markers.get(m, 'x'),
                 label=m,
                 linewidth=2.5,
                 markersize=9,
                 alpha=0.9)
    plt.plot(snr_list, results["CRB"]["rmse_r"],
             'k--', label='CRB', linewidth=3, alpha=0.6)
    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE Range (m)', fontsize=14, fontweight='bold')
    plt.title('Range Estimation Accuracy', fontsize=16, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.yscale('log')
    ax1.tick_params(labelsize=11)

    # 图2: 角度精度
    ax2 = plt.subplot(2, 3, 2)
    for m in methods:
        plt.plot(snr_list, results[m]["rmse_theta"],
                 color=colors.get(m, 'gray'),
                 marker=markers.get(m, 'x'),
                 label=m,
                 linewidth=2.5,
                 markersize=9,
                 alpha=0.9)
    plt.plot(snr_list, results["CRB"]["rmse_theta"],
             'k--', label='CRB', linewidth=3, alpha=0.6)
    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE Angle (°)', fontsize=14, fontweight='bold')
    plt.title('Angle Estimation Accuracy', fontsize=16, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.yscale('log')
    ax2.tick_params(labelsize=11)

    # 图3: 耗时对比
    ax3 = plt.subplot(2, 3, 3)
    for m in methods:
       t_ms = [t * 1000 for t in results[m]["time"]]
       plt.plot(snr_list, t_ms,
                color=colors.get(m, 'gray'),
                marker=markers.get(m, 'x'),
                label=m,
                linewidth=2.5,
                markersize=9,
                alpha=0.9)
    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Inference Time (ms)', fontsize=14, fontweight='bold')
    plt.title('Computational Efficiency', fontsize=16, fontweight='bold', pad=15)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--', which="both")
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    ax3.tick_params(labelsize=11)

    # 图4: 综合性能雷达图
    ax4 = plt.subplot(2, 3, 4, projection='polar')

    # 计算归一化指标 (越小越好，归一化到 [0,1])
    metrics = {}
    for m in methods:
       avg_rmse_r = np.mean(results[m]["rmse_r"])
       avg_rmse_theta = np.mean(results[m]["rmse_theta"])
       avg_time = np.mean(results[m]["time"]) * 1000  # ms

       # 归一化 (反转，使得越小的值得分越高)
       max_r = max([np.mean(results[mm]["rmse_r"]) for mm in methods])
       max_theta = max([np.mean(results[mm]["rmse_theta"]) for mm in methods])
       max_time = max([np.mean(results[mm]["time"]) for mm in methods]) * 1000

       metrics[m] = [
           1 - avg_rmse_r / max_r,      # Range 准确度
           1 - avg_rmse_theta / max_theta,  # Angle 准确度
           1 - avg_time / max_time      # 速度
       ]

    categories = ['Range\nAccuracy', 'Angle\nAccuracy', 'Speed']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for m in methods:
       values = metrics[m]
       values += values[:1]
       ax4.plot(angles, values, 'o-', linewidth=2.5,
                label=m, color=colors.get(m, 'gray'),
                markersize=8, alpha=0.8)
       ax4.fill(angles, values, alpha=0.15, color=colors.get(m, 'gray'))

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=11)
    ax4.set_ylim(0, 1)
    ax4.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax4.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax4.set_title('Comprehensive Performance\n(Higher is Better)',
                 fontsize=14, fontweight='bold', pad=20)
    ax4.grid(True, linestyle='--', alpha=0.4)

    # 图5: 与 CRB 的相对性能
    ax5 = plt.subplot(2, 3, 5)

    # 计算距离估计相对于 CRB 的比值
    for m in methods:
       if m == "ESPRIT" and np.mean(results[m]["rmse_r"]) > 500:
           continue
       ratio_r = np.array(results[m]["rmse_r"]) / np.array(results["CRB"]["rmse_r"])
       plt.plot(snr_list, ratio_r,
                color=colors.get(m, 'gray'),
                marker=markers.get(m, 'x'),
                label=m,
                linewidth=2.5,
                markersize=9,
                alpha=0.9)

    plt.axhline(y=1, color='k', linestyle='--', linewidth=2.5, alpha=0.6, label='CRB')
    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized RMSE (Range / CRB)', fontsize=14, fontweight='bold')
    plt.title('Range: Distance to Optimality', fontsize=16, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.yscale('log')
    ax5.tick_params(labelsize=11)

    # 图6: 性能排名表
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # 构造表格数据
    all_methods = methods + ["CRB"]
    table_data = [['Method', 'Avg RMSE_r', 'Avg RMSE_θ', 'Avg Time', 'Rank']]

    # 计算排名 (综合距离和角度精度)
    rankings = {}
    for m in methods:
       avg_r = np.mean(results[m]["rmse_r"])
       avg_theta = np.mean(results[m]["rmse_theta"])
       # 综合得分 (归一化后平均)
       score = (avg_r / np.mean(results["CRB"]["rmse_r"]) +
                avg_theta / np.mean(results["CRB"]["rmse_theta"])) / 2
       rankings[m] = score

    sorted_methods = sorted(methods, key=lambda x: rankings[x])

    for rank, m in enumerate(sorted_methods, 1):
       avg_r = np.mean(results[m]["rmse_r"])
       avg_theta = np.mean(results[m]["rmse_theta"])
       avg_t = np.mean(results[m]["time"]) * 1000

       # 添加勋章
       if rank == 1:
           rank_str = '🥇 1st'
       elif rank == 2:
           rank_str = '🥈 2nd'
       elif rank == 3:
           rank_str = '🥉 3rd'
       else:
           rank_str = f'{rank}th'

       table_data.append([
           m,
           f'{avg_r:.2f}m',
           f'{avg_theta:.2f}°',
           f'{avg_t:.2f}ms',
           rank_str
       ])

    # 添加 CRB
    crb_r = np.mean(results["CRB"]["rmse_r"])
    crb_theta = np.mean(results["CRB"]["rmse_theta"])
    table_data.append(['CRB', f'{crb_r:.4f}m', f'{crb_theta:.4f}°', '(bound)', 'Ideal'])

    table = ax6.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.18, 0.2, 0.2, 0.2, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # 表头样式
    for i in range(5):
       table[(0, i)].set_facecolor('#2C3E50')
       table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=11)

    # 第一名高亮金色
    table[(1, 0)].set_facecolor('#FFD700')
    table[(1, 1)].set_facecolor('#FFD700')
    table[(1, 2)].set_facecolor('#FFD700')
    table[(1, 3)].set_facecolor('#FFD700')
    table[(1, 4)].set_facecolor('#FFD700')

    # 第二名银色
    table[(2, 0)].set_facecolor('#C0C0C0')
    table[(2, 1)].set_facecolor('#C0C0C0')
    table[(2, 2)].set_facecolor('#C0C0C0')
    table[(2, 3)].set_facecolor('#C0C0C0')
    table[(2, 4)].set_facecolor('#C0C0C0')

    # 第三名铜色
    table[(3, 0)].set_facecolor('#CD7F32')
    table[(3, 1)].set_facecolor('#CD7F32')
    table[(3, 2)].set_facecolor('#CD7F32')
    table[(3, 3)].set_facecolor('#CD7F32')
    table[(3, 4)].set_facecolor('#CD7F32')

    # CRB 行用灰色
    crb_row = len(all_methods)
    for i in range(5):
       table[(crb_row, i)].set_facecolor('#BDC3C7')
       table[(crb_row, i)].set_text_props(fontweight='bold')

    ax6.set_title('Performance Ranking\n(Based on Accuracy)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('benchmark_final_ultimate.png', dpi=300, bbox_inches='tight')
    print("\n✅ 图表已保存: benchmark_final_ultimate.png")

    # 额外保存高分辨率 PDF
    plt.savefig('benchmark_final_ultimate.pdf', dpi=300, bbox_inches='tight')
    print("✅ PDF 版本已保存: benchmark_final_ultimate.pdf")


# ==========================================
# 主函数
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 FDA-MIMO 雷达参数估计对比实验")
    print("="*70)
    print("算法清单:")
    print("  1. CVNN (复数神经网络)")
    print("  2. Real-CNN (实数神经网络基线)")
    print("  3. MUSIC (子空间方法)")
    print("  4. ESPRIT (旋转不变性方法)")
    print("  5. OMP (稀疏重构方法)")
    print("  6. CRB (理论下界)")
    print("="*70 + "\n")

    # 运行实验
    snr_list, results = run_benchmark()

    # 绘图
    plot_results(snr_list, results)

    print("\n" + "="*70)
    print("🎉 实验完成！")
    print("="*70)

    # 输出最终结论
    print("\n📊 关键发现:")
    methods = [m for m in results.keys() if m != "CRB"]

    # 找出最佳算法
    avg_scores = {}
    for m in methods:
       avg_r = np.mean(results[m]["rmse_r"])
       avg_theta = np.mean(results[m]["rmse_theta"])
       crb_r = np.mean(results["CRB"]["rmse_r"])
       crb_theta = np.mean(results["CRB"]["rmse_theta"])
       # 综合得分 (相对于 CRB 的倍数)
       score = (avg_r / crb_r + avg_theta / crb_theta) / 2
       avg_scores[m] = score

    best_method = min(avg_scores, key=avg_scores.get)
    print(f"  🥇 最佳精度: {best_method} (相对 CRB: {avg_scores[best_method]:.2f}x)")

    # 最快算法
    fastest = min(methods, key=lambda m: np.mean(results[m]["time"]))
    print(f"  ⚡ 最快速度: {fastest} ({np.mean(results[fastest]['time'])*1000:.2f} ms)")

    print("\n💾 结果文件:")
    print("  - benchmark_final_ultimate.png (综合对比图)")
    print("  - benchmark_final_ultimate.pdf (高清 PDF 版本)")
    print()

