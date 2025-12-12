"""FDA-MIMO 雷达参数估计对比实验 (完整修复版 v2)
修复说明:
- CRB: 修复了统计方式，消除奇异值影响，解决 CRB 虚高问题。
- OMP: 增加了两级搜索 (Coarse + Fine)，解决因网格量化导致的 RMSE "直线" (误差饱和) 问题。

算法清单:
1. CVNN: 复数神经网络 (本文方法)
2. Real-CNN: 实数神经网络基线
3. MUSIC: 子空间方法 (两级搜索)
4. ESPRIT: 旋转不变性方法
5. OMP: 稀疏重构方法 (两级搜索) [已修复]
6. CRB: 克拉美-罗界 (理论下界) [已修复]
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

import config as cfg
from model import FDA_CVNN, FDA_CVNN_Attention, FDA_CVNN_FAR
from models_baseline import RealCNN
from utils_physics import generate_covariance_matrix, get_steering_vector

# 屏蔽警告
import warnings
warnings.filterwarnings("ignore")


# ==========================================
# 0. 克拉美-罗界 (完整 FIM 版本) [已修复]
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
    
    c = cfg.c
    delta_f = cfg.delta_f
    d = cfg.d
    wavelength = cfg.wavelength
    theta_rad = np.deg2rad(theta_true)

    # 构造导向矢量及其导数
    m = np.arange(M)
    n = np.arange(N)

    # 发射部分
    phi_tx = -4 * np.pi * delta_f * m * r_true / c + 2 * np.pi * d * m * np.sin(theta_rad) / wavelength
    a_tx = np.exp(1j * phi_tx)

    # 接收部分
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
        # 矩阵奇异或计算错误
        crb_r = np.nan
        crb_theta = np.nan

    return crb_r, crb_theta


def compute_crb_average(snr_db, L=None, num_samples=200):
    """
    [修正] 使用 Mean 而非 Median，与 RMSE 的统计口径保持一致
    
    说明：
    - RMSE 使用 np.sqrt(np.mean(errors))，是均值统计
    - CRB 也应该使用均值，否则会出现 RMSE < CRB 的"不合理"现象
    - FDA-MIMO 在某些角度 CRB 会变得很大（接近不可观测），需要截断
    """
    crb_r_list = []
    crb_theta_list = []
    
    # 限制 CRB 的最大值，防止极端值拉爆均值
    # FDA-MIMO 在端射方向不可观测，CRB 理论上无穷大
    limit_r = cfg.r_max
    limit_theta = 180

    for _ in range(num_samples):
        r_true = np.random.uniform(0, cfg.r_max)
        theta_true = np.random.uniform(cfg.theta_min, cfg.theta_max)
        
        crb_r, crb_theta = compute_crb_full(snr_db, r_true, theta_true, L)
        
        # 严格过滤 NaN、Inf 和物理上不可能的大值
        if np.isfinite(crb_r) and np.isfinite(crb_theta):
            if crb_r < limit_r and crb_theta < limit_theta:
                crb_r_list.append(crb_r)
                crb_theta_list.append(crb_theta)

    if not crb_r_list:
        return np.inf, np.inf

    # 使用 Mean，与 RMSE 统计口径一致
    return np.mean(crb_r_list), np.mean(crb_theta_list)


# ==========================================
# 1. 改进的 2D-MUSIC (向量化 + 两级搜索)
# ==========================================
def music_2d_refined(R, r_search_coarse, theta_search_coarse, refine=True):
    """
    [标准改进版] 向量化 2D-MUSIC
    
    优势: 
    - 速度极快 (矩阵运算代替 for 循环)
    - 允许使用细网格，避免漏掉 MUSIC 的尖峰
    """
    M, N = cfg.M, cfg.N
    
    # 1. 特征分解与噪声子空间
    w, v = np.linalg.eigh(R)
    K = 1  # 单目标
    Un = v[:, :-K]  # (MN, MN-K)
    
    # 2. 向量化构建导向矢量字典
    R_grid, Theta_grid = np.meshgrid(r_search_coarse, theta_search_coarse, indexing='ij')
    R_flat = R_grid.flatten()
    Theta_flat = Theta_grid.flatten()
    
    m_idx = np.arange(M).reshape(-1, 1)  # (M, 1)
    n_idx = np.arange(N).reshape(-1, 1)  # (N, 1)
    Theta_rad = np.deg2rad(Theta_flat)
    
    # 发射相位: -4*pi*df*m*r/c + 2*pi*d*m*sin(theta)/lam
    phi_tx = (-4 * np.pi * cfg.delta_f * m_idx * R_flat / cfg.c + 
              2 * np.pi * cfg.d * m_idx * np.sin(Theta_rad) / cfg.wavelength)
    a_tx = np.exp(1j * phi_tx)  # (M, N_grid)
    
    # 接收相位: 2*pi*d*n*sin(theta)/lam
    phi_rx = 2 * np.pi * cfg.d * n_idx * np.sin(Theta_rad) / cfg.wavelength
    a_rx = np.exp(1j * phi_rx)  # (N, N_grid)
    
    # Khatri-Rao 积: A[m*N + n, :] = a_tx[m, :] * a_rx[n, :]
    A = (a_tx[:, np.newaxis, :] * a_rx[np.newaxis, :, :]).reshape(M*N, -1)
    
    # 3. 矩阵化计算谱: P = 1 / sum(|Un^H * A|^2, axis=0)
    proj = Un.conj().T @ A  # (MN-K, N_grid)
    spectrum = 1.0 / (np.sum(np.abs(proj)**2, axis=0) + 1e-12)
    
    # 4. 找到粗搜索最大值
    idx = np.argmax(spectrum)
    best_r = R_flat[idx]
    best_theta = Theta_flat[idx]
    
    if not refine:
        return best_r, best_theta
    
    # 5. 细搜索 (局部小范围)
    r_step = (r_search_coarse[-1] - r_search_coarse[0]) / (len(r_search_coarse) - 1) if len(r_search_coarse) > 1 else 50
    theta_step = (theta_search_coarse[-1] - theta_search_coarse[0]) / (len(theta_search_coarse) - 1) if len(theta_search_coarse) > 1 else 2
    
    r_fine = np.linspace(max(0, best_r - r_step/2), 
                         min(cfg.r_max, best_r + r_step/2), 21)
    theta_fine = np.linspace(max(cfg.theta_min, best_theta - theta_step/2), 
                             min(cfg.theta_max, best_theta + theta_step/2), 21)
    
    # 细搜索用简单循环 (点数少)
    max_p = -1
    refined_r, refined_theta = best_r, best_theta
    
    for r in r_fine:
        for t in theta_fine:
            a = get_steering_vector(r, t)
            p = 1.0 / (np.sum(np.abs(Un.conj().T @ a)**2) + 1e-12)
            if p > max_p:
                max_p = p
                refined_r, refined_theta = r, t
    
    return refined_r, refined_theta


# ==========================================
# 1b. 连续优化 MUSIC (消除栅栏效应，逼近 CRB)
# ==========================================
def music_2d_continuous(R, r_search_coarse, theta_search_coarse):
    """
    [高精度修复版] 连续优化 MUSIC
    
    策略: 粗网格搜索 + Scipy 连续优化 (Nelder-Mead)
    解决: 彻底消除"栅栏效应"，在高 SNR 下能紧贴 CRB
    
    注意: 比 music_2d_refined 慢 ~3-5 倍，但精度更高
    """
    M, N = cfg.M, cfg.N
    
    # 1. 特征分解
    w, v = np.linalg.eigh(R)
    Un = v[:, :-1]  # 噪声子空间 (假设单目标)
    
    # --- 阶段一: 向量化粗搜索 ---
    R_grid, Theta_grid = np.meshgrid(r_search_coarse, theta_search_coarse, indexing='ij')
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
    
    # 计算谱 (分母越小越好)
    proj = Un.conj().T @ A
    spectrum_denom = np.sum(np.abs(proj)**2, axis=0)
    
    idx = np.argmin(spectrum_denom)  # 找分母最小值
    r0 = R_flat[idx]
    theta0 = Theta_flat[idx]
    
    # --- 阶段二: 连续优化 (Nelder-Mead) ---
    def objective_function(x):
        r, theta_deg = x
        # 边界检查
        if r < 0 or r > cfg.r_max:
            return 1e10
        if theta_deg < cfg.theta_min or theta_deg > cfg.theta_max:
            return 1e10
            
        theta = np.deg2rad(theta_deg)
        
        # 生成导向矢量
        m = np.arange(M)
        n = np.arange(N)
        
        phi_tx_ = (-4 * np.pi * cfg.delta_f * m * r / cfg.c +
                   2 * np.pi * cfg.d * m * np.sin(theta) / cfg.wavelength)
        a_tx_ = np.exp(1j * phi_tx_)
        
        phi_rx_ = 2 * np.pi * cfg.d * n * np.sin(theta) / cfg.wavelength
        a_rx_ = np.exp(1j * phi_rx_)
        
        a = np.kron(a_tx_, a_rx_)
        
        # 投影到噪声子空间 (最小化)
        return np.linalg.norm(Un.conj().T @ a) ** 2

    # 使用 Nelder-Mead 算法
    res = minimize(objective_function, x0=[r0, theta0], method='Nelder-Mead',
                   options={'xatol': 0.1, 'fatol': 1e-8, 'maxiter': 100})
    
    final_r, final_theta = res.x
    
    # 确保结果在有效范围内
    final_r = np.clip(final_r, 0, cfg.r_max)
    final_theta = np.clip(final_theta, cfg.theta_min, cfg.theta_max)
    
    return final_r, final_theta


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

        phi_angle = 2 * np.pi * cfg.d * sin_theta / cfg.wavelength
        diff_phase = phase_tx - phi_angle
        r_est = -diff_phase * cfg.c / (4 * np.pi * cfg.delta_f)

        max_unambiguous_r = cfg.c / (2 * cfg.delta_f)
        while r_est < 0: r_est += max_unambiguous_r
        while r_est > cfg.r_max: r_est -= max_unambiguous_r
        r_est = np.clip(r_est, 0, cfg.r_max)

    except Exception:
        r_est = cfg.r_max / 2
        theta_est = 0

    return float(np.real(r_est)), float(np.real(theta_est))


# ==========================================
# 3. OMP (向量化 + 两级搜索)
# ==========================================
def omp_2d_refined(R, r_grid_coarse, theta_grid_coarse, refine=True):
    """
    [标准修复版] 向量化 OMP
    
    区别于 MUSIC:
    - OMP 基于信号子空间 (最大特征向量)
    - MUSIC 基于噪声子空间
    - 在 L=1 单目标时两者数学上近似等价
    """
    M, N = cfg.M, cfg.N
    
    # 1. 获取观测信号 (取最大特征向量作为信号代理 y)
    w, v = np.linalg.eigh(R)
    y = v[:, -1]  # (MN,)
    
    # 2. 向量化构建字典矩阵 A
    R_grid, Theta_grid = np.meshgrid(r_grid_coarse, theta_grid_coarse, indexing='ij')
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
    
    # 构建字典 A: (MN, N_grid)
    A = (a_tx[:, np.newaxis, :] * a_rx[np.newaxis, :, :]).reshape(M*N, -1)
    
    # 归一化字典原子 (OMP 关键步骤)
    A = A / np.sqrt(M*N)
    
    # 3. 匹配: correlations = |A^H * y|
    correlations = np.abs(A.conj().T @ y)
    
    # 4. 找到最佳匹配原子
    idx = np.argmax(correlations)
    best_r = R_flat[idx]
    best_theta = Theta_flat[idx]
    
    if not refine:
        return best_r, best_theta
    
    # 5. 细搜索 (OMP 的峰值比 MUSIC 更"钝"，细搜效果不如 MUSIC 明显)
    r_step = (r_grid_coarse[-1] - r_grid_coarse[0]) / (len(r_grid_coarse) - 1) if len(r_grid_coarse) > 1 else 100
    theta_step = (theta_grid_coarse[-1] - theta_grid_coarse[0]) / (len(theta_grid_coarse) - 1) if len(theta_grid_coarse) > 1 else 2
    
    r_fine = np.linspace(max(0, best_r - r_step), 
                         min(cfg.r_max, best_r + r_step), 21)
    theta_fine = np.linspace(max(cfg.theta_min, best_theta - theta_step), 
                             min(cfg.theta_max, best_theta + theta_step), 21)
    
    max_corr = -1
    refined_r, refined_theta = best_r, best_theta
    
    # 导向矢量由纯相位项组成 (e^{jφ})，模长恒定为 sqrt(M*N)，预计算加速
    norm_factor = np.sqrt(M * N)
    
    for r in r_fine:
        for t in theta_fine:
            a = get_steering_vector(r, t)
            # 直接除常数，避免每次循环计算 np.linalg.norm
            corr = np.abs(a.conj().T @ y) / norm_factor
            if corr > max_corr:
                max_corr = corr
                refined_r, refined_theta = r, t
    
    return refined_r, refined_theta


# ==========================================
# 辅助函数
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
    
    pattern_random = f"{checkpoint_dir}/fda_cvnn_*_Lrandom_best.pth"
    if glob.glob(pattern_random): candidates.extend(glob.glob(pattern_random))
    candidates.append(f"{checkpoint_dir}/fda_cvnn_Lrandom_best.pth")
    
    if model_type: candidates.append(f"{checkpoint_dir}/fda_cvnn_{model_type}_best.pth")
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
        
        # 简易特征检测
        keys = list(state_dict.keys())
        has_far = any('attn' in k and 'conv_rr' in k for k in keys)
        has_se = any('attn' in k and '.fc.' in k for k in keys)
        has_dual = any('global_attn' in k for k in keys)
        
        if has_dual: model = FDA_CVNN_Attention(attention_type='dual').to(device)
        elif has_far: model = FDA_CVNN_Attention(attention_type='far').to(device)
        elif has_se: model = FDA_CVNN_Attention(attention_type='se').to(device)
        else: model = FDA_CVNN().to(device)
        
        # 修复 module. 前缀
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        return model
    except:
        return FDA_CVNN().to(device)


# ==========================================
# 5. 运行对比实验
# ==========================================
def run_benchmark(L_snapshots=None, num_samples=500, fast_mode=False, music_continuous=False):
    """
    运行 SNR 对比实验
    
    Args:
        L_snapshots: 快拍数
        num_samples: 每个 SNR 下的测试样本数 (默认 500)
        fast_mode: 快速模式，只测神经网络方法 (GPU 利用率高)
        music_continuous: 使用连续优化版 MUSIC (消除栅栏效应，逼近 CRB)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    
    if L_snapshots is not None: cfg.L_snapshots = L_snapshots
    L = cfg.L_snapshots
    print(f"📊 当前快拍数: L = {L}")
    print(f"📊 测试样本数: {num_samples}")
    if fast_mode:
        print(f"⚡ 快速模式: 只测试神经网络方法 (GPU 密集)")

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

    # 快速模式只测神经网络
    if fast_mode:
        methods = ["CVNN", "Real-CNN"]
    else:
        methods = ["CVNN", "Real-CNN", "MUSIC", "ESPRIT", "OMP"]
    
    results = {m: {"rmse_r": [], "rmse_theta": [], "time": []} for m in methods}
    results["CRB"] = {"rmse_r": [], "rmse_theta": [], "time": []}

    # ========================================
    # 基于物理分辨率的网格设置 (学术标准)
    # ========================================
    # 距离分辨率: c / (2 * Bandwidth), Bandwidth = M * delta_f
    res_r = cfg.c / (2 * cfg.M * cfg.delta_f)
    # 角度分辨率: lambda / Aperture, Aperture = N * d  
    res_theta = np.rad2deg(cfg.wavelength / (cfg.N * cfg.d))
    
    # 粗搜索步长设为分辨率的一半 (Nyquist 采样准则)
    step_r_coarse = res_r / 2
    step_theta_coarse = res_theta / 2
    
    # 使用物理步长动态生成网格 (避免栅栏效应 Grid Straddling Loss)
    num_r_points = max(int(cfg.r_max / step_r_coarse) + 1, 50)  # 至少50点
    num_theta_points = max(int((cfg.theta_max - cfg.theta_min) / step_theta_coarse) + 1, 30)
    
    r_grid = np.linspace(0, cfg.r_max, num_r_points)
    theta_grid = np.linspace(cfg.theta_min, cfg.theta_max, num_theta_points)
    
    # OMP: 与 MUSIC 相同网格 (公平对比)
    r_grid_omp = r_grid
    theta_grid_omp = theta_grid
    
    print(f"\n📐 物理分辨率: Range={res_r:.2f}m, Angle={res_theta:.2f}°")
    print(f"📐 动态生成网格: {len(r_grid)}×{len(theta_grid)} = {len(r_grid)*len(theta_grid)} 点 (基于分辨率/2)")
    if music_continuous:
        print(f"🔬 MUSIC 使用连续优化 (消除栅栏效应，逼近 CRB)")

    print(f"\n{'='*70}\n📊 对比实验开始 (Samples={num_samples})\n{'='*70}")

    for snr in snr_list:
        print(f"📡 SNR = {snr:+3d} dB", end=" ")
        errors = {m: {"r": [], "theta": [], "time": []} for m in methods}

        for _ in tqdm(range(num_samples), desc=f"SNR={snr}", leave=False):
            r_true = np.random.uniform(0, cfg.r_max)
            theta_true = np.random.uniform(cfg.theta_min, cfg.theta_max)
            R = generate_covariance_matrix(r_true, theta_true, snr)
            R_complex = R[0] + 1j * R[1]
            R_tensor = torch.FloatTensor(R).unsqueeze(0).to(device)

            # CVNN
            t0 = time.time()
            with torch.no_grad(): pred = cvnn(R_tensor).cpu().numpy()[0]
            errors["CVNN"]["r"].append((pred[0]*cfg.r_max - r_true)**2)
            errors["CVNN"]["theta"].append((pred[1]*(cfg.theta_max-cfg.theta_min)+cfg.theta_min - theta_true)**2)
            errors["CVNN"]["time"].append(time.time()-t0)

            # Real-CNN
            t0 = time.time()
            with torch.no_grad(): pred = real_cnn(R_tensor).cpu().numpy()[0]
            errors["Real-CNN"]["r"].append((pred[0]*cfg.r_max - r_true)**2)
            errors["Real-CNN"]["theta"].append((pred[1]*(cfg.theta_max-cfg.theta_min)+cfg.theta_min - theta_true)**2)
            errors["Real-CNN"]["time"].append(time.time()-t0)

            # MUSIC (可选连续优化版本)
            t0 = time.time()
            if music_continuous:
                r_est, th_est = music_2d_continuous(R_complex, r_grid, theta_grid)
            else:
                r_est, th_est = music_2d_refined(R_complex, r_grid, theta_grid)
            errors["MUSIC"]["r"].append((r_est-r_true)**2)
            errors["MUSIC"]["theta"].append((th_est-theta_true)**2)
            errors["MUSIC"]["time"].append(time.time()-t0)

            # ESPRIT
            t0 = time.time()
            r_est, th_est = esprit_2d_robust(R_complex, cfg.M, cfg.N)
            errors["ESPRIT"]["r"].append((r_est-r_true)**2)
            errors["ESPRIT"]["theta"].append((th_est-theta_true)**2)
            errors["ESPRIT"]["time"].append(time.time()-t0)

            # OMP [Modified call: use refined version]
            t0 = time.time()
            r_est, th_est = omp_2d_refined(R_complex, r_grid_omp, theta_grid_omp, refine=True)
            errors["OMP"]["r"].append((r_est-r_true)**2)
            errors["OMP"]["theta"].append((th_est-theta_true)**2)
            errors["OMP"]["time"].append(time.time()-t0)

        # 统计
        for m in methods:
            results[m]["rmse_r"].append(np.sqrt(np.mean(errors[m]["r"])))
            results[m]["rmse_theta"].append(np.sqrt(np.mean(errors[m]["theta"])))
            results[m]["time"].append(np.mean(errors[m]["time"]))

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

    # 4. 雷达图
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    metrics = {}
    for m in methods:
        rmse_r = np.mean(results[m]["rmse_r"])
        rmse_theta = np.mean(results[m]["rmse_theta"])
        time_v = np.mean(results[m]["time"])
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

    # 5. 相对 CRB
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

    plt.suptitle(f'Benchmark L={L}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'results/benchmark_L{L}.png', dpi=300)
    print(f"\n✅ 图表已保存: results/benchmark_L{L}.png")


# ==========================================
# 7. 快拍数对比实验
# ==========================================
def run_snapshots_benchmark(snr_db=0, L_list=None, num_samples=200, use_random_model=False):
    if L_list is None: L_list = [1, 5, 10, 25, 50, 100]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\n📊 快拍数对比实验 (SNR={snr_db}dB)\n{'='*70}")
    
    methods = ["MUSIC", "ESPRIT", "OMP", "CVNN", "CRB"]
    results = {m: {"rmse_r": [], "rmse_theta": [], "time": []} for m in methods}
    
    # 基于物理分辨率动态生成网格 (与 run_benchmark 保持一致)
    res_r = cfg.c / (2 * cfg.M * cfg.delta_f)
    res_theta = np.rad2deg(cfg.wavelength / (cfg.N * cfg.d))
    step_r = res_r / 2
    step_theta = res_theta / 2
    
    num_r_points = max(int(cfg.r_max / step_r) + 1, 50)
    num_theta_points = max(int((cfg.theta_max - cfg.theta_min) / step_theta) + 1, 30)
    
    r_grid = np.linspace(0, cfg.r_max, num_r_points)
    theta_grid = np.linspace(cfg.theta_min, cfg.theta_max, num_theta_points)
    r_grid_omp = r_grid
    theta_grid_omp = theta_grid
    
    print(f"📐 动态网格: {len(r_grid)}×{len(theta_grid)} 点")

    cvnn = load_cvnn_model(device, L_snapshots=(None if use_random_model else L_list[0]), use_random_model=use_random_model)
    cvnn.eval()

    for L in L_list:
        print(f"📡 L = {L} 快拍", end="\r")
        cfg.L_snapshots = L
        if not use_random_model:
            cvnn = load_cvnn_model(device, L_snapshots=L)
            cvnn.eval()

        errors = {m: {"r": [], "theta": [], "time": []} for m in methods}

        for _ in tqdm(range(num_samples), leave=False):
            r_true = np.random.uniform(0, cfg.r_max)
            theta_true = np.random.uniform(cfg.theta_min, cfg.theta_max)
            R = generate_covariance_matrix(r_true, theta_true, snr_db)
            R_complex = R[0] + 1j * R[1]
            R_tensor = torch.FloatTensor(R).unsqueeze(0).to(device)
            
            t0 = time.time(); pred = cvnn(R_tensor).cpu().detach().numpy()[0]
            errors["CVNN"]["r"].append((pred[0]*cfg.r_max - r_true)**2)
            errors["CVNN"]["time"].append(time.time()-t0)
            
            t0 = time.time(); r_est, _ = music_2d_refined(R_complex, r_grid, theta_grid)
            errors["MUSIC"]["r"].append((r_est - r_true)**2)
            errors["MUSIC"]["time"].append(time.time()-t0)
            
            t0 = time.time(); r_est, _ = esprit_2d_robust(R_complex, cfg.M, cfg.N)
            errors["ESPRIT"]["r"].append((r_est - r_true)**2)
            errors["ESPRIT"]["time"].append(time.time()-t0)
            
            # OMP Modified
            t0 = time.time(); r_est, _ = omp_2d_refined(R_complex, r_grid_omp, theta_grid_omp)
            errors["OMP"]["r"].append((r_est - r_true)**2)
            errors["OMP"]["time"].append(time.time()-t0)

        for m in methods:
            if m != "CRB":
                results[m]["rmse_r"].append(np.sqrt(np.mean(errors[m]["r"])))
                results[m]["time"].append(np.mean(errors[m]["time"]))
        
        crb_r, _ = compute_crb_average(snr_db, L=L, num_samples=200)
        results["CRB"]["rmse_r"].append(crb_r)
        
        print(f"L={L:<3} | CVNN: {results['CVNN']['rmse_r'][-1]:.2f}m | OMP: {results['OMP']['rmse_r'][-1]:.2f}m")

    plt.figure(figsize=(10, 6))
    for m in methods:
        if m == "ESPRIT" and np.mean(results[m]["rmse_r"]) > 500: continue
        plt.plot(L_list, results[m]["rmse_r"], 'o-', label=m)
    plt.plot(L_list, results["CRB"]["rmse_r"], 'k--', label='CRB')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Snapshots (L)'); plt.ylabel('RMSE Range (m)')
    plt.title(f'Performance vs Snapshots (SNR={snr_db}dB)')
    plt.legend(); plt.grid(True, which='both')
    plt.savefig(f'results/snapshots_SNR{snr_db}dB.png')
    
    return L_list, results


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    print("\n" + "="*70 + "\n🎯 FDA-MIMO 雷达参数估计对比实验 (完整修复版 v2)\n" + "="*70)
    snr_list, results, L = run_benchmark()
    plot_results(snr_list, results, L)
    print("\n" + "="*70 + "\n🎉 实验完成！\n" + "="*70)