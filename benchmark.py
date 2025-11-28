"""
FDA-MIMO 雷达参数估计对比实验脚本
对比算法：
1. Proposed CVNN (复数神经网络)
2. 2D-MUSIC (传统超分辨算法)
3. Real-CNN (实数神经网络基线)
4. ESPRIT (旋转不变子空间法)
5. OMP (正交匹配追踪)
6. RAM (降维交替最小化，FDA专用)
+ CRB (克拉美-罗界，理论下界)
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
from tqdm import tqdm

import config as cfg
from model import FDA_CVNN
from models_baseline import RealCNN
from utils_physics import generate_covariance_matrix, get_steering_vector


# ==========================================
# 0. 克拉美-罗界 (CRB) 计算 [已修正]
# ==========================================
def compute_crb(snr_db, L=None, M=None, N=None):
    """
    计算 FDA-MIMO 系统的克拉美-罗界 (Cramér-Rao Bound)
    
    标准 ULA CRB 公式: 
    RMSE >= sqrt( 6 / (L * SNR * (2πd/λ)^2 * N(N^2-1)) )
    
    参数:
        snr_db: 信噪比 (dB)
        L: 快拍数
        M: 发射阵元数
        N: 接收阵元数
    
    返回:
        crb_r: 距离估计的 CRB (标准差, 米)
        crb_theta: 角度估计的 CRB (标准差, 度)
    """
    L = L or cfg.L_snapshots
    M = M or cfg.M
    N = N or cfg.N
    
    # 转换 SNR 到线性
    snr_linear = 10 ** (snr_db / 10.0)
    
    # 物理参数
    c = cfg.c
    delta_f = cfg.delta_f
    d = cfg.d
    wavelength = cfg.wavelength
    
    # --- 1. 角度估计 CRB ---
    # 因子: (2π * d / λ)^2
    factor_theta = (2 * np.pi * d / wavelength) ** 2
    # 标准 ULA 公式: var = 6 / (L * SNR * factor * N(N^2-1))
    var_sin_theta = 6.0 / (L * snr_linear * factor_theta * N * (N**2 - 1))
    # 转换为角度 (度)
    crb_theta_deg = np.sqrt(var_sin_theta) * 180 / np.pi
    
    # --- 2. 距离估计 CRB ---
    # 因子: (4π * Δf / c)^2
    factor_r = (4 * np.pi * delta_f / c) ** 2
    # 公式: var = 6 / (L * SNR * factor * M(M^2-1))
    var_r = 6.0 / (L * snr_linear * factor_r * M * (M**2 - 1))
    crb_r = np.sqrt(var_r)
    
    return crb_r, crb_theta_deg


def compute_crb_full(snr_db, r_true, theta_true, L=None):
    """
    计算完整的 Fisher 信息矩阵 (FIM) 并求逆得到 CRB
    这是更精确的计算方法
    
    参数:
        snr_db: 信噪比 (dB)
        r_true: 真实距离 (米)
        theta_true: 真实角度 (度)
        L: 快拍数
    
    返回:
        crb_r: 距离估计的 CRB 标准差 (米)
        crb_theta: 角度估计的 CRB 标准差 (度)
    """
    L = L or cfg.L_snapshots
    M = cfg.M
    N = cfg.N
    MN = M * N
    
    snr_linear = 10 ** (snr_db / 10.0)
    sigma2 = 1.0 / snr_linear  # 噪声方差 (信号功率归一化为1)
    
    c = cfg.c
    delta_f = cfg.delta_f
    d = cfg.d
    wavelength = cfg.wavelength
    
    theta_rad = np.deg2rad(theta_true)
    
    # 构造导向矢量 a(r, theta) 及其导数
    m = np.arange(M)
    n = np.arange(N)
    
    # 发射导向矢量相位
    phi_tx = -4 * np.pi * delta_f * m * r_true / c + 2 * np.pi * d * m * np.sin(theta_rad) / wavelength
    a_tx = np.exp(1j * phi_tx)
    
    # 接收导向矢量相位  
    phi_rx = 2 * np.pi * d * n * np.sin(theta_rad) / wavelength
    a_rx = np.exp(1j * phi_rx)
    
    # 联合导向矢量 (Kronecker 积)
    a = np.kron(a_tx, a_rx)  # [MN,]
    
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
    
    # 链式法则
    da_dtheta = np.kron(da_tx_dtheta, a_rx) + np.kron(a_tx, da_rx_dtheta)
    
    # Fisher 信息矩阵 (2x2)
    # FIM[i,j] = 2*L*Re{(da/dθ_i)^H * R^{-1} * (da/dθ_j)} / sigma^2
    # 对于单目标高 SNR 近似: R ≈ sigma^2 * I
    # FIM[i,j] ≈ 2*L*SNR * Re{(da/dθ_i)^H * (da/dθ_j)}
    
    D = np.column_stack([da_dr, da_dtheta * np.pi / 180])  # 转换为弧度
    
    # FIM = 2 * L * SNR * Re{D^H @ D}
    FIM = 2 * L * snr_linear * np.real(D.conj().T @ D)
    
    # CRB = FIM^{-1}
    try:
        CRB = np.linalg.inv(FIM)
        crb_r = np.sqrt(CRB[0, 0])
        crb_theta = np.sqrt(CRB[1, 1])  # 已经是度
    except:
        crb_r = np.inf
        crb_theta = np.inf
    
    return crb_r, crb_theta


# ==========================================
# 1. 2D-MUSIC 算法实现
# ==========================================
def music_2d(R, r_search, theta_search):
    """
    简化的2D-MUSIC算法
    R: 协方差矩阵 (MN, MN)
    """
    # 1. 特征分解
    w, v = np.linalg.eigh(R)
    # 排序 (小到大)，取前 MN-K 个为噪声子空间
    # 假设单目标 K=1
    Un = v[:, :-1] 
    
    # 2. 构造空间谱
    # P(r, theta) = 1 / |a(r,theta)^H * Un * Un^H * a(r,theta)|
    
    # 预计算噪声投影矩阵 Pn = Un * Un^H
    Pn = Un @ Un.conj().T
    
    max_p = -1
    best_r = 0
    best_theta = 0
    
    # 网格搜索
    for r in r_search:
        for theta in theta_search:
            # 生成导向矢量
            a = get_steering_vector(r, theta)
            
            # 计算谱值
            # denom = a.conj().T @ Pn @ a
            # 优化计算: |Un^H * a|^2
            proj = Un.conj().T @ a
            denom = np.sum(np.abs(proj)**2)
            
            if denom < 1e-10:
                spectrum = 1e10
            else:
                spectrum = 1.0 / denom
            
            if spectrum > max_p:
                max_p = spectrum
                best_r = r
                best_theta = theta
                
    return best_r, best_theta


# ==========================================
# 2. ESPRIT 算法实现
# ==========================================
def esprit_2d(R, M, N):
    """
    2D-ESPRIT 算法用于 FDA-MIMO
    利用阵列的旋转不变性，无需谱搜索
    
    参数:
        R: 协方差矩阵 (MN, MN)
        M: 发射阵元数
        N: 接收阵元数
    
    返回:
        r_est, theta_est: 估计的距离和角度
    """
    MN = M * N
    K = 1  # 单目标
    
    # 1. 特征分解，获取信号子空间
    w, v = np.linalg.eigh(R)
    # 取最大的 K 个特征值对应的特征向量作为信号子空间
    Us = v[:, -K:]  # [MN, K]
    
    # 2. 构造选择矩阵 (用于提取子阵列)
    # 对于 FDA-MIMO，我们需要分别处理发射和接收维度
    
    # 发射维度选择矩阵 (用于距离估计)
    J1_tx = np.zeros((N*(M-1), MN))
    J2_tx = np.zeros((N*(M-1), MN))
    for i in range(M-1):
        for j in range(N):
            J1_tx[i*N + j, i*N + j] = 1
            J2_tx[i*N + j, (i+1)*N + j] = 1
    
    # 接收维度选择矩阵 (用于角度估计)
    J1_rx = np.zeros((M*(N-1), MN))
    J2_rx = np.zeros((M*(N-1), MN))
    for i in range(M):
        for j in range(N-1):
            J1_rx[i*(N-1) + j, i*N + j] = 1
            J2_rx[i*(N-1) + j, i*N + j + 1] = 1
    
    # 3. 提取子阵列信号子空间
    Us1_tx = J1_tx @ Us
    Us2_tx = J2_tx @ Us
    Us1_rx = J1_rx @ Us
    Us2_rx = J2_rx @ Us
    
    # 4. 最小二乘估计旋转算子
    # Φ = (Us1^H * Us1)^(-1) * Us1^H * Us2
    try:
        # 发射维度 (距离相关)
        Phi_tx = np.linalg.lstsq(Us1_tx, Us2_tx, rcond=None)[0]
        eigenvalues_tx = np.linalg.eigvals(Phi_tx)
        
        # 接收维度 (角度相关)
        Phi_rx = np.linalg.lstsq(Us1_rx, Us2_rx, rcond=None)[0]
        eigenvalues_rx = np.linalg.eigvals(Phi_rx)
        
        # 5. 从特征值中提取参数
        # 发射相位: exp(j * (-4π*Δf*r/c + 2π*d*sin(θ)/λ))
        # 接收相位: exp(j * 2π*d*sin(θ)/λ)
        
        phase_tx = np.angle(eigenvalues_tx[0])
        phase_rx = np.angle(eigenvalues_rx[0])
        
        # 从接收相位估计角度
        # phase_rx = 2π * d * sin(θ) / λ
        sin_theta = phase_rx * cfg.wavelength / (2 * np.pi * cfg.d)
        sin_theta = np.clip(sin_theta, -1, 1)
        theta_est = np.rad2deg(np.arcsin(sin_theta))
        
        # 从发射相位估计距离
        # phase_tx = -4π*Δf*r/c + 2π*d*sin(θ)/λ
        # r = (2π*d*sin(θ)/λ - phase_tx) * c / (4π*Δf)
        r_est = (2 * np.pi * cfg.d * sin_theta / cfg.wavelength - phase_tx) * cfg.c / (4 * np.pi * cfg.delta_f)
        r_est = np.clip(r_est, 0, cfg.r_max)
        
    except:
        # 如果计算失败，返回中间值
        r_est = cfg.r_max / 2
        theta_est = 0
    
    return float(np.real(r_est)), float(np.real(theta_est))


# ==========================================
# 3. OMP 算法实现
# ==========================================
def omp_2d(R, r_grid, theta_grid, K=1):
    """
    2D-OMP (正交匹配追踪) 算法
    基于稀疏重构的参数估计
    
    参数:
        R: 协方差矩阵 (MN, MN)
        r_grid: 距离搜索网格
        theta_grid: 角度搜索网格
        K: 目标数量
    
    返回:
        r_est, theta_est: 估计的距离和角度
    """
    MN = cfg.M * cfg.N
    
    # 从协方差矩阵提取主特征向量作为观测向量
    w, v = np.linalg.eigh(R)
    y = v[:, -1]  # 最大特征值对应的特征向量 [MN,]
    
    # 构造字典矩阵 A: 每列是一个导向矢量
    num_r = len(r_grid)
    num_theta = len(theta_grid)
    A = np.zeros((MN, num_r * num_theta), dtype=complex)
    
    for i, r in enumerate(r_grid):
        for j, theta in enumerate(theta_grid):
            A[:, i * num_theta + j] = get_steering_vector(r, theta)
    
    # OMP 迭代
    residual = y.copy()
    support = []
    
    for _ in range(K):
        # 计算相关性
        correlations = np.abs(A.conj().T @ residual)
        
        # 找到最大相关性的原子
        best_idx = np.argmax(correlations)
        support.append(best_idx)
        
        # 更新残差 (正交投影)
        A_s = A[:, support]
        x_s = np.linalg.lstsq(A_s, y, rcond=None)[0]
        residual = y - A_s @ x_s
    
    # 从支撑集中提取参数
    best_idx = support[0]
    r_idx = best_idx // num_theta
    theta_idx = best_idx % num_theta
    
    r_est = r_grid[r_idx]
    theta_est = theta_grid[theta_idx]
    
    return r_est, theta_est


# ==========================================
# 4. RAM 算法实现 (FDA专用)
# ==========================================
def ram_fda(R, r_grid, theta_grid, max_iter=10):
    """
    RAM (Reduced-dimension Alternating Minimization) 算法
    专门针对 FDA-MIMO 的降维交替最小化算法
    
    核心思想：交替固定距离/角度，降维搜索另一个参数
    
    参数:
        R: 协方差矩阵 (MN, MN)
        r_grid: 距离搜索网格
        theta_grid: 角度搜索网格
        max_iter: 最大迭代次数
    
    返回:
        r_est, theta_est: 估计的距离和角度
    """
    M, N = cfg.M, cfg.N
    
    # 特征分解
    w, v = np.linalg.eigh(R)
    Un = v[:, :-1]  # 噪声子空间
    
    # 初始化：用 MUSIC 粗搜索的结果
    # 使用稀疏网格快速初始化
    r_coarse = np.linspace(0, cfg.r_max, 20)
    theta_coarse = np.linspace(cfg.theta_min, cfg.theta_max, 20)
    
    best_r = cfg.r_max / 2
    best_theta = 0
    max_spectrum = -1
    
    for r in r_coarse:
        for theta in theta_coarse:
            a = get_steering_vector(r, theta)
            proj = Un.conj().T @ a
            spectrum = 1.0 / (np.sum(np.abs(proj)**2) + 1e-10)
            if spectrum > max_spectrum:
                max_spectrum = spectrum
                best_r = r
                best_theta = theta
    
    r_est = best_r
    theta_est = best_theta
    
    # 交替迭代优化
    for _ in range(max_iter):
        # Step 1: 固定 theta，优化 r
        max_spectrum = -1
        for r in r_grid:
            a = get_steering_vector(r, theta_est)
            proj = Un.conj().T @ a
            spectrum = 1.0 / (np.sum(np.abs(proj)**2) + 1e-10)
            if spectrum > max_spectrum:
                max_spectrum = spectrum
                r_est = r
        
        # Step 2: 固定 r，优化 theta
        max_spectrum = -1
        for theta in theta_grid:
            a = get_steering_vector(r_est, theta)
            proj = Un.conj().T @ a
            spectrum = 1.0 / (np.sum(np.abs(proj)**2) + 1e-10)
            if spectrum > max_spectrum:
                max_spectrum = spectrum
                theta_est = theta
    
    return r_est, theta_est


# ==========================================
# 5. 运行对比实验
# ==========================================
def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载 CVNN 模型
    cvnn = FDA_CVNN().to(device)
    cvnn_path = "checkpoints/fda_cvnn_best.pth"
    if os.path.exists(cvnn_path):
        try:
            checkpoint = torch.load(cvnn_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                cvnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                cvnn.load_state_dict(checkpoint)
            print(f"成功加载 CVNN 权重: {cvnn_path}")
        except Exception as e:
            print(f"加载 CVNN 权重失败: {e}，使用随机权重")
    else:
        print("未找到 CVNN 权重文件，使用随机权重演示...")
    cvnn.eval()
    
    # 2. 加载 Real-CNN 模型 (如果有)
    real_cnn = RealCNN().to(device)
    real_cnn_path = "checkpoints/real_cnn_best.pth"
    has_real_cnn = False
    if os.path.exists(real_cnn_path):
        try:
            checkpoint = torch.load(real_cnn_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                real_cnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                real_cnn.load_state_dict(checkpoint)
            print(f"成功加载 Real-CNN 权重: {real_cnn_path}")
            has_real_cnn = True
        except:
            pass
    if not has_real_cnn:
        print("未找到 Real-CNN 权重，将使用随机权重参与对比 (仅供参考速度)")
    real_cnn.eval()
    
    # 参数设置
    snr_list = [-5, 0, 5, 10, 15, 20]
    num_samples = 50  # 每个SNR测试样本数 (传统算法较慢)
    
    # 结果存储 (同时记录距离和角度)
    methods = ["CVNN", "Real-CNN", "MUSIC", "ESPRIT", "OMP", "RAM"]
    results = {m: {"rmse_r": [], "rmse_theta": [], "time": []} for m in methods}
    # 添加 CRB 作为理论下界
    results["CRB"] = {"rmse_r": [], "rmse_theta": [], "time": []}
    
    # 搜索网格
    r_grid = np.linspace(0, 2000, 200)    # 10m 步长
    theta_grid = np.linspace(-60, 60, 60) # 2度 步长
    
    # OMP 用稀疏网格 (否则字典太大)
    r_grid_omp = np.linspace(0, 2000, 100)    # 20m 步长
    theta_grid_omp = np.linspace(-60, 60, 40) # 3度 步长
    
    print(f"\n开始对比实验 (样本数={num_samples})...")
    print(f"对比算法: {methods}")
    print(f"MUSIC/RAM 网格: {len(r_grid)}x{len(theta_grid)} = {len(r_grid)*len(theta_grid)} 点")
    print(f"OMP 字典大小: {len(r_grid_omp)}x{len(theta_grid_omp)} = {len(r_grid_omp)*len(theta_grid_omp)} 原子")
    
    for snr in snr_list:
        print(f"\n正在测试 SNR = {snr} dB ...")
        
        # 各算法误差存储
        errors = {m: {"r": [], "theta": [], "time": []} for m in methods}
        
        for _ in tqdm(range(num_samples)):
            # 生成数据
            r_true = np.random.uniform(0, 2000)
            theta_true = np.random.uniform(-60, 60)
            R = generate_covariance_matrix(r_true, theta_true, snr)
            R_complex = R[0] + 1j * R[1]  # 复数形式
            
            # --- 1. 测试 CVNN ---
            t0 = time.time()
            R_tensor = torch.FloatTensor(R).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = cvnn(R_tensor).cpu().numpy()[0]
            r_pred = pred[0] * 2000
            theta_pred = pred[1] * 120 - 60
            t1 = time.time()
            errors["CVNN"]["r"].append((r_pred - r_true)**2)
            errors["CVNN"]["theta"].append((theta_pred - theta_true)**2)
            errors["CVNN"]["time"].append(t1 - t0)
            
            # --- 2. 测试 Real-CNN ---
            t0 = time.time()
            with torch.no_grad():
                pred = real_cnn(R_tensor).cpu().numpy()[0]
            r_pred = pred[0] * 2000
            theta_pred = pred[1] * 120 - 60
            t1 = time.time()
            errors["Real-CNN"]["r"].append((r_pred - r_true)**2)
            errors["Real-CNN"]["theta"].append((theta_pred - theta_true)**2)
            errors["Real-CNN"]["time"].append(t1 - t0)
            
            # --- 3. 测试 MUSIC ---
            t0 = time.time()
            r_pred, theta_pred = music_2d(R_complex, r_grid, theta_grid)
            t1 = time.time()
            errors["MUSIC"]["r"].append((r_pred - r_true)**2)
            errors["MUSIC"]["theta"].append((theta_pred - theta_true)**2)
            errors["MUSIC"]["time"].append(t1 - t0)
            
            # --- 4. 测试 ESPRIT ---
            t0 = time.time()
            r_pred, theta_pred = esprit_2d(R_complex, cfg.M, cfg.N)
            t1 = time.time()
            errors["ESPRIT"]["r"].append((r_pred - r_true)**2)
            errors["ESPRIT"]["theta"].append((theta_pred - theta_true)**2)
            errors["ESPRIT"]["time"].append(t1 - t0)
            
            # --- 5. 测试 OMP ---
            t0 = time.time()
            r_pred, theta_pred = omp_2d(R_complex, r_grid_omp, theta_grid_omp)
            t1 = time.time()
            errors["OMP"]["r"].append((r_pred - r_true)**2)
            errors["OMP"]["theta"].append((theta_pred - theta_true)**2)
            errors["OMP"]["time"].append(t1 - t0)
            
            # --- 6. 测试 RAM ---
            t0 = time.time()
            r_pred, theta_pred = ram_fda(R_complex, r_grid, theta_grid, max_iter=5)
            t1 = time.time()
            errors["RAM"]["r"].append((r_pred - r_true)**2)
            errors["RAM"]["theta"].append((theta_pred - theta_true)**2)
            errors["RAM"]["time"].append(t1 - t0)
        
        # 计算并存储 RMSE
        for m in methods:
            rmse_r = np.sqrt(np.mean(errors[m]["r"]))
            rmse_theta = np.sqrt(np.mean(errors[m]["theta"]))
            avg_time = np.mean(errors[m]["time"])
            
            results[m]["rmse_r"].append(rmse_r)
            results[m]["rmse_theta"].append(rmse_theta)
            results[m]["time"].append(avg_time)
        
        # 计算 CRB (使用中间值作为参考点)
        crb_r, crb_theta = compute_crb(snr, L=cfg.L_snapshots)
        results["CRB"]["rmse_r"].append(crb_r)
        results["CRB"]["rmse_theta"].append(crb_theta)
        results["CRB"]["time"].append(0)  # CRB 不是算法，无计算时间
        
        # 打印结果
        print(f"  {'Method':<10} {'RMSE_r (m)':>12} {'RMSE_θ (°)':>12} {'Time (ms)':>12}")
        print(f"  {'-'*48}")
        for m in methods:
            rmse_r = results[m]["rmse_r"][-1]
            rmse_theta = results[m]["rmse_theta"][-1]
            avg_time = results[m]["time"][-1] * 1000
            print(f"  {m:<10} {rmse_r:>12.2f} {rmse_theta:>12.2f} {avg_time:>12.2f}")
        # 打印 CRB
        print(f"  {'CRB':<10} {crb_r:>12.2f} {crb_theta:>12.2f} {'(bound)':>12}")

    return snr_list, results

# ==========================================
# 6. 绘图
# ==========================================
def plot_results(snr_list, results):
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass
    
    # 分离算法和 CRB
    methods = [m for m in results.keys() if m != "CRB"]
    colors = {'CVNN': 'b', 'Real-CNN': 'g', 'MUSIC': 'r', 
              'ESPRIT': 'c', 'OMP': 'm', 'RAM': 'orange'}
    markers = {'CVNN': 'o', 'Real-CNN': '^', 'MUSIC': 's', 
               'ESPRIT': 'd', 'OMP': 'v', 'RAM': 'p'}
    
    plt.figure(figsize=(18, 12))
    
    # 图1: 距离精度对比 (含 CRB)
    # 注意: ESPRIT 在 FDA 距离估计上可能失效，如果误差过大则跳过
    plt.subplot(2, 2, 1)
    for m in methods:
        # 如果 ESPRIT 距离误差 > 500m，说明失效，跳过以免破坏坐标轴
        if m == "ESPRIT" and np.mean(results[m]["rmse_r"]) > 500:
            continue
        plt.plot(snr_list, results[m]["rmse_r"], 
                 color=colors.get(m, 'gray'), marker=markers.get(m, 'x'), 
                 label=m, linewidth=2, markersize=8)
    # 绘制 CRB (虚线，黑色)
    plt.plot(snr_list, results["CRB"]["rmse_r"], 
             'k--', label='CRB', linewidth=2.5)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('RMSE Range (m)', fontsize=12)
    plt.title('Range Estimation Accuracy vs. SNR', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='upper right')
    plt.yscale('log')  # 用对数坐标更好展示 CRB
    
    # 图2: 角度精度对比 (含 CRB)
    plt.subplot(2, 2, 2)
    for m in methods:
        plt.plot(snr_list, results[m]["rmse_theta"], 
                 color=colors.get(m, 'gray'), marker=markers.get(m, 'x'), 
                 label=m, linewidth=2, markersize=8)
    # 绘制 CRB
    plt.plot(snr_list, results["CRB"]["rmse_theta"], 
             'k--', label='CRB', linewidth=2.5)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('RMSE Angle (°)', fontsize=12)
    plt.title('Angle Estimation Accuracy vs. SNR', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='upper right')
    plt.yscale('log')
    
    # 图3: 耗时对比 (对数坐标，不含 CRB)
    plt.subplot(2, 2, 3)
    for m in methods:
        t_ms = [t * 1000 for t in results[m]["time"]]
        plt.plot(snr_list, t_ms, 
                 color=colors.get(m, 'gray'), marker=markers.get(m, 'x'), 
                 label=m, linewidth=2, markersize=8)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Inference Time (ms)', fontsize=12)
    plt.title('Computation Efficiency (Log Scale)', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which="both")
    plt.legend(fontsize=9, loc='upper right')
    
    # 图4: 综合表格 (含 CRB)
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # 构造表格数据
    all_methods = methods + ["CRB"]
    table_data = [['Method', 'Avg RMSE_r (m)', 'Avg RMSE_θ (°)', 'Avg Time (ms)']]
    for m in all_methods:
        avg_r = np.mean(results[m]["rmse_r"])
        avg_theta = np.mean(results[m]["rmse_theta"])
        if m == "CRB":
            table_data.append([m, f'{avg_r:.4f}', f'{avg_theta:.4f}', '(bound)'])
        else:
            avg_t = np.mean(results[m]["time"]) * 1000
            table_data.append([m, f'{avg_r:.2f}', f'{avg_theta:.2f}', f'{avg_t:.2f}'])
    
    table = plt.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.22, 0.22, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # CRB 行用灰色背景
    crb_row = len(all_methods)
    for i in range(4):
        table[(crb_row, i)].set_facecolor('#E0E0E0')
    
    # 高亮最佳算法结果 (不含 CRB)
    best_r_idx = np.argmin([np.mean(results[m]["rmse_r"]) for m in methods]) + 1
    best_theta_idx = np.argmin([np.mean(results[m]["rmse_theta"]) for m in methods]) + 1
    best_time_idx = np.argmin([np.mean(results[m]["time"]) for m in methods]) + 1
    
    table[(best_r_idx, 1)].set_facecolor('#90EE90')
    table[(best_theta_idx, 2)].set_facecolor('#90EE90')
    table[(best_time_idx, 3)].set_facecolor('#90EE90')
    
    plt.title('Average Performance Summary (Green = Best, Gray = CRB)', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存至 benchmark_comparison.png")


if __name__ == "__main__":
    snr_list, results = run_benchmark()
    plot_results(snr_list, results)
