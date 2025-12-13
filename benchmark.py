"""
FDA-MIMO 标准基准测试 (Standard Benchmark)
基于 config.py 的物理参数进行公平对比。

对比算法:
1. CVNN (本文方法, 连续值预测)
2. Real-CNN (基线, 实数网络)
3. MUSIC (经典子空间法, 1/10 分辨率网格搜索)
4. ESPRIT (旋转不变子空间法, 解析解)
5. OMP (压缩感知, 1/10 分辨率网格搜索)
6. CRB (理论下界)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
from tqdm import tqdm
import scipy.linalg

# 导入你的配置和模型
try:
    import config as cfg
    from model import FDA_CVNN, FDA_CVNN_Attention, FDA_CVNN_FAR
    from models_baseline import RealCNN
except ImportError:
    print("❌ 错误: 未找到项目文件 (config.py, model.py 等)")
    exit(1)

# 屏蔽警告
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# 1. 物理模型工具 (确保与 Config 一致)
# =========================================================
def get_steering_vector(r, theta):
    """生成 FDA-MIMO 导向矢量"""
    theta_rad = np.deg2rad(theta)
    m = np.arange(cfg.M).reshape(-1, 1)
    n = np.arange(cfg.N).reshape(-1, 1)

    # 发射导向矢量 (包含距离 r 和角度 theta 信息)
    # phi_tx = -4*pi*delta_f*m*r/c + 2*pi*d*m*sin(theta)/lam
    phi_tx = -4 * np.pi * cfg.delta_f * m * r / cfg.c + \
              2 * np.pi * cfg.d * m * np.sin(theta_rad) / cfg.wavelength
    a_tx = np.exp(1j * phi_tx)

    # 接收导向矢量 (仅角度 theta)
    phi_rx = 2 * np.pi * cfg.d * n * np.sin(theta_rad) / cfg.wavelength
    a_rx = np.exp(1j * phi_rx)

    # 联合导向矢量 (Khatri-Rao 积) -> (MN, 1)
    a = np.kron(a_tx, a_rx)
    return a

def generate_covariance_matrix(r, theta, snr_db, L=None):
    """生成采样协方差矩阵 (SCM)"""
    if L is None: L = cfg.L_snapshots

    # 信号功率 (假设为1)
    signal_power = 1.0
    # 噪声功率
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    # 导向矢量
    a = get_steering_vector(r, theta) # (MN, 1)

    # 信号源信号 S: (1, L)
    # 假设信号服从复高斯分布
    s = np.sqrt(signal_power/2) * (np.random.randn(1, L) + 1j * np.random.randn(1, L))

    # 接收信号 X = A * S + N
    # A * S -> (MN, L)
    X_pure = a @ s

    # 噪声 N: (MN, L)
    noise = np.sqrt(noise_power/2) * (np.random.randn(cfg.MN, L) + 1j * np.random.randn(cfg.MN, L))

    X = X_pure + noise

    # 采样协方差矩阵 R = (1/L) * X * X^H
    R = (X @ X.conj().T) / L

    # 转换为实部+虚部通道 (2, MN, MN) 用于网络输入
    R_tensor = np.stack([R.real, R.imag], axis=0)

    return R_tensor, R  # 返回 Tensor格式 和 复数格式

# =========================================================
# 2. 算法实现 (Standard Implementation)
# =========================================================

def compute_crb(snr_db, r, theta, L):
    """计算 Cramer-Rao Bound (理论下界)"""
    snr_lin = 10**(snr_db/10)
    theta_rad = np.deg2rad(theta)

    m = np.arange(cfg.M)
    n = np.arange(cfg.N)

    # 基础相位项
    psi_tx = -4*np.pi*cfg.delta_f*m*r/cfg.c + 2*np.pi*cfg.d*m*np.sin(theta_rad)/cfg.wavelength
    psi_rx = 2*np.pi*cfg.d*n*np.sin(theta_rad)/cfg.wavelength

    at = np.exp(1j * psi_tx)
    ar = np.exp(1j * psi_rx)

    # 导数计算
    # da/dr
    d_at_dr = 1j * (-4*np.pi*cfg.delta_f*m/cfg.c) * at
    da_dr = np.kron(d_at_dr, ar)

    # da/dtheta
    d_at_dt = 1j * (2*np.pi*cfg.d*m*np.cos(theta_rad)/cfg.wavelength) * at
    d_ar_dt = 1j * (2*np.pi*cfg.d*n*np.cos(theta_rad)/cfg.wavelength) * ar
    da_dt = np.kron(d_at_dt, ar) + np.kron(at, d_ar_dt)

    # Fisher Information Matrix
    # D = [da/dr, da/dt]
    D = np.column_stack((da_dr, da_dt * np.pi/180)) # 转换成角度制

    # FIM = 2 * L * SNR * real(D^H * D) (简化版，假设单目标且噪声白化)
    # 更严谨版本: FIM_ij = 2*L * SNR * Re( (d_i)^H (I - a a^H / MN) d_j ) ??
    # 对于单目标且已知方差，FIM = 2 * L / sigma^2 * Re(D^H D)
    # 这里 sigma^2 = 1/SNR
    FIM = 2 * L * snr_lin * np.real(D.conj().T @ D)

    try:
        CRB = np.linalg.inv(FIM)
        return np.sqrt(CRB[0,0]), np.sqrt(CRB[1,1])
    except:
        return np.nan, np.nan

def music_algorithm(R, grid_r, grid_theta):
    """
    MUSIC 算法 (Standard 2D Grid Search)
    """
    # 1. 特征分解
    w, v = np.linalg.eigh(R)
    # 噪声子空间 (特征值小的对应噪声)
    Un = v[:, :-1] # (MN, MN-1)

    # 2. 构建字典 (Grid)
    # 为了速度，我们使用矩阵运算，但这会消耗内存。
    # 如果内存不足，可以改为循环。这里 M*N=100，网格点数不多，直接矩阵运算。
    R_mesh, T_mesh = np.meshgrid(grid_r, grid_theta, indexing='ij')
    r_flat = R_mesh.flatten()
    t_flat = T_mesh.flatten()

    # 批量生成导向矢量 (Vectorized Steering Vector Generation)
    # A: (MN, N_grid)
    # 这是一个稍微复杂点的广播，为了代码清晰，我们手写一下
    M, N = cfg.M, cfg.N
    n_grid = len(r_flat)

    m_vec = np.arange(M).reshape(-1, 1)
    n_vec = np.arange(N).reshape(-1, 1)
    t_rad = np.deg2rad(t_flat)

    # Phase terms
    phi_tx = -4*np.pi*cfg.delta_f * m_vec * r_flat / cfg.c + \
              2*np.pi*cfg.d * m_vec * np.sin(t_rad) / cfg.wavelength
    phi_rx = 2*np.pi*cfg.d * n_vec * np.sin(t_rad) / cfg.wavelength

    At = np.exp(1j * phi_tx) # (M, N_grid)
    Ar = np.exp(1j * phi_rx) # (N, N_grid)

    # Khatri-Rao product A = At o Ar -> (MN, N_grid)
    # A[m*N + n, k] = At[m, k] * Ar[n, k]
    # Reshape method:
    A = (At[:, np.newaxis, :] * Ar[np.newaxis, :, :]).reshape(M*N, n_grid)

    # 3. 计算谱 P = 1 / |Un^H * A|^2
    # Un^H * A -> (MN-1, N_grid)
    proj = Un.conj().T @ A
    denom = np.sum(np.abs(proj)**2, axis=0)

    # 4. 找最大峰值
    idx = np.argmin(denom)

    return r_flat[idx], t_flat[idx]

def omp_algorithm(R, grid_r, grid_theta):
    """
    OMP 算法 (此处退化为匹配追踪 Beamforming，因为是单目标)
    """
    # 取主特征向量作为信号代理
    w, v = np.linalg.eigh(R)
    y = v[:, -1] # (MN,)

    # 构建字典 (与 MUSIC 相同，可以复用代码，这里为了独立性重写)
    R_mesh, T_mesh = np.meshgrid(grid_r, grid_theta, indexing='ij')
    r_flat = R_mesh.flatten()
    t_flat = T_mesh.flatten()

    M, N = cfg.M, cfg.N
    n_grid = len(r_flat)
    m_vec = np.arange(M).reshape(-1, 1); n_vec = np.arange(N).reshape(-1, 1)
    t_rad = np.deg2rad(t_flat)

    phi_tx = -4*np.pi*cfg.delta_f * m_vec * r_flat / cfg.c + 2*np.pi*cfg.d * m_vec * np.sin(t_rad) / cfg.wavelength
    phi_rx = 2*np.pi*cfg.d * n_vec * np.sin(t_rad) / cfg.wavelength
    At = np.exp(1j * phi_tx); Ar = np.exp(1j * phi_rx)
    A = (At[:, np.newaxis, :] * Ar[np.newaxis, :, :]).reshape(M*N, n_grid)

    # 归一化导向矢量
    A = A / np.sqrt(M*N)

    # 匹配: Maximize |y^H * a|
    corr = np.abs(y.conj().T @ A)
    idx = np.argmax(corr)

    return r_flat[idx], t_flat[idx]

def esprit_algorithm(R):
    """
    TLS-ESPRIT 算法
    """
    M, N = cfg.M, cfg.N
    # 信号子空间 (单目标 K=1)
    w, v = np.linalg.eigh(R)
    Us = v[:, -1:] # (MN, 1)

    # 1. 估计角度 (利用接收阵列不变性)
    # J1: 选择前 N-1 个接收阵元; J2: 选择后 N-1 个
    # 对应到 MN 维度:
    # J1 选择所有 m 的前 N-1 个 n
    mask1 = np.tile([True]*(N-1) + [False], M)
    mask2 = np.tile([False] + [True]*(N-1), M)

    Us1 = Us[mask1, :]
    Us2 = Us[mask2, :]

    # Phi_rx = (Us1^H Us1)^-1 Us1^H Us2 (LS) or TLS
    # 简单 LS
    try:
        Phi_rx = np.linalg.lstsq(Us1, Us2, rcond=None)[0]
        evals_rx = np.linalg.eigvals(Phi_rx)
        phi_rx = np.angle(evals_rx[0])

        # phi_rx = 2*pi*d*sin(theta)/lam
        sin_theta = phi_rx * cfg.wavelength / (2 * np.pi * cfg.d)
        # 截断防溢出
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        theta_est = np.rad2deg(np.arcsin(sin_theta))
    except:
        theta_est = 0.0

    # 2. 估计距离 (利用发射阵列不变性)
    # J1: 选择前 M-1 个发射; J2: 选择后 M-1 个
    # block selection
    Us1_tx = Us[:(M-1)*N, :]
    Us2_tx = Us[N:, :]

    try:
        Phi_tx = np.linalg.lstsq(Us1_tx, Us2_tx, rcond=None)[0]
        evals_tx = np.linalg.eigvals(Phi_tx)
        phi_tx_measured = np.angle(evals_tx[0])

        # phi_tx_measured = -4*pi*delta_f*r/c + 2*pi*d*sin(theta)/lam
        # 我们已知 theta_est，可以消去第二项
        term_theta = 2 * np.pi * cfg.d * sin_theta / cfg.wavelength
        # phi_r = phi_tx_measured - term_theta = -4*pi*delta_f*r/c
        phi_r = phi_tx_measured - term_theta

        # r = -phi_r * c / (4*pi*delta_f)
        r_est = -phi_r * cfg.c / (4 * np.pi * cfg.delta_f)

        # 解模糊 (De-ambiguity)
        # 真正的物理 R_max
        R_amb = cfg.c / (2 * cfg.delta_f)

        # 将 r_est 映射到 [0, R_amb]
        while r_est < 0: r_est += R_amb
        while r_est > R_amb: r_est -= R_amb

        # 简单截断到 config 范围
        r_est = np.clip(r_est, cfg.r_min, cfg.r_max)

    except:
        r_est = cfg.r_max / 2.0

    return r_est, theta_est

# =========================================================
# 3. 辅助函数
# =========================================================

def load_models(device, L):
    """加载 CVNN 和 Real-CNN 模型"""
    # 路径
    path_cvnn = f"{cfg.checkpoint_dir}/fda_cvnn_L{L}_best.pth"
    path_cvnn_fallback = f"{cfg.checkpoint_dir}/fda_cvnn_best.pth"
    path_rcnn = f"{cfg.checkpoint_dir}/real_cnn_best.pth"

    # 加载 CVNN
    model_cvnn = FDA_CVNN().to(device)
    if os.path.exists(path_cvnn):
        print(f"✅ 加载 CVNN 模型: {path_cvnn}")
        ckpt = torch.load(path_cvnn, map_location=device)
        # 处理可能的 module. 前缀
        state_dict = {k.replace('module.', ''): v for k, v in ckpt.items()}
        if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        model_cvnn.load_state_dict(state_dict, strict=False)
    elif os.path.exists(path_cvnn_fallback):
        print(f"⚠️ 使用通用模型: {path_cvnn_fallback}")
        ckpt = torch.load(path_cvnn_fallback, map_location=device)
        model_cvnn.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    else:
        print("⚠️ 未找到 CVNN 模型，使用随机初始化！")

    # 加载 Real-CNN
    model_rcnn = RealCNN().to(device)
    if os.path.exists(path_rcnn):
        print(f"✅ 加载 Real-CNN 模型: {path_rcnn}")
        ckpt = torch.load(path_rcnn, map_location=device)
        model_rcnn.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)

    model_cvnn.eval()
    model_rcnn.eval()
    return model_cvnn, model_rcnn

# =========================================================
# 4. 主流程
# =========================================================

def run_benchmark(L_snapshots=None, num_samples=200):
    if L_snapshots is None: L_snapshots = cfg.L_snapshots
    device = cfg.device
    print(f"\n🚀 开始评测: L={L_snapshots}, Samples={num_samples}")
    print(f"   物理参数: f0={cfg.f0/1e9}G, delta_f={cfg.delta_f/1e3}k, R_max={cfg.R_max:.0f}m")

    # 1. 准备模型
    cvnn, rcnn = load_models(device, L_snapshots)

    # 2. 准备网格 (MUSIC/OMP)
    # 策略：为了“公平”且展示网格效应，步长设为物理分辨率的 1/10
    # 距离分辨率 Res_r = c / (2 * M * delta_f) = 3e8 / (20 * 70e3) ≈ 214 m
    # 角度分辨率 Res_t = lambda / (N * d) ≈ 2 / N (rad) ≈ 11.5 度
    res_r = cfg.c / (2 * cfg.M * cfg.delta_f)
    res_t = np.rad2deg(cfg.wavelength / (cfg.N * cfg.d))

    grid_factor = 10.0 # 1/10 分辨率
    step_r = res_r / grid_factor
    step_t = res_t / grid_factor

    grid_r = np.arange(cfg.r_min, cfg.r_max, step_r)
    grid_theta = np.arange(cfg.theta_min, cfg.theta_max, step_t)

    print(f"   网格设置: Range Step={step_r:.2f}m, Angle Step={step_t:.2f}°")
    print(f"   网格点数: {len(grid_r)} x {len(grid_theta)} = {len(grid_r)*len(grid_theta)}")

    # 3. SNR 循环
    snr_list = [-10, -5, 0, 5, 10, 15, 20]
    methods = ['CVNN', 'Real-CNN', 'MUSIC', 'ESPRIT', 'OMP']
    results = {m: {'r': [], 't': [], 'time': []} for m in methods}
    results['CRB'] = {'r': [], 't': []}

    for snr in snr_list:
        print(f"Running SNR = {snr} dB ...")

        # 临时存储误差
        errs = {m: {'r': [], 't': [], 'time': []} for m in methods}
        crb_sums = {'r': [], 't': []}

        for _ in tqdm(range(num_samples), leave=False):
            # 生成真值
            r_true = np.random.uniform(cfg.r_min, cfg.r_max)
            t_true = np.random.uniform(cfg.theta_min, cfg.theta_max)

            # 生成数据
            R_tensor, R_complex = generate_covariance_matrix(r_true, t_true, snr, L_snapshots)
            R_torch = torch.FloatTensor(R_tensor).unsqueeze(0).to(device)

            # --- CVNN ---
            t0 = time.time()
            with torch.no_grad():
                pred = cvnn(R_torch).cpu().numpy()[0] # [r_norm, t_norm]
            t_cvnn = time.time() - t0

            # 反归一化
            r_pred_cvnn = pred[0] * cfg.r_max
            t_pred_cvnn = pred[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min

            errs['CVNN']['r'].append((r_pred_cvnn - r_true)**2)
            errs['CVNN']['t'].append((t_pred_cvnn - t_true)**2)
            errs['CVNN']['time'].append(t_cvnn)

            # --- Real-CNN ---
            t0 = time.time()
            with torch.no_grad():
                pred = rcnn(R_torch).cpu().numpy()[0]
            t_rcnn = time.time() - t0
            r_pred_rcnn = pred[0] * cfg.r_max
            t_pred_rcnn = pred[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min

            errs['Real-CNN']['r'].append((r_pred_rcnn - r_true)**2)
            errs['Real-CNN']['t'].append((t_pred_rcnn - t_true)**2)
            errs['Real-CNN']['time'].append(t_rcnn)

            # --- MUSIC ---
            t0 = time.time()
            r_mus, t_mus = music_algorithm(R_complex, grid_r, grid_theta)
            t_mus_end = time.time() - t0

            errs['MUSIC']['r'].append((r_mus - r_true)**2)
            errs['MUSIC']['t'].append((t_mus - t_true)**2)
            errs['MUSIC']['time'].append(t_mus_end)

            # --- OMP ---
            t0 = time.time()
            r_omp, t_omp = omp_algorithm(R_complex, grid_r, grid_theta)
            t_omp_end = time.time() - t0

            errs['OMP']['r'].append((r_omp - r_true)**2)
            errs['OMP']['t'].append((t_omp - t_true)**2)
            errs['OMP']['time'].append(t_omp_end)

            # --- ESPRIT ---
            t0 = time.time()
            r_esp, t_esp = esprit_algorithm(R_complex)
            t_esp_end = time.time() - t0

            # ESPRIT 在低信噪比下可能极度离谱，做一点截断防止 RMSE 爆炸无法看
            if abs(r_esp - r_true) < 1000: # 仅统计合理范围内的，或者都统计
                errs['ESPRIT']['r'].append((r_esp - r_true)**2)
                errs['ESPRIT']['t'].append((t_esp - t_true)**2)
            else:
                # 给一个惩罚值，避免 nan
                errs['ESPRIT']['r'].append(cfg.r_max**2)
                errs['ESPRIT']['t'].append((cfg.theta_max - cfg.theta_min)**2)

            errs['ESPRIT']['time'].append(t_esp_end)

            # --- CRB ---
            cr_r, cr_t = compute_crb(snr, r_true, t_true, L_snapshots)
            if not np.isnan(cr_r) and cr_r < cfg.r_max:
                crb_sums['r'].append(cr_r)
                crb_sums['t'].append(cr_t)

        # 统计平均 RMSE
        for m in methods:
            rmse_r = np.sqrt(np.mean(errs[m]['r']))
            rmse_t = np.sqrt(np.mean(errs[m]['t']))
            avg_time = np.mean(errs[m]['time'])

            results[m]['r'].append(rmse_r)
            results[m]['t'].append(rmse_t)
            results[m]['time'].append(avg_time)

        # CRB 平均
        results['CRB']['r'].append(np.mean(crb_sums['r']) if crb_sums['r'] else 0)
        results['CRB']['t'].append(np.mean(crb_sums['t']) if crb_sums['t'] else 0)

        print(f"   RMSE_R: CVNN={results['CVNN']['r'][-1]:.2f}m, MUSIC={results['MUSIC']['r'][-1]:.2f}m")

    return snr_list, results

# =========================================================
# 5. 绘图
# =========================================================
def plot_benchmark(snr_list, results, L):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 10))

    methods = ['CVNN', 'Real-CNN', 'MUSIC', 'ESPRIT', 'OMP']
    colors = {'CVNN': 'blue', 'Real-CNN': 'green', 'MUSIC': 'red', 'ESPRIT': 'orange', 'OMP': 'purple'}
    markers = {'CVNN': 'o', 'Real-CNN': '^', 'MUSIC': 's', 'ESPRIT': 'd', 'OMP': 'v'}

    # 1. 距离 RMSE
    ax1 = plt.subplot(2, 3, 1)
    for m in methods:
        plt.plot(snr_list, results[m]['r'], label=m, color=colors[m], marker=markers[m])
    plt.plot(snr_list, results['CRB']['r'], 'k--', label='CRB', linewidth=2)
    plt.yscale('log')
    plt.xlabel('SNR (dB)'); plt.ylabel('RMSE Range (m)')
    plt.title(f'Range Accuracy (L={L})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    # 2. 角度 RMSE
    ax2 = plt.subplot(2, 3, 2)
    for m in methods:
        plt.plot(snr_list, results[m]['t'], label=m, color=colors[m], marker=markers[m])
    plt.plot(snr_list, results['CRB']['t'], 'k--', label='CRB', linewidth=2)
    plt.yscale('log')
    plt.xlabel('SNR (dB)'); plt.ylabel('RMSE Angle (deg)')
    plt.title(f'Angle Accuracy (L={L})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    # 3. 运行时间
    ax3 = plt.subplot(2, 3, 3)
    for m in methods:
        plt.plot(snr_list, [t*1000 for t in results[m]['time']], label=m, color=colors[m], marker=markers[m])
    plt.yscale('log')
    plt.xlabel('SNR (dB)'); plt.ylabel('Time (ms)')
    plt.title('Inference Time')
    plt.legend()
    plt.grid(True)

    # 4. 综合雷达图
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    # 归一化分数 (越小越好 -> 越大越好)
    # 取 SNR=20dB 的点做展示
    idx = -1
    stats = {}
    for m in methods:
        r_err = results[m]['r'][idx]
        t_err = results[m]['t'][idx]
        time_v = results[m]['time'][idx]
        stats[m] = [r_err, t_err, time_v]

    # 计算最大值用于归一化
    max_vals = [
        max([v[0] for v in stats.values()]),
        max([v[1] for v in stats.values()]),
        max([v[2] for v in stats.values()])
    ]

    angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist() + [0]
    labels = ['Range Acc', 'Angle Acc', 'Speed']

    for m in methods:
        # 分数计算：1 - (val / max)，值越小分数越高
        vals = [
            1 - stats[m][0]/(max_vals[0]+1e-6),
            1 - stats[m][1]/(max_vals[1]+1e-6),
            1 - stats[m][2]/(max_vals[2]+1e-6)
        ]
        vals += [vals[0]]
        ax4.plot(angles, vals, label=m, color=colors[m])
        ax4.fill(angles, vals, alpha=0.1, color=colors[m])

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(labels)
    ax4.set_title('Comprehensive Score (at max SNR)')

    # 5. 表格
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    col_labels = ['Method', 'RMSE_R (m)', 'RMSE_T (deg)', 'Time (ms)']
    cell_text = []
    # 按 RMSE_R 排序
    sorted_methods = sorted(methods, key=lambda x: results[x]['r'][idx])
    for m in sorted_methods:
        cell_text.append([
            m,
            f"{results[m]['r'][idx]:.2f}",
            f"{results[m]['t'][idx]:.2f}",
            f"{results[m]['time'][idx]*1000:.2f}"
        ])

    table = ax5.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.scale(1, 1.5)
    ax5.set_title('Performance @ Max SNR')

    plt.tight_layout()
    plt.savefig(f'benchmark_result_L{L}.png')
    print(f"📊 图表已保存: benchmark_result_L{L}.png")


if __name__ == "__main__":
    # 使用 config 中的默认快拍数，或者手动指定
    L = cfg.L_snapshots
    snr_list, results = run_benchmark(L_snapshots=L, num_samples=200)
    plot_benchmark(snr_list, results, L)