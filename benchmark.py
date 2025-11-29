"""
FDA-MIMO 雷达参数估计对比实验脚本 - 最终修正版
关键改进:
1. MUSIC 添加两级搜索 (粗搜索 + 局部细化)
2. ESPRIT 添加相位解模糊
3. 使用完整 FIM 计算 CRB
4. OMP 字典归一化
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


def compute_crb_average(snr_db, L=None, num_samples=50):
    """
    计算多个随机目标位置的平均 CRB
    """
    crb_r_list = []
    crb_theta_list = []

    for _ in range(num_samples):
        r_true = np.random.uniform(0, cfg.r_max)
        theta_true = np.random.uniform(cfg.theta_min, cfg.theta_max)
        crb_r, crb_theta = compute_crb_full(snr_db, r_true, theta_true, L)
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
    2. 局部细化搜索 (可选)
    """
    # 特征分解
    w, v = np.linalg.eigh(R)
    idx = np.argsort(w)
    w = w[idx]
    v = v[:, idx]

    Un = v[:, :-1]

    def compute_music_spectrum(r, theta):
        """计算 MUSIC 谱值"""
        a = get_steering_vector(r, theta)
        proj = Un.conj().T @ a
        denom = np.sum(np.abs(proj)**2)
        return 1.0 / (denom + 1e-10)

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
        # phase_tx = -4π*Δf*r/c + 2π*d*sin(θ)/λ
        phi_angle = 2 * np.pi * cfg.d * sin_theta / cfg.wavelength
        diff_phase = phase_tx - phi_angle

        # 计算距离 (带解模糊)
        r_est = -diff_phase * cfg.c / (4 * np.pi * cfg.delta_f)

        # === 相位解模糊 ===
        # 无模糊距离
        max_unambiguous_r = cfg.c / (2 * cfg.delta_f)

        # 如果算出负值，加周期
        while r_est < 0:
            r_est += max_unambiguous_r

        # 如果超出范围，取模
        while r_est > cfg.r_max:
            r_est -= max_unambiguous_r

        r_est = np.clip(r_est, 0, cfg.r_max)

    except Exception as e:
        # 如果失败，返回中间值
        r_est = cfg.r_max / 2
        theta_est = 0

    return float(np.real(r_est)), float(np.real(theta_est))


# ==========================================
# 3. OMP (已归一化)
# ==========================================
def omp_2d(R, r_grid, theta_grid, K=1):
    """
    正交匹配追踪，字典原子已归一化
    """
    MN = cfg.M * cfg.N

    w, v = np.linalg.eigh(R)
    y = v[:, -1]
    y = y / np.linalg.norm(y)

    num_r = len(r_grid)
    num_theta = len(theta_grid)
    A = np.zeros((MN, num_r * num_theta), dtype=complex)

    # 构造归一化字典
    for i, r in enumerate(r_grid):
        for j, theta in enumerate(theta_grid):
            a = get_steering_vector(r, theta)
            A[:, i * num_theta + j] = a / np.linalg.norm(a)

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
# 4. RAM (FDA专用)
# ==========================================
def ram_fda(R, r_grid, theta_grid, max_iter=10):
    """
    降维交替最小化算法 (用 ESPRIT 初始化)
    """
    M, N = cfg.M, cfg.N

    w, v = np.linalg.eigh(R)
    Un = v[:, :-1]

    def compute_spectrum(r, theta):
        a = get_steering_vector(r, theta)
        proj = Un.conj().T @ a
        return 1.0 / (np.sum(np.abs(proj)**2) + 1e-10)

    # 用 ESPRIT 快速初始化
    r_est, theta_est = esprit_2d_robust(R, M, N)

    # 交替迭代优化
    for _ in range(max_iter):
        # 固定 theta，优化 r
        max_spectrum = -1
        for r in r_grid:
            spectrum = compute_spectrum(r, theta_est)
            if spectrum > max_spectrum:
                max_spectrum = spectrum
                r_est = r

        # 固定 r，优化 theta
        max_spectrum = -1
        for theta in theta_grid:
            spectrum = compute_spectrum(r_est, theta)
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

    # 加载模型 (保持不变)
    cvnn = FDA_CVNN().to(device)
    cvnn_path = "checkpoints/fda_cvnn_best.pth"
    if os.path.exists(cvnn_path):
        try:
            checkpoint = torch.load(cvnn_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                cvnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                cvnn.load_state_dict(checkpoint)
            print(f"✓ 成功加载 CVNN 权重")
        except Exception as e:
            print(f"✗ 加载 CVNN 失败: {e}")
    cvnn.eval()

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
            print(f"✓ 成功加载 Real-CNN 权重")
            has_real_cnn = True
        except:
            pass
    real_cnn.eval()

    # 参数设置
    snr_list = [-5, 0, 5, 10, 15, 20]
    num_samples = 50

    methods = ["CVNN", "Real-CNN", "MUSIC", "ESPRIT", "OMP", "RAM"]
    results = {m: {"rmse_r": [], "rmse_theta": [], "time": []} for m in methods}
    results["CRB"] = {"rmse_r": [], "rmse_theta": [], "time": []}

    # 搜索网格 (MUSIC 粗网格，会自动细化)
    r_grid = np.linspace(0, cfg.r_max, 100)      # 20m 步长
    theta_grid = np.linspace(cfg.theta_min, cfg.theta_max, 60)  # 2度 步长

    r_grid_omp = np.linspace(0, cfg.r_max, 100)
    theta_grid_omp = np.linspace(cfg.theta_min, cfg.theta_max, 40)

    print(f"\n{'='*60}")
    print(f"对比实验配置:")
    print(f"  样本数: {num_samples}")
    print(f"  MUSIC 粗网格: {len(r_grid)}×{len(theta_grid)} (+ 自动细化)")
    print(f"  OMP 字典: {len(r_grid_omp)}×{len(theta_grid_omp)} 原子")
    print(f"{'='*60}\n")

    for snr in snr_list:
        print(f"📊 测试 SNR = {snr} dB ...")

        errors = {m: {"r": [], "theta": [], "time": []} for m in methods}

        for _ in tqdm(range(num_samples), desc=f"SNR={snr}dB"):
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

            # MUSIC (两级搜索)
            t0 = time.time()
            r_pred, theta_pred = music_2d_refined(R_complex, r_grid, theta_grid, refine=True)
            t1 = time.time()
            errors["MUSIC"]["r"].append((r_pred - r_true)**2)
            errors["MUSIC"]["theta"].append((theta_pred - theta_true)**2)
            errors["MUSIC"]["time"].append(t1 - t0)

            # ESPRIT (改进版)
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

            # RAM
            t0 = time.time()
            r_pred, theta_pred = ram_fda(R_complex, r_grid, theta_grid, max_iter=5)
            t1 = time.time()
            errors["RAM"]["r"].append((r_pred - r_true)**2)
            errors["RAM"]["theta"].append((theta_pred - theta_true)**2)
            errors["RAM"]["time"].append(t1 - t0)

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

        # 打印结果表格
        print(f"\n  {'Method':<12} {'RMSE_r (m)':>14} {'RMSE_θ (°)':>14} {'Time (ms)':>14}")
        print(f"  {'-'*56}")
        for m in methods:
            rmse_r = results[m]["rmse_r"][-1]
            rmse_theta = results[m]["rmse_theta"][-1]
            avg_time = results[m]["time"][-1] * 1000
            print(f"  {m:<12} {rmse_r:>14.3f} {rmse_theta:>14.3f} {avg_time:>14.2f}")
        print(f"  {'CRB':<12} {crb_r:>14.3f} {crb_theta:>14.3f} {'(theoretical)':>14}")
        print()

    return snr_list, results


# ==========================================
# 6. 绘图 (保持不变)
# ==========================================
def plot_results(snr_list, results):
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass

    methods = [m for m in results.keys() if m != "CRB"]
    colors = {'CVNN': '#1f77b4', 'Real-CNN': '#2ca02c', 'MUSIC': '#d62728',
              'ESPRIT': '#ff7f0e', 'OMP': '#9467bd', 'RAM': '#8c564b'}
    markers = {'CVNN': 'o', 'Real-CNN': '^', 'MUSIC': 's',
               'ESPRIT': 'd', 'OMP': 'v', 'RAM': 'p'}

    plt.figure(figsize=(18, 12))

    # 图1: 距离精度
    plt.subplot(2, 2, 1)
    for m in methods:
        if m == "ESPRIT" and np.mean(results[m]["rmse_r"]) > 500:
            continue
        plt.plot(snr_list, results[m]["rmse_r"],
                 color=colors.get(m, 'gray'), marker=markers.get(m, 'x'),
                 label=m, linewidth=2.5, markersize=9)
    plt.plot(snr_list, results["CRB"]["rmse_r"],
             'k--', label='CRB', linewidth=3, alpha=0.7)
    plt.xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    plt.ylabel('RMSE Range (m)', fontsize=13, fontweight='bold')
    plt.title('Range Estimation vs. SNR', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.yscale('log')

    # 图2: 角度精度
    plt.subplot(2, 2, 2)
    for m in methods:
        plt.plot(snr_list, results[m]["rmse_theta"],
                 color=colors.get(m, 'gray'), marker=markers.get(m, 'x'),
                 label=m, linewidth=2.5, markersize=9)
    plt.plot(snr_list, results["CRB"]["rmse_theta"],
             'k--', label='CRB', linewidth=3, alpha=0.7)
    plt.xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    plt.ylabel('RMSE Angle (°)', fontsize=13, fontweight='bold')
    plt.title('Angle Estimation vs. SNR', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.yscale('log')

    # 图3: 耗时对比
    plt.subplot(2, 2, 3)
    for m in methods:
        t_ms = [t * 1000 for t in results[m]["time"]]
        plt.plot(snr_list, t_ms,
                 color=colors.get(m, 'gray'), marker=markers.get(m, 'x'),
                 label=m, linewidth=2.5, markersize=9)
    plt.xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    plt.ylabel('Inference Time (ms)', fontsize=13, fontweight='bold')
    plt.title('Computational Efficiency', fontsize=15, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which="both")
    plt.legend(fontsize=10, loc='best')

    # 图4: 性能表格
    plt.subplot(2, 2, 4)
    plt.axis('off')

    all_methods = methods + ["CRB"]
    table_data = [['Method', 'Avg RMSE_r', 'Avg RMSE_θ', 'Avg Time']]
    for m in all_methods:
        avg_r = np.mean(results[m]["rmse_r"])
        avg_theta = np.mean(results[m]["rmse_theta"])
        if m == "CRB":
            table_data.append([m, f'{avg_r:.4f}m', f'{avg_theta:.4f}°', '(bound)'])
        else:
            avg_t = np.mean(results[m]["time"]) * 1000
            table_data.append([m, f'{avg_r:.2f}m', f'{avg_theta:.2f}°', f'{avg_t:.2f}ms'])

    table = plt.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.22, 0.24, 0.24, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    crb_row = len(all_methods)
    for i in range(4):
        table[(crb_row, i)].set_facecolor('#E0E0E0')

    best_r_idx = np.argmin([np.mean(results[m]["rmse_r"]) for m in methods]) + 1
    best_theta_idx = np.argmin([np.mean(results[m]["rmse_theta"]) for m in methods]) + 1
    best_time_idx = np.argmin([np.mean(results[m]["time"]) for m in methods]) + 1

    table[(best_r_idx, 1)].set_facecolor('#90EE90')
    table[(best_theta_idx, 2)].set_facecolor('#90EE90')
    table[(best_time_idx, 3)].set_facecolor('#90EE90')

    plt.title('Performance Summary\n(Green=Best, Gray=Theoretical Bound)',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('benchmark_comparison_final.png', dpi=300, bbox_inches='tight')
    print("\n✓ 图表已保存: benchmark_comparison_final.png")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FDA-MIMO 雷达参数估计对比实验 - 最终版本")
    print("="*60)
    snr_list, results = run_benchmark()
    plot_results(snr_list, results)
    print("\n✓ 实验完成！")
