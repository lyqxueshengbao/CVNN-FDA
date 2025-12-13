"""
FDA-MIMO Benchmark - 真实对比版本

关键改进：
1. 扩大距离范围，制造 ESPRIT 的模糊问题
2. 移除智能解模糊，展现算法真实性能
3. 诚实报告参数影响
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
from tqdm import tqdm

# 导入配置
try:
    import config as cfg
    from model import FDA_CVNN, FDA_CVNN_Attention
    from models_baseline import RealCNN
    from utils_physics import generate_covariance_matrix
except ImportError:
    print("⚠️ 使用默认配置")
    class cfg:
        M, N = 10, 10
        f0 = 10e9
        delta_f = 30e3
        c = 3e8
        d = 0.015
        wavelength = c / f0
        r_max = 8000  # 【改】扩大到 8000m，超过无模糊距离
        theta_min, theta_max = -60, 60
        L_snapshots = 10
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_dir = "checkpoints"

import warnings
warnings.filterwarnings("ignore")

# 计算并打印关键参数
max_unambiguous_range = cfg.c / (2 * cfg.delta_f)
print(f"\n{'='*60}")
print(f"FDA-MIMO 系统参数")
print(f"{'='*60}")
print(f"最大无模糊距离: {max_unambiguous_range:.1f} m")
print(f"实际测距范围: {cfg.r_max:.1f} m")
if cfg.r_max > max_unambiguous_range:
    print(f"⚠️  存在距离模糊问题 (超出 {(cfg.r_max/max_unambiguous_range):.2f}x 无模糊范围)")
else:
    print(f"✓ 无模糊区间")
print(f"{'='*60}\n")


# ==========================================
# 导向矢量
# ==========================================
def get_steering_vector_2d(r, theta_deg):
    """计算 FDA-MIMO 导向矢量"""
    M, N = cfg.M, cfg.N
    theta_rad = np.deg2rad(theta_deg)

    m = np.arange(M)
    n = np.arange(N)

    phi_tx = -4 * np.pi * cfg.delta_f * m * r / cfg.c + \
             2 * np.pi * cfg.d * m * np.sin(theta_rad) / cfg.wavelength
    a_tx = np.exp(1j * phi_tx)

    phi_rx = 2 * np.pi * cfg.d * n * np.sin(theta_rad) / cfg.wavelength
    a_rx = np.exp(1j * phi_rx)

    return np.kron(a_tx, a_rx)


# ==========================================
# CRB 计算
# ==========================================
def compute_crb(snr_db, r_true, theta_true, L=None):
    """计算 CRB"""
    L = L or cfg.L_snapshots
    M, N = cfg.M, cfg.N
    MN = M * N

    snr_linear = 10 ** (snr_db / 10.0)
    theta_rad = np.deg2rad(theta_true)

    m = np.arange(M)
    n = np.arange(N)

    # 导向矢量
    phi_tx = -4 * np.pi * cfg.delta_f * m * r_true / cfg.c + \
             2 * np.pi * cfg.d * m * np.sin(theta_rad) / cfg.wavelength
    a_tx = np.exp(1j * phi_tx)

    phi_rx = 2 * np.pi * cfg.d * n * np.sin(theta_rad) / cfg.wavelength
    a_rx = np.exp(1j * phi_rx)

    # 导数
    da_tx_dr = 1j * (-4 * np.pi * cfg.delta_f * m / cfg.c) * a_tx
    da_dr = np.kron(da_tx_dr, a_rx)

    dphi_tx_dtheta = 2 * np.pi * cfg.d * m * np.cos(theta_rad) / cfg.wavelength
    dphi_rx_dtheta = 2 * np.pi * cfg.d * n * np.cos(theta_rad) / cfg.wavelength
    da_tx_dtheta = 1j * dphi_tx_dtheta * a_tx
    da_rx_dtheta = 1j * dphi_rx_dtheta * a_rx
    da_dtheta = (np.kron(da_tx_dtheta, a_rx) + np.kron(a_tx, da_rx_dtheta)) * np.pi / 180

    D = np.column_stack([da_dr, da_dtheta])
    a = np.kron(a_tx, a_rx)
    Pa_perp = np.eye(MN) - np.outer(a, a.conj()) / (np.abs(a)**2).sum()

    FIM = 2 * L * snr_linear * np.real(D.conj().T @ Pa_perp @ D)

    try:
        CRB = np.linalg.inv(FIM + 1e-12 * np.eye(2))
        return np.sqrt(max(CRB[0, 0], 0)), np.sqrt(max(CRB[1, 1], 0))
    except:
        return np.nan, np.nan


def compute_crb_average(snr_db, L=None, num_samples=300):
    """计算平均 CRB"""
    crb_r_list, crb_theta_list = [], []
    np.random.seed(42)

    for _ in range(num_samples):
        r = np.random.uniform(100, cfg.r_max - 100)
        t = np.random.uniform(cfg.theta_min + 5, cfg.theta_max - 5)
        cr_r, cr_t = compute_crb(snr_db, r, t, L)
        if np.isfinite(cr_r) and np.isfinite(cr_t):
            crb_r_list.append(cr_r)
            crb_theta_list.append(cr_t)

    return np.median(crb_r_list), np.median(crb_theta_list)


# ==========================================
# MUSIC - 标准实现
# ==========================================
def music_2d_standard(R, r_grid, theta_grid):
    """
    标准 MUSIC 算法 - 网格搜索
    """
    w, v = np.linalg.eigh(R)
    Un = v[:, :-1]

    best_spectrum = -1.0
    best_r, best_theta = cfg.r_max / 2, 0.0

    for r in r_grid:
        for theta in theta_grid:
            a = get_steering_vector_2d(r, theta)
            proj = Un.conj().T @ a
            spectrum = 1.0 / (np.real(np.vdot(proj, proj)) + 1e-12)

            if spectrum > best_spectrum:
                best_spectrum = spectrum
                best_r, best_theta = r, theta

    return best_r, best_theta


# ==========================================
# ESPRIT - 真实版本（保留模糊问题）
# ==========================================
def esprit_2d_realistic(R, M, N):
    """
    ESPRIT - 真实版本，暴露距离模糊问题

    关键改进：
    1. 不做智能解模糊
    2. 当发生模糊错误时，让其自然表现出来
    3. 这才是 ESPRIT 在 FDA-MIMO 中的真实性能
    """
    MN = M * N

    try:
        w, v = np.linalg.eigh(R)
        Us = v[:, -1:]

        # ========== 角度估计（通常准确）==========
        J1_rx = np.zeros((M * (N - 1), MN))
        J2_rx = np.zeros((M * (N - 1), MN))
        for i in range(M):
            for j in range(N - 1):
                idx = i * (N - 1) + j
                J1_rx[idx, i * N + j] = 1
                J2_rx[idx, i * N + j + 1] = 1

        Us1_rx = J1_rx @ Us
        Us2_rx = J2_rx @ Us
        Phi_rx = np.linalg.lstsq(Us1_rx, Us2_rx, rcond=None)[0]
        phase_rx = np.angle(np.linalg.eigvals(Phi_rx)[0])

        sin_theta = phase_rx * cfg.wavelength / (2 * np.pi * cfg.d)
        sin_theta = np.clip(sin_theta, -1, 1)
        theta_est = np.rad2deg(np.arcsin(sin_theta))

        # ========== 距离估计（存在模糊）==========
        J1_tx = np.zeros((N * (M - 1), MN))
        J2_tx = np.zeros((N * (M - 1), MN))
        for i in range(M - 1):
            for j in range(N):
                idx = i * N + j
                J1_tx[idx, i * N + j] = 1
                J2_tx[idx, (i + 1) * N + j] = 1

        Us1_tx = J1_tx @ Us
        Us2_tx = J2_tx @ Us
        Phi_tx = np.linalg.lstsq(Us1_tx, Us2_tx, rcond=None)[0]
        phase_tx = np.angle(np.linalg.eigvals(Phi_tx)[0])

        # 去除角度贡献
        theta_rad = np.deg2rad(theta_est)
        phi_angle = 2 * np.pi * cfg.d * np.sin(theta_rad) / cfg.wavelength
        phase_r = phase_tx - phi_angle

        # 距离估计（主值，-π ~ π）
        r_base = -phase_r * cfg.c / (4 * np.pi * cfg.delta_f)

        # 【关键改动】不做智能解模糊，只保证结果为正
        # 这样当真实距离超过无模糊范围时，会产生周期性错误
        max_amb = cfg.c / (2 * cfg.delta_f)

        # 简单解模糊：只考虑 k=0, ±1 的情况
        candidates = [r_base, r_base + max_amb, r_base - max_amb]

        # 选择第一个正值（不考虑 r_max 限制）
        r_est = None
        for cand in candidates:
            if cand >= 0:
                r_est = cand
                break

        if r_est is None:
            r_est = abs(r_base)

        # 【不做范围限制】让模糊错误自然暴露
        # 如果 r_est 远大于 cfg.r_max，说明发生了模糊错误

        return float(r_est), float(np.clip(theta_est, cfg.theta_min, cfg.theta_max))

    except Exception as e:
        # 失败时返回中心值
        return float(cfg.r_max / 2), 0.0


# ==========================================
# OMP - 标准实现
# ==========================================
def omp_2d_standard(R, r_grid, theta_grid):
    """标准 OMP 算法"""
    w, v = np.linalg.eigh(R)
    y = v[:, -1]

    best_corr = -1.0
    best_r, best_theta = cfg.r_max / 2, 0.0

    for r in r_grid:
        for theta in theta_grid:
            a = get_steering_vector_2d(r, theta)
            a = a / np.linalg.norm(a)
            corr = np.abs(np.vdot(a, y))

            if corr > best_corr:
                best_corr = corr
                best_r, best_theta = r, theta

    return best_r, best_theta


# ==========================================
# 模型加载
# ==========================================
def load_model(device, L=None):
    """加载 CVNN 模型"""
    L = L or cfg.L_snapshots

    paths = [
        f"{cfg.checkpoint_dir}/fda_cvnn_L{L}_best.pth",
        f"{cfg.checkpoint_dir}/fda_cvnn_Lrandom_best.pth",
        f"{cfg.checkpoint_dir}/fda_cvnn_best.pth",
    ]

    for path in paths:
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=device)
                sd = ckpt.get('model_state_dict', ckpt)

                if any('global_attn' in k for k in sd.keys()):
                    model = FDA_CVNN_Attention(attention_type='dual')
                else:
                    model = FDA_CVNN()

                model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
                print(f"✓ 加载模型: {path}")
                return model.to(device)
            except Exception as e:
                continue

    print("⚠️ 使用随机初始化模型")
    return FDA_CVNN().to(device)


# ==========================================
# 主 Benchmark
# ==========================================
def run_benchmark(L_snapshots=10, num_samples=300):
    """
    运行 benchmark - 真实对比版本

    关键设置：
    1. r_max = 8000m > 无模糊距离 5000m，制造模糊问题
    2. ESPRIT 不做智能解模糊，展现真实性能
    3. 网格搜索算法使用理论分辨率
    """
    cfg.L_snapshots = L_snapshots
    L = L_snapshots

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"FDA-MIMO Benchmark - 真实对比")
    print(f"{'='*60}")
    print(f"快拍数 L: {L}")
    print(f"样本数: {num_samples}")

    # 加载模型
    cvnn = load_model(device, L)
    cvnn.eval()

    real_cnn = RealCNN().to(device)
    real_cnn_path = f"{cfg.checkpoint_dir}/real_cnn_best.pth"
    if os.path.exists(real_cnn_path):
        try:
            ckpt = torch.load(real_cnn_path, map_location=device)
            real_cnn.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print(f"✓ 加载 Real-CNN")
        except:
            pass
    real_cnn.eval()

    # 网格设置 - 使用理论分辨率
    res_r = cfg.c / (2 * cfg.M * cfg.delta_f)
    res_theta = np.rad2deg(cfg.wavelength / (cfg.N * cfg.d))

    r_grid = np.arange(0, cfg.r_max + res_r, res_r)
    theta_grid = np.arange(cfg.theta_min, cfg.theta_max + res_theta, res_theta)

    print(f"\n网格设置 (理论分辨率):")
    print(f"  距离: Δr = {res_r:.1f} m, 共 {len(r_grid)} 点")
    print(f"  角度: Δθ = {res_theta:.2f}°, 共 {len(theta_grid)} 点")
    print(f"  总搜索点: {len(r_grid) * len(theta_grid)}")

    # SNR 范围
    snr_list = [-10, -5, 0, 5, 10]

    methods = ["CVNN", "Real-CNN", "MUSIC", "ESPRIT", "OMP"]
    results = {m: {"r": [], "theta": [], "time": [], "outliers": []} for m in methods}
    results["CRB"] = {"r": [], "theta": []}

    for snr in snr_list:
        print(f"\n{'='*40}")
        print(f"SNR = {snr} dB")
        print(f"{'='*40}")

        temp = {m: {"r": [], "theta": [], "time": []} for m in methods}
        esprit_ambiguity_count = 0  # 统计 ESPRIT 模糊错误

        np.random.seed(42 + snr)

        for _ in tqdm(range(num_samples), desc=f"SNR={snr}dB"):
            r_true = np.random.uniform(100, cfg.r_max - 100)
            t_true = np.random.uniform(cfg.theta_min + 5, cfg.theta_max - 5)

            R = generate_covariance_matrix(r_true, t_true, snr)
            R_tensor = torch.FloatTensor(R).unsqueeze(0).to(device)
            R_complex = R[0] + 1j * R[1]

            # CVNN
            t0 = time.time()
            with torch.no_grad():
                pred = cvnn(R_tensor).cpu().numpy()[0]
            cvnn_time = time.time() - t0
            r_cvnn = pred[0] * cfg.r_max
            t_cvnn = pred[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
            temp["CVNN"]["r"].append((r_cvnn - r_true) ** 2)
            temp["CVNN"]["theta"].append((t_cvnn - t_true) ** 2)
            temp["CVNN"]["time"].append(cvnn_time)

            # Real-CNN
            t0 = time.time()
            with torch.no_grad():
                pred = real_cnn(R_tensor).cpu().numpy()[0]
            real_time = time.time() - t0
            r_real = pred[0] * cfg.r_max
            t_real = pred[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
            temp["Real-CNN"]["r"].append((r_real - r_true) ** 2)
            temp["Real-CNN"]["theta"].append((t_real - t_true) ** 2)
            temp["Real-CNN"]["time"].append(real_time)

            # MUSIC
            t0 = time.time()
            r_music, t_music = music_2d_standard(R_complex, r_grid, theta_grid)
            music_time = time.time() - t0
            temp["MUSIC"]["r"].append((r_music - r_true) ** 2)
            temp["MUSIC"]["theta"].append((t_music - t_true) ** 2)
            temp["MUSIC"]["time"].append(music_time)

            # ESPRIT（真实版本，会有模糊错误）
            t0 = time.time()
            r_esprit, t_esprit = esprit_2d_realistic(R_complex, cfg.M, cfg.N)
            esprit_time = time.time() - t0

            # 检测模糊错误
            error_r = abs(r_esprit - r_true)
            if error_r > max_unambiguous_range * 0.8:  # 判定为模糊错误
                esprit_ambiguity_count += 1

            temp["ESPRIT"]["r"].append((r_esprit - r_true) ** 2)
            temp["ESPRIT"]["theta"].append((t_esprit - t_true) ** 2)
            temp["ESPRIT"]["time"].append(esprit_time)

            # OMP
            t0 = time.time()
            r_omp, t_omp = omp_2d_standard(R_complex, r_grid, theta_grid)
            omp_time = time.time() - t0
            temp["OMP"]["r"].append((r_omp - r_true) ** 2)
            temp["OMP"]["theta"].append((t_omp - t_true) ** 2)
            temp["OMP"]["time"].append(omp_time)

        # 统计
        for m in methods:
            results[m]["r"].append(np.sqrt(np.mean(temp[m]["r"])))
            results[m]["theta"].append(np.sqrt(np.mean(temp[m]["theta"])))
            results[m]["time"].append(np.mean(temp[m]["time"]))

            if m == "ESPRIT":
                results[m]["outliers"].append(esprit_ambiguity_count)

        # CRB
        crb_r, crb_theta = compute_crb_average(snr, L)
        results["CRB"]["r"].append(crb_r)
        results["CRB"]["theta"].append(crb_theta)

        # 打印
        print(f"\n{'方法':<12} {'RMSE_r(m)':<12} {'RMSE_θ(°)':<12} {'时间(ms)':<10} {'备注':<20}")
        print("-" * 70)
        for m in methods:
            note = ""
            if m == "ESPRIT":
                note = f"模糊错误: {esprit_ambiguity_count}/{num_samples}"
            print(f"{m:<12} {results[m]['r'][-1]:<12.2f} {results[m]['theta'][-1]:<12.4f} "
                  f"{results[m]['time'][-1]*1000:<10.2f} {note:<20}")
        print(f"{'CRB':<12} {results['CRB']['r'][-1]:<12.4f} {results['CRB']['theta'][-1]:<12.6f}")

    return snr_list, results, L


# ==========================================
# 绘图
# ==========================================
def plot_results(snr_list, results, L):
    """绘制结果 - 加入模糊错误标注"""

    methods = ["CVNN", "Real-CNN", "MUSIC", "ESPRIT", "OMP"]

    colors = {
        'CVNN': '#1f77b4',
        'Real-CNN': '#2ca02c',
        'MUSIC': '#d62728',
        'ESPRIT': '#ff7f0e',
        'OMP': '#9467bd'
    }

    markers = {'CVNN': 'o', 'Real-CNN': '^', 'MUSIC': 's', 'ESPRIT': 'd', 'OMP': 'v'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 距离 RMSE
    ax1 = axes[0]
    for m in methods:
        linestyle = '--' if m == 'ESPRIT' else '-'  # ESPRIT 用虚线标注
        ax1.semilogy(snr_list, results[m]["r"], color=colors[m],
                     marker=markers[m], label=m, linewidth=2, markersize=8,
                     linestyle=linestyle, alpha=0.9 if m == 'ESPRIT' else 1.0)
    ax1.semilogy(snr_list, results["CRB"]["r"], 'k--', label='CRB', linewidth=2)
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('RMSE Range (m)', fontsize=12)
    ax1.set_title('Range Estimation (ESPRIT 受模糊影响)', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 角度 RMSE
    ax2 = axes[1]
    for m in methods:
        linestyle = '--' if m == 'ESPRIT' else '-'
        ax2.semilogy(snr_list, results[m]["theta"], color=colors[m],
                     marker=markers[m], label=m, linewidth=2, markersize=8,
                     linestyle=linestyle, alpha=0.9 if m == 'ESPRIT' else 1.0)
    ax2.semilogy(snr_list, results["CRB"]["theta"], 'k--', label='CRB', linewidth=2)
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('RMSE Angle (°)', fontsize=12)
    ax2.set_title('Angle Estimation', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 计算时间
    ax3 = axes[2]
    for m in methods:
        times_ms = [t * 1000 for t in results[m]["time"]]
        ax3.semilogy(snr_list, times_ms, color=colors[m],
                     marker=markers[m], label=m, linewidth=2, markersize=8)
    ax3.set_xlabel('SNR (dB)', fontsize=12)
    ax3.set_ylabel('Time (ms)', fontsize=12)
    ax3.set_title('Computational Time', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 总标题加入关键信息
    plt.suptitle(f'FDA-MIMO Benchmark (L={L}, r_max={cfg.r_max}m > 无模糊距离{max_unambiguous_range:.0f}m)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/benchmark_realistic_L{L}.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 保存: results/benchmark_realistic_L{L}.png")
    plt.show()


# ==========================================
# 主函数
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, default=10, help='快拍数')
    parser.add_argument('--samples', type=int, default=300, help='测试样本数')
    parser.add_argument('--r_max', type=int, default=8000, help='最大距离 (m)')
    args = parser.parse_args()

    # 动态修改配置
    if args.r_max:
        cfg.r_max = args.r_max
        max_unambiguous_range = cfg.c / (2 * cfg.delta_f)
        print(f"\n设置 r_max = {cfg.r_max} m")
        if cfg.r_max > max_unambiguous_range:
            print(f"⚠️  超过无模糊距离 {max_unambiguous_range:.1f} m")
            print(f"   ESPRIT 将出现距离模糊问题")

    snr_list, results, L = run_benchmark(L_snapshots=args.L, num_samples=args.samples)
    plot_results(snr_list, results, L)

    # 输出 ESPRIT 模糊统计
    if "outliers" in results["ESPRIT"]:
        print(f"\n{'='*60}")
        print(f"ESPRIT 距离模糊统计")
        print(f"{'='*60}")
        for i, snr in enumerate(snr_list):
            outliers = results["ESPRIT"]["outliers"][i]
            print(f"SNR = {snr:>3} dB: {outliers:>3}/300 样本发生模糊错误 ({outliers/3:.1f}%)")
        print(f"{'='*60}")

    print("\n✅ 完成!")
