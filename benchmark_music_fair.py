# benchmark_music_fair.py
"""
公平对比的 MUSIC 基准测试。

与 CVNN 使用完全相同的：
1) 物理参数（config.py）
2) 导向矢量模型（utils_physics.get_steering_vector）
3) SNR 定义（按信号/噪声功率比）
4) 数据分布（全域随机采样）

额外保存 CRLB（文中常写 CRB）下界：avg_crlb_theta/avg_crlb_r。
"""

import os
import time

import numpy as np
from numpy.linalg import eig, inv

import config as cfg
from utils_physics import get_steering_vector

# ======================== 参数设置 ========================
M = cfg.M
N = cfg.N
MN = cfg.MN
f0 = cfg.f0
delta_f = cfg.delta_f
c = cfg.c
d = cfg.d
wavelength = cfg.wavelength
r_min = cfg.r_min
r_max = cfg.r_max
theta_min = cfg.theta_min
theta_max = cfg.theta_max

SNR_dB_list = [-15, -10, -5, 0, 5, 10, 15, 20]
L = 1
Monte_Carlo = 200

Grid_theta = np.arange(theta_min, theta_max + 1, 1)  # 1°
Grid_r = np.arange(r_min, r_max + 1, 50)  # 50 m


def compute_crlb(r: float, theta_deg: float, snr_db: float):
    theta_rad = np.deg2rad(theta_deg)
    sigma2 = 10 ** (-snr_db / 10)
    xi_power = 1 / sigma2
    tau = 2 * r / c

    coeff_theta = 2 * np.pi * d / wavelength
    coeff_r = 4 * np.pi * delta_f / c

    a_t = np.exp(-1j * 2 * np.pi * np.arange(M) * delta_f * tau)
    a_r = np.exp(1j * 2 * np.pi * d / wavelength * np.sin(theta_rad) * np.arange(N))

    d_ar = 1j * coeff_theta * np.cos(theta_rad) * np.arange(N) * a_r
    u_theta = np.kron(a_t, d_ar)

    d_at = -1j * coeff_r * np.arange(M) * a_t
    u_r = np.kron(d_at, a_r)

    J = np.zeros((2, 2))
    J[0, 0] = 2 * xi_power * np.real(u_theta.conj().T @ u_theta)
    J[1, 1] = 2 * xi_power * np.real(u_r.conj().T @ u_r)
    J[0, 1] = 2 * xi_power * np.real(u_theta.conj().T @ u_r)
    J[1, 0] = J[0, 1]

    try:
        CRLB = inv(J)
        crlb_theta = float(np.sqrt(CRLB[0, 0]) * 180 / np.pi)
        crlb_r = float(np.sqrt(CRLB[1, 1]))
    except Exception:
        crlb_theta = np.nan
        crlb_r = np.nan

    return crlb_theta, crlb_r


def generate_signal_consistent(r: float, theta_deg: float, snr_db: float, L_snapshots: int):
    u = get_steering_vector(r, theta_deg).reshape(-1, 1)
    s = (np.random.randn(1, L_snapshots) + 1j * np.random.randn(1, L_snapshots)) / np.sqrt(2)
    X_clean = u @ s

    noise = (np.random.randn(MN, L_snapshots) + 1j * np.random.randn(MN, L_snapshots)) / np.sqrt(2)
    power_sig = np.mean(np.abs(X_clean) ** 2)
    power_noise = power_sig / (10 ** (snr_db / 10.0))

    return X_clean + np.sqrt(power_noise) * noise


def music_2d(Y: np.ndarray, K: int = 1):
    L_snapshots = Y.shape[1]
    R = Y @ Y.conj().T / L_snapshots

    D, V = eig(R)
    idx = np.argsort(np.real(D))[::-1]
    Un = V[:, idx[K:]]

    P = np.zeros((len(Grid_theta), len(Grid_r)))
    for i, theta in enumerate(Grid_theta):
        for j, r in enumerate(Grid_r):
            a = get_steering_vector(r, theta)
            denom = np.real(a.conj().T @ Un @ Un.conj().T @ a)
            P[i, j] = 1.0 / (denom + 1e-12)

    peak_idx = int(np.argmax(P))
    theta_idx, r_idx = np.unravel_index(peak_idx, P.shape)
    return float(Grid_theta[theta_idx]), float(Grid_r[r_idx])


def run_benchmark():
    print("=" * 60)
    print("MUSIC 公平基准测试")
    print("=" * 60)
    print("物理参数 (与 CVNN 一致):")
    print(f"  M={M}, N={N}, f0={f0/1e9}GHz, Δf={delta_f/1e3}kHz")
    print(f"  距离范围: [{r_min}, {r_max}] m")
    print(f"  角度范围: [{theta_min}, {theta_max}]°")
    print(f"  快拍数 L={L}, MonteCarlo={Monte_Carlo}")
    print(f"  网格: θ步进=1°, r步进=50m")
    print("=" * 60)

    results = {
        "snr_list": SNR_dB_list,
        "rmse_theta": [],
        "rmse_r": [],
        "avg_time": [],
        "avg_crlb_theta": [],
        "avg_crlb_r": [],
    }

    for snr_db in SNR_dB_list:
        theta_errors = []
        r_errors = []
        times = []
        crlb_theta_list = []
        crlb_r_list = []

        for _ in range(Monte_Carlo):
            r_true = float(np.random.uniform(r_min, r_max))
            theta_true = float(np.random.uniform(theta_min, theta_max))

            c_theta, c_r = compute_crlb(r_true, theta_true, snr_db)
            crlb_theta_list.append(c_theta)
            crlb_r_list.append(c_r)

            Y = generate_signal_consistent(r_true, theta_true, snr_db, L)

            t_start = time.time()
            theta_est, r_est = music_2d(Y, K=1)
            times.append(time.time() - t_start)

            theta_errors.append((theta_est - theta_true) ** 2)
            r_errors.append((r_est - r_true) ** 2)

        rmse_theta = float(np.sqrt(np.mean(theta_errors)))
        rmse_r = float(np.sqrt(np.mean(r_errors)))
        avg_time = float(np.mean(times) * 1000.0)
        avg_crlb_theta = float(np.nanmean(crlb_theta_list))
        avg_crlb_r = float(np.nanmean(crlb_r_list))

        results["rmse_theta"].append(rmse_theta)
        results["rmse_r"].append(rmse_r)
        results["avg_time"].append(avg_time)
        results["avg_crlb_theta"].append(avg_crlb_theta)
        results["avg_crlb_r"].append(avg_crlb_r)

        print(f"SNR={snr_db:3d}dB | RMSE_θ={rmse_theta:6.2f}° | RMSE_r={rmse_r:7.2f}m | Time={avg_time:6.2f}ms")

    np.savez("music_fair_benchmark.npz", **results)

    data_to_save = np.column_stack((
        np.array(results["snr_list"], dtype=float),
        np.array(results["rmse_theta"], dtype=float),
        np.array(results["rmse_r"], dtype=float),
        np.array(results["avg_time"], dtype=float),
        np.array(results["avg_crlb_theta"], dtype=float),
        np.array(results["avg_crlb_r"], dtype=float),
    ))

    np.savetxt(
        "music_fair_benchmark.txt",
        data_to_save,
        header="SNR_dB  RMSE_theta_MUSIC  RMSE_r_MUSIC  Time_ms  CRLB_theta  CRLB_r",
        fmt="%.6f",
    )

    os.makedirs("algorithm_comparison/单快拍", exist_ok=True)
    np.savetxt(
        "algorithm_comparison/单快拍/music_fair_rmse_data.txt",
        data_to_save,
        header="SNR_dB  RMSE_theta_MUSIC  RMSE_r_MUSIC  Time_ms  CRLB_theta  CRLB_r",
        fmt="%.6f",
    )

    print("\n结果已保存:")
    print("  - music_fair_benchmark.npz")
    print("  - music_fair_benchmark.txt")
    print("  - algorithm_comparison/单快拍/music_fair_rmse_data.txt")

    return results


if __name__ == "__main__":
    run_benchmark()

