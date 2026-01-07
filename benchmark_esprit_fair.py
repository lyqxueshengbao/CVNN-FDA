# benchmark_esprit_fair.py
"""
公平对比的 ESPRIT 基准测试 (智能相位解卷绕版)
针对 Delta_f=70kHz, R_max=2142m, 测试范围 2000m 的极端临界情况。
采用 Circular Modulo + Range Filtering 策略。
"""
import numpy as np
from numpy.linalg import eig, pinv, inv
import time
import config as cfg
from utils_physics import get_steering_vector
from pathlib import Path

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

# 计算最大不模糊距离
R_unambiguous = c / (2 * delta_f)
print(f"ESPRIT 物理参数检查:")
print(f"  Delta_f = {delta_f / 1000} kHz")
print(f"  R_max (循环周期) = {R_unambiguous:.2f} m")
print(f"  测试范围 = [{r_min}, {r_max}] m")
print(f"  安全间隙 (Margin) = {R_unambiguous - r_max:.2f} m")
if r_max > R_unambiguous:
    print("!! 警告: 测试范围超过了物理不模糊距离，必然存在不可解的混叠误差 !!")


# ======================== CRLB 计算 ========================
def compute_crlb(r, theta_deg, snr_db):
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
        crlb_theta = np.sqrt(CRLB[0, 0]) * 180 / np.pi
        crlb_r = np.sqrt(CRLB[1, 1])
    except:
        crlb_theta = np.nan
        crlb_r = np.nan
    return crlb_theta, crlb_r


# ======================== 算法实现 ========================
def generate_signal_consistent(r, theta_deg, snr_db, L):
    u = get_steering_vector(r, theta_deg).reshape(-1, 1)
    s = (np.random.randn(1, L) + 1j * np.random.randn(1, L)) / np.sqrt(2)
    X_clean = u @ s
    noise = (np.random.randn(MN, L) + 1j * np.random.randn(MN, L)) / np.sqrt(2)
    power_sig = np.mean(np.abs(X_clean) ** 2)
    power_noise = power_sig / (10 ** (snr_db / 10.0))
    return X_clean + np.sqrt(power_noise) * noise


def esprit_2d(Y, K=1):
    def wrap_to_pi(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    L_snapshots = Y.shape[1]
    R = Y @ Y.conj().T / L_snapshots
    D, V = eig(R)
    idx = np.argsort(np.real(D))[::-1]
    Es = V[:, idx[:K]]

    I_M = np.eye(M)
    J_r1 = np.hstack([np.eye(N - 1), np.zeros((N - 1, 1))])
    J_r2 = np.hstack([np.zeros((N - 1, 1)), np.eye(N - 1)])
    JR1 = np.kron(I_M, J_r1)
    JR2 = np.kron(I_M, J_r2)

    I_N = np.eye(N)
    J_t1 = np.hstack([np.eye(M - 1), np.zeros((M - 1, 1))])
    J_t2 = np.hstack([np.zeros((M - 1, 1)), np.eye(M - 1)])
    JT1 = np.kron(J_t1, I_N)
    JT2 = np.kron(J_t2, I_N)

    Psi_theta = pinv(JR1 @ Es) @ (JR2 @ Es)
    eig_theta, _ = eig(Psi_theta)
    theta_est = np.degrees(np.arcsin(np.angle(eig_theta) * wavelength / (2 * np.pi * d)))

    Psi_r = pinv(JT1 @ Es) @ (JT2 @ Es)
    eig_r, _ = eig(Psi_r)

    # 关键修正：发射维移位特征值相位同时包含“距离项 + 发射阵列角度项”
    # a_tx(m) = exp(j * (-4πΔf * m * r / c + 2π d * m * sinθ / λ))
    # 相邻阵元移位的特征值相位：phi_total = (-4πΔf r / c) + (2π d sinθ / λ)
    # 因此先用 theta_est 去除发射角度项，再反推 r
    theta_est_rad = np.deg2rad(theta_est[0])
    phi_total = np.angle(eig_r[0])
    phi_tx = 2 * np.pi * d * np.sin(theta_est_rad) / wavelength
    phi_range = wrap_to_pi(phi_total - phi_tx)

    r_est_raw = -(phi_range * c) / (4 * np.pi * delta_f)

    # --- 智能解卷绕逻辑 ---
    # 将结果映射到主值域，并结合已知距离范围选择最合理的折返圈数
    r_mod = np.mod(r_est_raw, R_unambiguous)            # [0, R_unambiguous)
    candidates = [r_mod, r_mod - R_unambiguous]         # 一个在 [0,Rmax)，一个在 (-Rmax,0]

    def dist_to_interval(x, lo, hi):
        if x < lo:
            return lo - x
        if x > hi:
            return x - hi
        return 0.0

    r_est = min(candidates, key=lambda x: dist_to_interval(x, r_min, r_max))
    r_est = float(np.clip(r_est, r_min, r_max))

    return theta_est[0], r_est


def run_benchmark():
    print("=" * 60)
    print("ESPRIT 公平基准测试 (Running...)")
    print("=" * 60)

    results = {'snr_list': SNR_dB_list, 'rmse_theta': [], 'rmse_r': [], 'avg_time': [], 'avg_crlb_theta': [],
               'avg_crlb_r': []}

    for snr_db in SNR_dB_list:
        theta_errors = []
        r_errors = []
        times = []
        crlb_theta_list = []
        crlb_r_list = []

        for mc in range(Monte_Carlo):
            r_true = np.random.uniform(r_min, r_max)
            theta_true = np.random.uniform(theta_min, theta_max)

            c_theta, c_r = compute_crlb(r_true, theta_true, snr_db)
            crlb_theta_list.append(c_theta)
            crlb_r_list.append(c_r)

            Y = generate_signal_consistent(r_true, theta_true, snr_db, L)

            t_start = time.time()
            try:
                theta_est, r_est = esprit_2d(Y, K=1)

                theta_errors.append((theta_est - theta_true) ** 2)
                r_errors.append((r_est - r_true) ** 2)
                times.append(time.time() - t_start)
            except:
                continue

        rmse_theta = np.sqrt(np.mean(theta_errors))
        rmse_r = np.sqrt(np.mean(r_errors))
        avg_time = np.mean(times) * 1000
        avg_crlb_theta = np.nanmean(crlb_theta_list)
        avg_crlb_r = np.nanmean(crlb_r_list)

        results['rmse_theta'].append(rmse_theta)
        results['rmse_r'].append(rmse_r)
        results['avg_time'].append(avg_time)
        results['avg_crlb_theta'].append(avg_crlb_theta)
        results['avg_crlb_r'].append(avg_crlb_r)

        print(f"SNR={snr_db:3d}dB | RMSE_θ={rmse_theta:6.2f}° | RMSE_r={rmse_r:7.2f}m | Time={avg_time:6.2f}ms")

    np.savez('esprit_fair_benchmark.npz', **results)

    data_to_save = np.column_stack((
        results['snr_list'],
        results['rmse_theta'],
        results['rmse_r'],
        results['avg_time'],
        results['avg_crlb_theta'],
        results['avg_crlb_r']
    ))
    np.savetxt('esprit_fair_benchmark.txt', data_to_save,
               header='SNR_dB  RMSE_theta_ESPRIT  RMSE_r_ESPRIT  Time_ms  CRLB_theta  CRLB_r',
               fmt='%.6f')

    out_dir = Path("algorithm_comparison") / "单快拍"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_dir / "esprit_fair_rmse_data.txt",
        data_to_save,
        header='SNR_dB  RMSE_theta_ESPRIT  RMSE_r_ESPRIT  Time_ms  CRLB_theta  CRLB_r',
        fmt='%.6f',
    )

    print("\n结果已保存到 esprit_fair_benchmark.npz 和 esprit_fair_benchmark.txt")
    return results


if __name__ == "__main__":
    run_benchmark()
