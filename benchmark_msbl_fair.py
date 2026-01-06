# -*- coding:utf-8 -*-
"""
作者：李钰钦
日期：2026年01月06日
"""
# benchmark_msbl_fair.py
"""
公平对比的 MSBL (稀疏贝叶斯学习) 基准测试
地位：传统算法中的"精度天花板"，通常作为高精度的 Baseline。
特点：
1. 抗噪性极强：在低信噪比下通常优于 MUSIC/OMP。
2. 计算量巨大：涉及多次矩阵求逆，速度极慢。
3. 网格依赖：依然受限于预设网格 (此处保持 50m/1度 粗网格以示公平)。
"""
import numpy as np
from numpy.linalg import norm, inv, solve
import time
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
Monte_Carlo = 50  # !! 警告：SBL 极慢，建议先用 50 次跑通，确认无误后再加到 200 !!

# ======================== 1. 字典构建 (预计算) ========================
# 保持与 MUSIC/OMP 完全一致的网格
Grid_theta = np.arange(theta_min, theta_max + 1, 1)  # 1度
Grid_r = np.arange(r_min, r_max + 1, 50)  # 50米

print(f"正在构建 SBL 字典 (网格: {len(Grid_theta)}x{len(Grid_r)})...")
Dictionary = []
Atom_Params = []

for theta in Grid_theta:
    for r in Grid_r:
        atom = get_steering_vector(r, theta)
        atom = atom / norm(atom)  # 归一化是 SBL 收敛的关键
        Dictionary.append(atom)
        Atom_Params.append((theta, r))

# 转置为 [MN, N_atoms]
Phi = np.array(Dictionary).T
N_atoms = Phi.shape[1]
print(f"字典构建完成: {Phi.shape}")
print("提示: SBL 涉及大矩阵求逆，运行时间将显著长于 OMP。")


# ======================== 2. MSBL 求解器 (核心) ========================
def msbl_solver(Y, max_iter=20, tol=1e-4):
    """
    基础版 SBL/RVM 算法实现
    Y: 观测信号 [MN, L]
    """
    MN_dim, L_snapshots = Y.shape

    # 1. 初始化
    # gamma: 信号功率的超参数 (对应网格点上的信号方差)
    gamma = np.ones(N_atoms)
    # sigma2: 噪声方差 (也可以学习，这里为了稳定先初始化)
    sigma2 = 1e-3

    mu = np.zeros((N_atoms, L_snapshots), dtype=complex)

    for i in range(max_iter):
        gamma_old = gamma.copy()

        # 2. 计算先验协方差矩阵 Sigma_y
        # Sigma_y = Phi * Gamma * Phi^H + sigma2 * I
        # 为了加速，仅操作非零 gamma (剪枝策略可选，这里用完整矩阵)
        Gamma = np.diag(gamma)
        Sigma_y = Phi @ Gamma @ Phi.conj().T + sigma2 * np.eye(MN_dim)

        # 3. 求逆 (最耗时步骤)与后验计算
        # 使用 solve 代替 inv 稍微快一点: Sigma_y_inv_Y = inv(Sigma_y) @ Y
        try:
            Sigma_y_inv_Y = solve(Sigma_y, Y)
            # 后验均值 mu (估计的信号系数)
            mu = Gamma @ Phi.conj().T @ Sigma_y_inv_Y

            # 后验协方差的对角线元素 (用于更新 gamma)
            # Sigma_x = Gamma - Gamma * Phi^H * inv(Sigma_y) * Phi * Gamma
            # 我们只需要对角线
            Sigma_y_inv = inv(Sigma_y)  # 这里必须求逆了
            Sigma_x = Gamma - Gamma @ Phi.conj().T @ Sigma_y_inv @ Phi @ Gamma
            Sigma_x_diag = np.real(np.diag(Sigma_x))

        except np.linalg.LinAlgError:
            # 矩阵奇异，停止迭代
            break

        # 4. 更新超参数 (EM 算法更新规则)
        # gamma_new = mu^2 + Sigma_x_diag
        mu_sq = np.sum(np.abs(mu) ** 2, axis=1) / L_snapshots
        gamma = mu_sq + Sigma_x_diag

        # 更新噪声方差 sigma2 (可选，这里简化为固定或简单更新)
        # residue = norm(Y - Phi @ mu)**2
        # sigma2 = (residue + sigma2 * np.sum(1 - gamma/gamma_old)) / MN_dim
        # 为保持基准测试稳定性，这里通常不频繁更新 sigma2 或令其收敛较慢

        # 5. 收敛检查
        if np.max(np.abs(gamma - gamma_old)) < tol:
            break

    # 寻找峰值
    best_idx = np.argmax(gamma)
    return Atom_Params[best_idx]


# ======================== 3. 辅助函数 ========================
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


def generate_signal_consistent(r, theta_deg, snr_db, L):
    u = get_steering_vector(r, theta_deg).reshape(-1, 1)
    s = (np.random.randn(1, L) + 1j * np.random.randn(1, L)) / np.sqrt(2)
    X_clean = u @ s
    noise = (np.random.randn(MN, L) + 1j * np.random.randn(MN, L)) / np.sqrt(2)
    power_sig = np.mean(np.abs(X_clean) ** 2)
    power_noise = power_sig / (10 ** (snr_db / 10.0))
    return X_clean + np.sqrt(power_noise) * noise


# ======================== 4. 主程序 ========================
def run_benchmark():
    print("=" * 60)
    print("MSBL (稀疏贝叶斯) 公平基准测试")
    print("注意: 这是一个高计算量的算法，请耐心等待...")
    print("=" * 60)

    results = {'snr_list': SNR_dB_list, 'rmse_theta': [], 'rmse_r': [], 'avg_time': [], 'avg_crlb_theta': [],
               'avg_crlb_r': []}

    for snr_db in SNR_dB_list:
        theta_errors = []
        r_errors = []
        times = []
        crlb_theta_list = []
        crlb_r_list = []

        # 进度条提示
        print(f"Running SNR = {snr_db} dB ...")

        for mc in range(Monte_Carlo):
            r_true = np.random.uniform(r_min, r_max)
            theta_true = np.random.uniform(theta_min, theta_max)

            c_theta, c_r = compute_crlb(r_true, theta_true, snr_db)
            crlb_theta_list.append(c_theta)
            crlb_r_list.append(c_r)

            Y = generate_signal_consistent(r_true, theta_true, snr_db, L)

            t_start = time.time()
            # 核心求解
            theta_est, r_est = msbl_solver(Y, max_iter=20)
            t_end = time.time()

            theta_errors.append((theta_est - theta_true) ** 2)
            r_errors.append((r_est - r_true) ** 2)
            times.append(t_end - t_start)

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

        print(f"  Result: RMSE_θ={rmse_theta:6.2f}° | RMSE_r={rmse_r:7.2f}m | Time={avg_time:6.2f}ms")

    # 保存
    np.savez('msbl_fair_benchmark.npz', **results)

    data_to_save = np.column_stack((
        results['snr_list'],
        results['rmse_theta'],
        results['rmse_r'],
        results['avg_time'],
        results['avg_crlb_theta'],
        results['avg_crlb_r']
    ))
    np.savetxt('msbl_fair_benchmark.txt', data_to_save,
               header='SNR_dB  RMSE_theta_MSBL  RMSE_r_MSBL  Time_ms  CRLB_theta  CRLB_r',
               fmt='%.6f')

    print("\n结果已保存到 msbl_fair_benchmark.npz 和 msbl_fair_benchmark.txt")
    return results


if __name__ == "__main__":
    run_benchmark()
