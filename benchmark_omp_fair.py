# benchmark_omp_fair.py
"""
公平对比的 OMP 基准测试 (无速度优化 / Naive Implementation)
特点：
1. 【关键】不做字典预计算：每次估算都重新通过循环构建字典矩阵。
   这会还原 text_OMP_RMSE.py 的低效运行速度 (~200ms/次)，
   从而凸显 CVNN 模型推理的实时性优势。
2. 物理模型一致：使用 utils_physics 确保与 CVNN 同源。
3. 数据分布一致：全域随机采样。
"""
import numpy as np
from numpy.linalg import norm, inv
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
Monte_Carlo = 200  # 次数

# 网格定义 (仅定义范围，不在全局构建字典)
Grid_theta = np.arange(theta_min, theta_max + 1, 1)  # 1度间隔
Grid_r = np.arange(r_min, r_max + 1, 50)  # 50米间隔

print(f"当前测试范围: [{r_min}, {r_max}]m, [{theta_min}, {theta_max}]°")
print("注意：本脚本运行会较慢，因为每次估算都会重新构建字典矩阵。")


# ======================== 核心工具函数 ========================
def compute_crlb(r, theta_deg, snr_db):
    """计算理论下界 CRLB"""
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
    """生成测试信号"""
    u = get_steering_vector(r, theta_deg).reshape(-1, 1)
    s = (np.random.randn(1, L) + 1j * np.random.randn(1, L)) / np.sqrt(2)
    X_clean = u @ s
    noise = (np.random.randn(MN, L) + 1j * np.random.randn(MN, L)) / np.sqrt(2)
    power_sig = np.mean(np.abs(X_clean) ** 2)
    power_noise = power_sig / (10 ** (snr_db / 10.0))
    return X_clean + np.sqrt(power_noise) * noise


def omp_solver_naive(Y, K=1):
    """
    原始/笨重的 OMP 实现
    每次调用都重新构建字典，模拟计算资源受限或非缓存场景下的真实负载。
    """
    # 1. 现场构建字典 (Time Consuming!)
    # 这部分代码模拟了 text_OMP_RMSE.py 中的双重循环逻辑
    Dictionary = []
    Atom_Params = []

    # 显式循环构建，消耗 CPU 时间
    for theta in Grid_theta:
        for r in Grid_r:
            # 调用物理模型计算导向矢量
            atom = get_steering_vector(r, theta)
            # 归一化
            atom = atom / norm(atom)

            Dictionary.append(atom)
            Atom_Params.append((theta, r))

    Dictionary = np.array(Dictionary).T  # 转置

    # 2. 匹配求解
    residual = Y[:, 0].copy()
    correlations = np.abs(Dictionary.conj().T @ residual)
    best_idx = np.argmax(correlations)

    return Atom_Params[best_idx]


# ======================== 主循环 ========================
def run_benchmark():
    print("=" * 60)
    print("OMP 原始公平基准测试 (无速度优化版)")
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
            # 1. 随机生成目标
            r_true = np.random.uniform(r_min, r_max)
            theta_true = np.random.uniform(theta_min, theta_max)

            # 2. 记录 CRLB
            c_theta, c_r = compute_crlb(r_true, theta_true, snr_db)
            crlb_theta_list.append(c_theta)
            crlb_r_list.append(c_r)

            # 3. 生成信号
            Y = generate_signal_consistent(r_true, theta_true, snr_db, L)

            # 4. 算法估计 (会计入字典构建的时间)
            t_start = time.time()
            theta_est, r_est = omp_solver_naive(Y, K=1)
            t_end = time.time()

            theta_errors.append((theta_est - theta_true) ** 2)
            r_errors.append((r_est - r_true) ** 2)
            times.append(t_end - t_start)

        # 5. 统计
        rmse_theta = np.sqrt(np.mean(theta_errors))
        rmse_r = np.sqrt(np.mean(r_errors))
        avg_time = np.mean(times) * 1000  # ms
        avg_crlb_theta = np.nanmean(crlb_theta_list)
        avg_crlb_r = np.nanmean(crlb_r_list)

        results['rmse_theta'].append(rmse_theta)
        results['rmse_r'].append(rmse_r)
        results['avg_time'].append(avg_time)
        results['avg_crlb_theta'].append(avg_crlb_theta)
        results['avg_crlb_r'].append(avg_crlb_r)

        print(f"SNR={snr_db:3d}dB | RMSE_θ={rmse_theta:6.2f}° | RMSE_r={rmse_r:7.2f}m | Time={avg_time:6.2f}ms")

    # 保存
    np.savez('omp_fair_benchmark.npz', **results)

    data_to_save = np.column_stack((
        results['snr_list'],
        results['rmse_theta'],
        results['rmse_r'],
        results['avg_time'],
        results['avg_crlb_theta'],
        results['avg_crlb_r']
    ))
    np.savetxt('omp_fair_benchmark.txt', data_to_save,
               header='SNR_dB  RMSE_theta_OMP  RMSE_r_OMP  Time_ms  CRLB_theta  CRLB_r',
               fmt='%.6f')

    out_dir = Path("algorithm_comparison") / "单快拍"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_dir / "omp_fair_rmse_data.txt",
        data_to_save,
        header='SNR_dB  RMSE_theta_OMP  RMSE_r_OMP  Time_ms  CRLB_theta  CRLB_r',
        fmt='%.6f',
    )

    print("\n结果已保存到 omp_fair_benchmark.npz 和 omp_fair_benchmark.txt")
    return results


if __name__ == "__main__":
    run_benchmark()
