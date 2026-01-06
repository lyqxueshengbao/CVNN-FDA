# benchmark_msbl_fast.py
"""
公平对比的 MSBL (快速优化版 - 已修复矩阵维度 Bug)
优化点：
1. 避免计算完整协方差矩阵，只计算对角线元素。
2. 修复了之前的广播错误 (Broadcasting Error)。
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
Monte_Carlo = 50 

# ======================== 1. 字典构建 ========================
Grid_theta = np.arange(theta_min, theta_max + 1, 1) # 1度
Grid_r = np.arange(r_min, r_max + 1, 50)            # 50米

print(f"正在构建 SBL 字典...")
Dictionary = []
Atom_Params = [] 
for theta in Grid_theta:
    for r in Grid_r:
        atom = get_steering_vector(r, theta)
        atom = atom / norm(atom) # 归一化
        Dictionary.append(atom)
        Atom_Params.append((theta, r))

Phi = np.array(Dictionary).T  # [MN, N_atoms] -> [100, 4961]
N_atoms = Phi.shape[1]
print(f"字典构建完成: {Phi.shape}")

# ======================== 2. 快速 MSBL 求解器 ========================
def msbl_solver_fast(Y, max_iter=15, tol=1e-3):
    """
    高度优化的 SBL 求解器
    """
    MN_dim, L_snapshots = Y.shape
    
    # 初始化
    gamma = np.ones(N_atoms) 
    sigma2 = 1e-2 
    
    # 预计算 Phi 的共轭转置
    Phi_H = Phi.conj().T # [N_atoms, MN]
    
    for i in range(max_iter):
        gamma_old = gamma.copy()
        
        # 1. 计算 Sigma_y (MN x MN) [100 x 100]
        # 利用广播机制: Phi @ Gamma @ Phi_H
        # 优化: Phi * gamma 相当于给 Phi 的每一列乘上 gamma
        Phi_Gamma = Phi * gamma[None, :]
        Sigma_y = Phi_Gamma @ Phi_H
        # 添加噪声对角阵
        np.fill_diagonal(Sigma_y, np.diag(Sigma_y) + sigma2)
        
        # 2. 求逆 (100x100)
        try:
            Sigma_y_inv = inv(Sigma_y)
        except:
            break 
            
        # 3. 计算后验均值 mu
        # (100x1) = (100x100) * (100x1)
        Sigma_y_inv_Y = Sigma_y_inv @ Y
        # (4961x1) = (4961x100) * (100x1)
        # 这里的 gamma 是 (4961,), Phi_H 是 (4961, 100)
        mu = gamma[:, None] * (Phi_H @ Sigma_y_inv_Y)
        mu_sq = np.real(np.sum(np.abs(mu)**2, axis=1)) / L_snapshots
        
        # 4. 计算 Sigma_x 的对角线元素 (Bug 修复处)
        # S_inv_Phi [100, 4961] = [100, 100] @ [100, 4961]
        S_inv_Phi = Sigma_y_inv @ Phi
        
        # 我们需要 diag(Phi^H @ S_inv_Phi)
        # Phi^H 是 [4961, 100], S_inv_Phi 是 [100, 4961]
        # 对角线元素 k 是: Phi^H 的第 k 行 与 S_inv_Phi 的第 k 列的点积
        # Phi^H 的第 k 行 = (Phi 的第 k 列) 的共轭
        # 所以等价于: sum(Phi.conj() * S_inv_Phi, axis=0) -> [100, 4961] 沿 100 维度求和 -> [4961]
        
        term_diag = np.real(np.sum(Phi.conj() * S_inv_Phi, axis=0))
        
        # Sigma_x_diag = gamma - gamma^2 * term_diag
        Sigma_x_diag = gamma - (gamma**2) * term_diag
        
        # 5. 更新 Gamma
        gamma = mu_sq + Sigma_x_diag
        
        # 6. 收敛判定
        if np.max(np.abs(gamma - gamma_old)) < tol:
            break
            
    best_idx = np.argmax(gamma)
    return Atom_Params[best_idx]

# ======================== 3. 主程序 ========================
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

def run_benchmark():
    print("=" * 60)
    print("MSBL (快速优化版) 公平基准测试")
    print("=" * 60)
    
    results = {'snr_list': SNR_dB_list, 'rmse_theta': [], 'rmse_r': [], 'avg_time': [], 'avg_crlb_theta': [], 'avg_crlb_r': []}

    for snr_db in SNR_dB_list:
        theta_errors = []
        r_errors = []
        times = []
        crlb_theta_list = []
        crlb_r_list = []

        print(f"Running SNR = {snr_db} dB ...")
        for mc in range(Monte_Carlo):
            r_true = np.random.uniform(r_min, r_max)
            theta_true = np.random.uniform(theta_min, theta_max)
            
            c_theta, c_r = compute_crlb(r_true, theta_true, snr_db)
            crlb_theta_list.append(c_theta)
            crlb_r_list.append(c_r)

            Y = generate_signal_consistent(r_true, theta_true, snr_db, L)

            t_start = time.time()
            try:
                theta_est, r_est = msbl_solver_fast(Y)
            except Exception as e:
                print(f"Error: {e}")
                theta_est, r_est = 0, 0
                
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

        print(f"  RMSE_θ={rmse_theta:6.2f}° | RMSE_r={rmse_r:7.2f}m | Time={avg_time:6.2f}ms")

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
               
    print("\n结果已保存.")
    return results

if __name__ == "__main__":
    run_benchmark()
