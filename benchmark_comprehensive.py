"""
FDA-MIMO 综合评测脚本
对比传统算法 (MUSIC, ESPRIT, OMP) 与深度学习方法 (CVNN)
"""
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy.linalg import eig, pinv, inv

import config as cfg
from model import FDA_CVNN, FDA_CVNN_Light, FDA_CVNN_Attention
from dataset import FDADataset
from utils_physics import denormalize_labels

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


# ==================== CRLB 计算 ====================
def crlb_fda_mimo(theta_true, r_true, M, N, f0, Delta_f, c0, d, lambda_, L, SNR_dB):
    """
    计算FDA-MIMO的克拉美罗下界 (CRLB)

    参数:
        theta_true: 真实角度 (弧度)
        r_true: 真实距离 (米)
        M, N: 发射/接收阵元数
        f0: 载频 (Hz)
        Delta_f: 频率偏移 (Hz)
        c0: 光速 (m/s)
        d: 阵元间距 (m)
        lambda_: 波长 (m)
        L: 快拍数
        SNR_dB: 信噪比 (dB)

    返回:
        CRLB_theta: 角度CRLB (度)
        CRLB_r: 距离CRLB (米)
    """
    if np.isscalar(theta_true):
        theta_true = np.array([theta_true])
        r_true = np.array([r_true])

    K = len(theta_true)
    sigma2 = 10 ** (-SNR_dB / 10)
    xi_power = 1 / sigma2

    F = np.zeros((2 * K, 2 * K))

    for i in range(K):
        # 导向矢量
        tau = 2 * r_true[i] / c0
        a_t = np.exp(-1j * 2 * np.pi * np.arange(M) * Delta_f * tau)
        a_r = np.exp(1j * 2 * np.pi * d / lambda_ * np.sin(theta_true[i]) * np.arange(N))
        u = np.kron(a_t, a_r)

        # 导向矢量对角度的导数
        d_phi_dtheta = 2 * np.pi * d / lambda_ * np.cos(theta_true[i])
        da_r_dtheta = 1j * d_phi_dtheta * np.arange(N) * a_r
        du_dtheta = np.kron(a_t, da_r_dtheta)

        # 导向矢量对距离的导数
        d_tau_dr = 2 / c0
        da_t_dr = -1j * 2 * np.pi * Delta_f * d_tau_dr * np.arange(M) * a_t
        du_dr = np.kron(da_t_dr, a_r)

        # FIM 元素
        F[2 * i, 2 * i] = 2 * L * xi_power * np.real(np.dot(du_dtheta.conj(), du_dtheta))
        F[2 * i + 1, 2 * i + 1] = 2 * L * xi_power * np.real(np.dot(du_dr.conj(), du_dr))
        F[2 * i, 2 * i + 1] = 2 * L * xi_power * np.real(np.dot(du_dtheta.conj(), du_dr))
        F[2 * i + 1, 2 * i] = F[2 * i, 2 * i + 1]

    invF = inv(F)
    CRLB_theta = np.sqrt(np.diag(invF[0::2, 0::2])) * 180 / np.pi
    CRLB_r = np.sqrt(np.diag(invF[1::2, 1::2]))

    return CRLB_theta, CRLB_r


# ==================== 2D-MUSIC 算法 ====================
def music_algorithm(Y, M, N, Delta_f, c0, d, lambda_, Grid_theta, Grid_r, K):
    """
    2D-MUSIC 算法实现

    参数:
        Y: 接收数据 [MN, L]
        M, N: 发射/接收阵元数
        Delta_f: 频率偏移
        c0: 光速
        d: 阵元间距
        lambda_: 波长
        Grid_theta: 角度网格 (度)
        Grid_r: 距离网格 (米)
        K: 目标数

    返回:
        theta_est: 角度估计 (弧度)
        r_est: 距离估计 (米)
    """
    # 协方差矩阵
    R = Y @ Y.conj().T / Y.shape[1]

    # 特征分解获取噪声子空间
    D, V = eig(R)
    idx = np.argsort(np.real(D))
    Un = V[:, idx[:V.shape[1] - K]]

    # 构建二维空间谱
    P = np.zeros((len(Grid_theta), len(Grid_r)))
    for i in range(len(Grid_theta)):
        theta = Grid_theta[i] * np.pi / 180
        for j in range(len(Grid_r)):
            r = Grid_r[j]
            tau = 2 * r / c0
            a_t = np.exp(-1j * 2 * np.pi * np.arange(M) * Delta_f * tau)
            a_r = np.exp(1j * 2 * np.pi * d / lambda_ * np.sin(theta) * np.arange(N))
            a = np.kron(a_t, a_r)
            P[i, j] = 1 / np.real(a.conj().T @ Un @ Un.conj().T @ a)

    # 搜索谱峰
    peak_idx = np.argpartition(P.flatten(), -K)[-K:]
    theta_idx, r_idx = np.unravel_index(peak_idx, P.shape)
    theta_est = Grid_theta[theta_idx] * np.pi / 180
    r_est = Grid_r[r_idx]

    return theta_est, r_est


# ==================== 2D-ESPRIT 算法 ====================
def esprit_algorithm(Y, M, N, Delta_f, c0, d, lambda_, K):
    """
    2D-ESPRIT 算法实现

    参数:
        Y: 接收数据 [MN, L]
        M, N: 发射/接收阵元数
        Delta_f: 频率偏移
        c0: 光速
        d: 阵元间距
        lambda_: 波长
        K: 目标数

    返回:
        theta_est: 角度估计 (弧度)
        r_est: 距离估计 (米)
    """
    # 协方差矩阵与信号子空间
    R = Y @ Y.conj().T / Y.shape[1]
    D, V = eig(R)
    idx = np.argsort(np.real(D))[::-1]
    Es = V[:, idx[:K]]

    # 选择矩阵
    JR1 = np.kron(np.eye(M), np.hstack([np.eye(N - 1), np.zeros((N - 1, 1))]))
    JR2 = np.kron(np.eye(M), np.hstack([np.zeros((N - 1, 1)), np.eye(N - 1)]))
    JT1 = np.kron(np.hstack([np.eye(M - 1), np.zeros((M - 1, 1))]), np.eye(N))
    JT2 = np.kron(np.hstack([np.zeros((M - 1, 1)), np.eye(M - 1)]), np.eye(N))

    # 角度估计
    T_theta = JR2 @ Es
    S_theta = JR1 @ Es
    Psi_theta = pinv(S_theta) @ T_theta
    D_theta, V_theta = eig(Psi_theta)
    phi_theta = D_theta
    theta_est = np.arcsin(np.angle(phi_theta) * lambda_ / (2 * np.pi * d))

    # 距离估计
    T_r = JT2 @ Es
    S_r = JT1 @ Es
    Psi_r = pinv(S_r) @ T_r
    D_r, V_r = eig(Psi_r)
    phi_r = D_r
    r_est = -(np.angle(phi_r) * c0) / (4 * np.pi * Delta_f)

    # 参数配对
    match_idx = np.argmax(np.abs(V_theta.conj().T @ V_r), axis=1)
    r_est = r_est[match_idx]

    return theta_est, r_est


# ==================== OMP 算法 ====================
def omp_algorithm(Y, M, N, Delta_f, c0, d, lambda_, Grid_theta, Grid_r, K):
    """
    OMP (正交匹配追踪) 算法实现

    参数:
        Y: 接收数据 [MN, L]
        M, N: 发射/接收阵元数
        Delta_f: 频率偏移
        c0: 光速
        d: 阵元间距
        lambda_: 波长
        Grid_theta: 角度网格 (度)
        Grid_r: 距离网格 (米)
        K: 目标数

    返回:
        theta_est: 角度估计 (弧度)
        r_est: 距离估计 (米)
    """
    X = Y[:, 0]  # 单快拍

    # 构建过完备字典
    A = []
    for i in range(len(Grid_theta)):
        theta = Grid_theta[i] * np.pi / 180
        for j in range(len(Grid_r)):
            r = Grid_r[j]
            tau = 2 * r / c0
            a_t = np.exp(-1j * 2 * np.pi * np.arange(M) * Delta_f * tau)
            a_r = np.exp(1j * 2 * np.pi * d / lambda_ * np.sin(theta) * np.arange(N))
            a = np.kron(a_t, a_r)
            A.append(a)
    A = np.array(A).T

    # 列归一化
    A = A / np.sqrt(np.sum(np.abs(A) ** 2, axis=0))

    # OMP 迭代
    r = X.copy()
    Lambda = []
    for iter in range(K):
        corr = np.abs(np.dot(A.conj().T, r))
        idx = np.argmax(corr)
        Lambda.append(idx)

        A_sub = A[:, Lambda]
        s_hat = pinv(A_sub) @ X
        X_hat = A_sub @ s_hat
        r = X - X_hat

    # 提取参数
    r_idx = np.array(Lambda) % len(Grid_r)
    theta_idx = np.array(Lambda) // len(Grid_r)
    theta_est = Grid_theta[theta_idx] * np.pi / 180
    r_est = Grid_r[r_idx]

    return theta_est, r_est


# ==================== 传统算法评测 ====================
def evaluate_classical_algorithm(algorithm_name, algorithm_func, params,
                                  SNR_dB_list, Monte_Carlo=100):
    """
    评测传统算法在多个SNR下的性能

    参数:
        algorithm_name: 算法名称
        algorithm_func: 算法函数
        params: 算法参数字典
        SNR_dB_list: SNR列表
        Monte_Carlo: 蒙特卡洛次数

    返回:
        results: 评测结果字典
    """
    M = params['M']
    N = params['N']
    Delta_f = params['Delta_f']
    c0 = params['c0']
    d = params['d']
    lambda_ = params['lambda_']
    K = params['K']
    L = params['L']
    theta_true = params['theta_true']
    r_true = params['r_true']
    Grid_theta = params.get('Grid_theta', None)
    Grid_r = params.get('Grid_r', None)

    rmse_theta_list = []
    rmse_r_list = []
    time_list = []

    print(f"\n评测 {algorithm_name} 算法...")

    for snr_idx, SNR in enumerate(tqdm(SNR_dB_list, desc=algorithm_name)):
        sigma2 = 10 ** (-SNR / 10)

        theta_err_total = 0
        r_err_total = 0
        time_total = 0

        for mc in range(Monte_Carlo):
            # 生成接收数据
            Y = np.zeros((M * N, L), dtype=complex)
            for k in range(K):
                theta = theta_true if K == 1 else theta_true[k]
                r = r_true if K == 1 else r_true[k]
                tau = 2 * r / c0
                a_t = np.exp(-1j * 2 * np.pi * np.arange(M) * Delta_f * tau)
                a_r = np.exp(1j * 2 * np.pi * d / lambda_ * np.sin(theta) * np.arange(N))
                a = np.kron(a_t, a_r)
                s = np.sqrt(1 / sigma2) * (np.random.randn(1, L) + 1j * np.random.randn(1, L))
                Y = Y + np.outer(a, s)
            Y = Y + np.sqrt(sigma2 / 2) * (np.random.randn(M * N, L) + 1j * np.random.randn(M * N, L))

            # 运行算法
            start_time = time.time()
            if algorithm_name in ['2D-MUSIC', 'OMP']:
                theta_est, r_est = algorithm_func(Y, M, N, Delta_f, c0, d, lambda_,
                                                  Grid_theta, Grid_r, K)
            else:  # ESPRIT
                theta_est, r_est = algorithm_func(Y, M, N, Delta_f, c0, d, lambda_, K)
            time_total += time.time() - start_time

            # 计算误差
            for k in range(K):
                theta_k = theta_true if K == 1 else theta_true[k]
                match_theta = np.argmin(np.abs(theta_est - theta_k))
                theta_err = (theta_est[match_theta] - theta_k) ** 2
                theta_err_total += theta_err

                r_k = r_true if K == 1 else r_true[k]
                match_r = np.argmin(np.abs(r_est - r_k))
                r_err = (r_est[match_r] - r_k) ** 2
                r_err_total += r_err

        # 计算 RMSE
        rmse_theta = np.sqrt(theta_err_total / (Monte_Carlo * K)) * 180 / np.pi
        rmse_r = np.sqrt(r_err_total / (Monte_Carlo * K))
        avg_time = time_total / Monte_Carlo

        rmse_theta_list.append(rmse_theta)
        rmse_r_list.append(rmse_r)
        time_list.append(avg_time)

    results = {
        'algorithm': algorithm_name,
        'snr_db_list': SNR_dB_list.tolist(),
        'rmse_theta': rmse_theta_list,
        'rmse_r': rmse_r_list,
        'avg_time': time_list
    }

    return results


# ==================== CVNN 评测 ====================
def evaluate_cvnn(model_path, SNR_dB_list, num_samples=1000, batch_size=64,
                  device='cuda', attention_type='dual', reduction=8, auto_detect=False):
    """
    评测 CVNN 模型在多个SNR下的性能

    参数:
        model_path: 模型权重路径
        SNR_dB_list: SNR列表
        num_samples: 每个SNR的测试样本数
        batch_size: 批次大小
        device: 设备
        attention_type: 注意力类型 ('dual', 'se', 'far', 'standard')
        reduction: 注意力模块的压缩比 (4, 8, 16等)
        auto_detect: 是否自动检测模型类型（默认False，使用手动指定）

    返回:
        results: 评测结果字典
    """
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        if auto_detect:
            # 自动检测模型类型（可能不准确）
            keys = list(state_dict.keys())
            has_dual = any('global_attn' in k or 'local_attn' in k for k in keys)
            has_far = any('attn' in k and 'conv_rr' in k for k in keys) and not has_dual
            has_se = any('attn' in k and '.fc.' in k for k in keys) and not has_dual

            if has_dual:
                attention_type = 'dual'
            elif has_far:
                attention_type = 'far'
            elif has_se:
                attention_type = 'se'
            else:
                attention_type = 'standard'

            # 尝试从权重形状推断reduction
            for key in keys:
                if 'attn1.fc.0.weight' in key or 'attn1.global_attn.fc.0.weight' in key:
                    weight_shape = state_dict[key].shape
                    reduction = 32 // weight_shape[0]  # 32是通道数
                    break

        # 根据指定的参数实例化模型
        if attention_type == 'dual':
            model = FDA_CVNN_Attention(attention_type='dual', se_reduction=reduction).to(device)
            print(f"✓ 使用模型: FDA_CVNN_Attention(dual, reduction={reduction})")
        elif attention_type == 'far':
            model = FDA_CVNN_Attention(attention_type='far', se_reduction=reduction).to(device)
            print(f"✓ 使用模型: FDA_CVNN_Attention(far, reduction={reduction})")
        elif attention_type == 'se':
            model = FDA_CVNN_Attention(attention_type='se', se_reduction=reduction).to(device)
            print(f"✓ 使用模型: FDA_CVNN_Attention(se, reduction={reduction})")
        else:
            model = FDA_CVNN().to(device)
            print(f"✓ 使用模型: FDA_CVNN(standard)")

        # 移除 module. 前缀（多GPU训练的产物）
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✓ 已加载模型: {model_path}")
    else:
        print(f"⚠ 警告: 模型文件不存在 {model_path}，使用随机初始化权重")
        model = FDA_CVNN().to(device)

    model.eval()

    rmse_theta_list = []
    rmse_r_list = []
    time_list = []

    print(f"\n评测 CVNN 模型...")

    for SNR in tqdm(SNR_dB_list, desc="CVNN"):
        # 创建测试数据集
        seed = abs(int(cfg.seed + SNR * 100)) % (2**31)
        dataset = FDADataset(num_samples, snr_db=SNR, online=False, seed=seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_labels = []
        time_total = 0

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                start_time = time.time()
                preds = model(batch_x)
                time_total += time.time() - start_time
                all_preds.append(preds.cpu())
                all_labels.append(batch_y)

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # 反归一化
        preds_physical = denormalize_labels(all_preds)
        labels_physical = denormalize_labels(all_labels)

        # 计算误差
        r_error = preds_physical[:, 0] - labels_physical[:, 0]
        theta_error = preds_physical[:, 1] - labels_physical[:, 1]

        rmse_r = float(np.sqrt(np.mean(r_error ** 2)))
        rmse_theta = float(np.sqrt(np.mean(theta_error ** 2)))
        avg_time = time_total / num_samples

        rmse_theta_list.append(rmse_theta)
        rmse_r_list.append(rmse_r)
        time_list.append(avg_time)

    results = {
        'algorithm': 'CVNN',
        'snr_db_list': SNR_dB_list.tolist(),
        'rmse_theta': rmse_theta_list,
        'rmse_r': rmse_r_list,
        'avg_time': time_list
    }

    return results


# ==================== 绘图函数 ====================
def plot_comparison(all_results, crlb_results, save_path='results'):
    """
    绘制所有算法的对比图

    参数:
        all_results: 所有算法的结果列表
        crlb_results: CRLB结果
        save_path: 保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    # 颜色和标记
    colors = ['b', 'g', 'r', 'm', 'c']
    markers = ['o', 's', '^', 'd', 'v']

    # 角度 RMSE
    plt.figure(figsize=(10, 6))
    for idx, result in enumerate(all_results):
        snr_list = result['snr_db_list']
        rmse_theta = result['rmse_theta']
        label = result['algorithm']
        plt.semilogy(snr_list, rmse_theta,
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    linewidth=2, markersize=8, label=label)

    # 绘制 CRLB
    plt.semilogy(crlb_results['snr_db_list'], crlb_results['crlb_theta'],
                'k--', linewidth=2, label='CRLB')

    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Angle RMSE (deg)', fontsize=14)
    plt.title('Angle Estimation Performance Comparison', fontsize=16)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_path}/angle_comparison.png', dpi=300)
    plt.savefig(f'{save_path}/angle_comparison.pdf')
    print(f"已保存角度对比图: {save_path}/angle_comparison.png")

    # 距离 RMSE
    plt.figure(figsize=(10, 6))
    for idx, result in enumerate(all_results):
        snr_list = result['snr_db_list']
        rmse_r = result['rmse_r']
        label = result['algorithm']
        plt.semilogy(snr_list, rmse_r,
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    linewidth=2, markersize=8, label=label)

    # 绘制 CRLB
    plt.semilogy(crlb_results['snr_db_list'], crlb_results['crlb_r'],
                'k--', linewidth=2, label='CRLB')

    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Range RMSE (m)', fontsize=14)
    plt.title('Range Estimation Performance Comparison', fontsize=16)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_path}/range_comparison.png', dpi=300)
    plt.savefig(f'{save_path}/range_comparison.pdf')
    print(f"已保存距离对比图: {save_path}/range_comparison.png")

    # 计算时间对比（柱状图）
    plt.figure(figsize=(10, 6))
    algorithms = [r['algorithm'] for r in all_results]
    avg_times = [np.mean(r['avg_time']) * 1000 for r in all_results]  # 转为毫秒

    bars = plt.bar(algorithms, avg_times, color=colors[:len(algorithms)], alpha=0.7, edgecolor='black')
    plt.ylabel('Average Time (ms)', fontsize=14)
    plt.title('Computational Time Comparison (Average)', fontsize=16)
    plt.yscale('log')
    plt.grid(True, axis='y', alpha=0.3)

    # 添加数值标签
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{save_path}/time_comparison_bar.png', dpi=300)
    plt.savefig(f'{save_path}/time_comparison_bar.pdf')
    print(f"已保存时间对比柱状图: {save_path}/time_comparison_bar.png")

    # 计算时间随SNR变化（曲线图）
    plt.figure(figsize=(10, 6))
    for idx, result in enumerate(all_results):
        snr_list = result['snr_db_list']
        time_list = [t * 1000 for t in result['avg_time']]  # 转为毫秒
        label = result['algorithm']
        plt.semilogy(snr_list, time_list,
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    linewidth=2, markersize=8, label=label)

    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Computational Time (ms)', fontsize=14)
    plt.title('Computational Time vs SNR', fontsize=16)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_path}/time_vs_snr.png', dpi=300)
    plt.savefig(f'{save_path}/time_vs_snr.pdf')
    print(f"已保存时间随SNR变化图: {save_path}/time_vs_snr.png")

    plt.show()


# ==================== 主函数 ====================
def main():
    # ========== CVNN 模型配置（根据实际训练参数修改） ==========
    CVNN_ATTENTION_TYPE = 'dual'  # 模型注意力类型: 'dual', 'se', 'far', 'standard'
    CVNN_REDUCTION = 8            # 注意力模块压缩比: 4, 8, 16等
    CVNN_AUTO_DETECT = False      # 是否自动检测模型类型（推荐False，手动指定更准确）

    # ========== 物理参数 (与 config.py 保持一致) ==========
    M = cfg.M
    N = cfg.N
    f0 = cfg.f0
    Delta_f = cfg.delta_f
    c0 = cfg.c
    lambda_ = cfg.wavelength
    d = cfg.d
    K = 1
    theta_true = 10.0 * np.pi / 180  # 10度
    r_true = 2000.0  # 2000米
    L = 1  # 快拍数（单快拍场景）

    # SNR 范围
    SNR_dB_list = np.arange(-15, 25, 5)

    # 蒙特卡洛参数
    Monte_Carlo_classical = 100  # 传统算法的蒙特卡洛次数
    num_samples_cvnn = 1000      # CVNN 的测试样本数

    # 网格参数 (用于 MUSIC 和 OMP)
    Grid_theta = np.arange(-50, 51, 1)  # -50° 到 50°，步长 1°
    Grid_r = np.arange(0, 5001, 100)    # 0m 到 5000m，步长 100m

    # CVNN 模型路径
    model_path = cfg.model_save_path

    # 结果保存路径
    save_path = 'results/comprehensive_benchmark'
    os.makedirs(save_path, exist_ok=True)

    # ========== 计算 CRLB ==========
    print("\n" + "=" * 60)
    print("计算 CRLB...")
    print("=" * 60)

    crlb_theta_list = []
    crlb_r_list = []

    for SNR in SNR_dB_list:
        crlb_theta, crlb_r = crlb_fda_mimo(
            theta_true, r_true, M, N, f0, Delta_f, c0, d, lambda_, L, SNR
        )
        crlb_theta_list.append(crlb_theta[0])
        crlb_r_list.append(crlb_r[0])

    crlb_results = {
        'snr_db_list': SNR_dB_list.tolist(),
        'crlb_theta': crlb_theta_list,
        'crlb_r': crlb_r_list
    }

    # ========== 评测 CVNN（优先评测，快速验证模型加载） ==========
    print("\n" + "=" * 60)
    print("评测 CVNN 模型...")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 手动指定模型配置（服务器上运行时确保正确）
    cvnn_results = evaluate_cvnn(
        model_path, SNR_dB_list, num_samples_cvnn,
        batch_size=64, device=device,
        attention_type=CVNN_ATTENTION_TYPE,
        reduction=CVNN_REDUCTION,
        auto_detect=CVNN_AUTO_DETECT
    )

    # ========== 评测传统算法 ==========
    print("\n" + "=" * 60)
    print("评测传统算法...")
    print("=" * 60)

    params = {
        'M': M, 'N': N, 'Delta_f': Delta_f, 'c0': c0, 'd': d, 'lambda_': lambda_,
        'K': K, 'L': L, 'theta_true': theta_true, 'r_true': r_true,
        'Grid_theta': Grid_theta, 'Grid_r': Grid_r
    }

    classical_results = []

    # 2D-MUSIC
    music_results = evaluate_classical_algorithm(
        '2D-MUSIC', music_algorithm, params, SNR_dB_list, Monte_Carlo_classical
    )
    classical_results.append(music_results)

    # 2D-ESPRIT
    esprit_results = evaluate_classical_algorithm(
        '2D-ESPRIT', esprit_algorithm, params, SNR_dB_list, Monte_Carlo_classical
    )
    classical_results.append(esprit_results)

    # OMP
    omp_results = evaluate_classical_algorithm(
        'OMP', omp_algorithm, params, SNR_dB_list, Monte_Carlo_classical
    )
    classical_results.append(omp_results)

    # ========== 合并结果 ==========
    all_results = [cvnn_results] + classical_results

    # ========== 保存结果 ==========
    print("\n" + "=" * 60)
    print("保存结果...")
    print("=" * 60)

    results_dict = {
        'crlb': crlb_results,
        'algorithms': all_results,
        'parameters': {
            'M': M, 'N': N, 'f0': f0, 'Delta_f': Delta_f,
            'theta_true_deg': theta_true * 180 / np.pi,
            'r_true_m': r_true,
            'L': L,
            'Monte_Carlo_classical': Monte_Carlo_classical,
            'num_samples_cvnn': num_samples_cvnn
        }
    }

    with open(f'{save_path}/benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"已保存结果: {save_path}/benchmark_results.json")

    # ========== 绘制对比图 ==========
    print("\n" + "=" * 60)
    print("绘制对比图...")
    print("=" * 60)

    plot_comparison(all_results, crlb_results, save_path)

    # ========== 打印性能总结 ==========
    print("\n" + "=" * 60)
    print("性能总结 (SNR = 20dB)")
    print("=" * 60)
    print(f"{'Algorithm':<15} {'Angle RMSE (deg)':<20} {'Range RMSE (m)':<20} {'Time (ms)':<15}")
    print("-" * 70)

    snr_20_idx = np.where(SNR_dB_list == 20)[0]
    if len(snr_20_idx) > 0:
        idx = snr_20_idx[0]
        for result in all_results:
            algo = result['algorithm']
            angle_rmse = result['rmse_theta'][idx]
            range_rmse = result['rmse_r'][idx]
            avg_time = result['avg_time'][idx] * 1000
            print(f"{algo:<15} {angle_rmse:<20.4f} {range_rmse:<20.4f} {avg_time:<15.4f}")

        # CRLB
        crlb_angle = crlb_results['crlb_theta'][idx]
        crlb_range = crlb_results['crlb_r'][idx]
        print("-" * 70)
        print(f"{'CRLB':<15} {crlb_angle:<20.4f} {crlb_range:<20.4f} {'-':<15}")

    print("\n" + "=" * 60)
    print("评测完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
