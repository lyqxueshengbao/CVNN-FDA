# -*- coding: utf-8 -*-
"""
FDA-MIMO 2D-MUSIC 算法基准实现
用于评估 CVNN 应该达到的性能水平
"""

import numpy as np
import time
from config import r_min, r_max, theta_min, theta_max, M, N, c, delta_f, wavelength, d
from utils import generate_joint_steering_vector, generate_echo_signal, compute_sample_covariance_matrix


def music_2d_fda(R, num_signals=1, r_grid=None, theta_grid=None):
    """
    2D-MUSIC 算法用于 FDA-MIMO
    
    Args:
        R: 协方差矩阵 (MN, MN)
        num_signals: 信号源数量（单目标=1）
        r_grid: 距离搜索网格
        theta_grid: 角度搜索网格
    
    Returns:
        r_est, theta_est: 估计的距离和角度
        spectrum: MUSIC 谱
    """
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    
    # 降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 噪声子空间 (取后 MN - num_signals 个特征向量)
    noise_subspace = eigenvectors[:, num_signals:]
    
    # 默认搜索网格
    if r_grid is None:
        r_grid = np.linspace(r_min, r_max, 400)  # 更密集：步长5m
    if theta_grid is None:
        theta_grid = np.linspace(theta_min, theta_max, 240)  # 更密集：步长0.5°
    
    # 构造 MUSIC 谱
    spectrum = np.zeros((len(r_grid), len(theta_grid)))
    
    for i, r in enumerate(r_grid):
        for j, theta in enumerate(theta_grid):
            # 生成导向矢量
            a = generate_joint_steering_vector(r, theta)
            
            # MUSIC 谱函数: P = 1 / (a^H * Un * Un^H * a)
            projection = noise_subspace @ noise_subspace.conj().T @ a
            denominator = np.abs(np.vdot(a, projection))
            
            # 避免除零
            spectrum[i, j] = 1.0 / (denominator + 1e-10)
    
    # 找谱峰
    max_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    r_est = r_grid[max_idx[0]]
    theta_est = theta_grid[max_idx[1]]
    
    return r_est, theta_est, spectrum


def music_2d_refined(R, r_coarse, theta_coarse, num_signals=1, search_range_r=200, search_range_theta=10):
    """
    精细搜索版本：先粗搜索，再在峰值附近精细搜索
    
    Args:
        R: 协方差矩阵
        r_coarse, theta_coarse: 粗搜索结果
        num_signals: 信号源数量
        search_range_r: 精细搜索的距离范围 [m]
        search_range_theta: 精细搜索的角度范围 [度]
    
    Returns:
        r_est, theta_est: 精细估计结果
    """
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    noise_subspace = eigenvectors[:, num_signals:]
    
    # 精细搜索网格（更密集）
    r_fine = np.linspace(
        max(r_min, r_coarse - search_range_r/2),
        min(r_max, r_coarse + search_range_r/2),
        100
    )
    theta_fine = np.linspace(
        max(theta_min, theta_coarse - search_range_theta/2),
        min(theta_max, theta_coarse + search_range_theta/2),
        50
    )
    
    spectrum = np.zeros((len(r_fine), len(theta_fine)))
    
    for i, r in enumerate(r_fine):
        for j, theta in enumerate(theta_fine):
            a = generate_joint_steering_vector(r, theta)
            projection = noise_subspace @ noise_subspace.conj().T @ a
            denominator = np.abs(np.vdot(a, projection))
            spectrum[i, j] = 1.0 / (denominator + 1e-10)
    
    max_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    r_est = r_fine[max_idx[0]]
    theta_est = theta_fine[max_idx[1]]
    
    return r_est, theta_est


def evaluate_music(num_samples=100, snr_db=10, verbose=True):
    """
    评估 MUSIC 算法性能
    
    Args:
        num_samples: 测试样本数
        snr_db: 信噪比 [dB]
        verbose: 是否打印详细信息
    
    Returns:
        rmse_r, rmse_theta: 距离和角度的 RMSE
    """
    np.random.seed(42)
    
    errors_r = []
    errors_theta = []
    times = []
    
    print(f"\n{'='*70}")
    print(f"MUSIC 算法评估: SNR = {snr_db} dB, 样本数 = {num_samples}")
    print(f"{'='*70}\n")
    
    for i in range(num_samples):
        # 随机生成目标
        r_true = np.random.uniform(r_min, r_max)
        theta_true = np.random.uniform(theta_min, theta_max)
        
        # 生成信号
        Y = generate_echo_signal([(r_true, theta_true)], snr_db, L=500)
        R = compute_sample_covariance_matrix(Y)
        
        # MUSIC 估计（两步法：粗搜索+精细搜索）
        start_time = time.time()
        
        # 步骤1: 粗搜索
        r_coarse, theta_coarse, _ = music_2d_fda(R, num_signals=1)
        
        # 步骤2: 精细搜索
        r_est, theta_est = music_2d_refined(R, r_coarse, theta_coarse)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        # 计算误差
        error_r = abs(r_est - r_true)
        error_theta = abs(theta_est - theta_true)
        
        errors_r.append(error_r)
        errors_theta.append(error_theta)
        
        if verbose and (i + 1) % 20 == 0:
            print(f"样本 {i+1}/{num_samples}: "
                  f"真值=({r_true:.1f}m, {theta_true:.1f}°), "
                  f"估计=({r_est:.1f}m, {theta_est:.1f}°), "
                  f"误差=({error_r:.1f}m, {error_theta:.2f}°), "
                  f"时间={elapsed:.3f}s")
    
    # 统计结果
    rmse_r = np.sqrt(np.mean(np.array(errors_r)**2))
    rmse_theta = np.sqrt(np.mean(np.array(errors_theta)**2))
    avg_time = np.mean(times)
    
    print(f"\n{'='*70}")
    print(f"MUSIC 性能总结 (SNR = {snr_db} dB)")
    print(f"{'='*70}")
    print(f"  RMSE_r     = {rmse_r:.2f} m")
    print(f"  RMSE_θ     = {rmse_theta:.2f}°")
    print(f"  平均时间   = {avg_time:.3f} s/样本")
    print(f"  总时间     = {sum(times):.1f} s")
    print(f"{'='*70}\n")
    
    return rmse_r, rmse_theta


def evaluate_music_vs_snr(snr_range=range(-10, 25, 5), num_samples=50):
    """
    评估 MUSIC 在不同 SNR 下的性能
    
    Args:
        snr_range: SNR 范围
        num_samples: 每个 SNR 的测试样本数
    """
    print(f"\n{'='*70}")
    print(f"MUSIC 算法 vs SNR 性能评估")
    print(f"{'='*70}\n")
    
    results = []
    
    for snr in snr_range:
        rmse_r, rmse_theta = evaluate_music(num_samples=num_samples, snr_db=snr, verbose=False)
        results.append({
            'snr': snr,
            'rmse_r': rmse_r,
            'rmse_theta': rmse_theta
        })
        print(f"SNR = {snr:3d} dB: RMSE_r = {rmse_r:6.2f} m, RMSE_θ = {rmse_theta:5.2f}°")
    
    print(f"\n{'='*70}")
    print("MUSIC 性能基准建立完成")
    print(f"{'='*70}\n")
    
    return results


def test_single_case():
    """测试单个案例，可视化 MUSIC 谱"""
    print("\n" + "="*70)
    print("单案例测试：可视化 MUSIC 谱")
    print("="*70 + "\n")
    
    # 生成信号
    r_true, theta_true = 1000, 0
    snr = 30
    print(f"真实目标: r = {r_true} m, θ = {theta_true}°, SNR = {snr} dB\n")
    
    Y = generate_echo_signal([(r_true, theta_true)], snr, L=500)
    R = compute_sample_covariance_matrix(Y)
    
    # MUSIC 估计
    print("执行 MUSIC 算法...")
    start = time.time()
    
    # 粗搜索
    r_coarse, theta_coarse, spectrum = music_2d_fda(R, num_signals=1)
    print(f"粗搜索结果: r = {r_coarse:.1f} m, θ = {theta_coarse:.1f}°")
    
    # 精细搜索
    r_est, theta_est = music_2d_refined(R, r_coarse, theta_coarse)
    elapsed = time.time() - start
    
    print(f"精细搜索结果: r = {r_est:.1f} m, θ = {theta_est:.1f}°")
    print(f"估计误差: Δr = {abs(r_est - r_true):.1f} m, Δθ = {abs(theta_est - theta_true):.2f}°")
    print(f"计算时间: {elapsed:.3f} s")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FDA-MIMO 2D-MUSIC 基准测试")
    print("="*70)
    
    # 测试1: 单案例
    test_single_case()
    
    # 测试2: 固定 SNR 性能评估
    print("\n测试 1: 固定 SNR = 10 dB")
    evaluate_music(num_samples=100, snr_db=10, verbose=False)
    
    # 测试3: 不同 SNR 性能
    print("\n测试 2: MUSIC vs SNR")
    evaluate_music_vs_snr(snr_range=range(-10, 25, 5), num_samples=50)
    
    print("\n提示: MUSIC 的性能可以作为 CVNN 的参考基准")
    print("      如果 CVNN 能超越 MUSIC，说明深度学习有优势！")
