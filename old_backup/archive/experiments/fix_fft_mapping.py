#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDA-MIMO 2D-FFT 正确映射关系推导与验证

通过数据驱动方法找到FFT索引到参数的正确映射
"""

import numpy as np
from config import M, N, c, f0, delta_f, d, wavelength, r_min, r_max
from utils import generate_echo_signal, compute_sample_covariance_matrix

def calibrate_fft_mapping(fft_size_M=64, fft_size_N=64, n_samples=100):
    """
    通过数据驱动方法校准FFT映射
    
    方法：生成已知(r,θ)的信号，记录FFT峰值位置，拟合映射关系
    """
    
    print("=" * 70)
    print("校准FFT到参数的映射关系")
    print("=" * 70)
    print(f"FFT尺寸: {fft_size_M} x {fft_size_N}")
    
    # 采样网格
    r_samples = np.linspace(100, 1900, 20)
    theta_samples = np.linspace(-55, 55, 23)
    
    results = []
    
    print(f"\n生成校准数据 ({len(r_samples)}x{len(theta_samples)}={len(r_samples)*len(theta_samples)} 点)...")
    
    for r in r_samples:
        for theta in theta_samples:
            # 生成信号
            Y = generate_echo_signal([(r, theta)], snr_db=20.0, L=100)
            R = compute_sample_covariance_matrix(Y)
            
            # 重排: u = b ⊗ a, 所以 reshape 要注意顺序
            # u[n*M + m] = b[n] * a[m] 
            # reshape(N, M).T 得到 (M, N)
            signal_vec = R[:, 0].reshape(N, M).T
            
            # 2D-FFT
            beamspace = np.fft.fft2(signal_vec, s=(fft_size_M, fft_size_N))
            beamspace = np.fft.fftshift(beamspace)
            
            mag = np.abs(beamspace)
            idx_m, idx_n = np.unravel_index(np.argmax(mag), mag.shape)
            
            results.append({
                'r': r, 'theta': theta,
                'sin_theta': np.sin(np.deg2rad(theta)),
                'idx_m': idx_m, 'idx_n': idx_n
            })
    
    # 转为数组
    r_arr = np.array([res['r'] for res in results])
    theta_arr = np.array([res['theta'] for res in results])
    sin_theta_arr = np.array([res['sin_theta'] for res in results])
    idx_m_arr = np.array([res['idx_m'] for res in results])
    idx_n_arr = np.array([res['idx_n'] for res in results])
    
    print("\n拟合映射关系...")
    
    # 角度映射: idx_n = k1 * sin(θ) + b1
    A_n = np.column_stack([sin_theta_arr, np.ones_like(sin_theta_arr)])
    params_n, _, _, _ = np.linalg.lstsq(A_n, idx_n_arr, rcond=None)
    k1, b1 = params_n
    
    # 距离映射: idx_m = k2 * sin(θ) + k3 * r + b2
    A_m = np.column_stack([sin_theta_arr, r_arr, np.ones_like(r_arr)])
    params_m, _, _, _ = np.linalg.lstsq(A_m, idx_m_arr, rcond=None)
    k2, k3, b2 = params_m
    
    print(f"\n映射公式:")
    print(f"  角度: idx_n = {k1:.4f} * sin(θ) + {b1:.4f}")
    print(f"  距离: idx_m = {k2:.4f} * sin(θ) + {k3:.8f} * r + {b2:.4f}")
    
    print(f"\n反推公式:")
    print(f"  θ = arcsin((idx_n - {b1:.4f}) / {k1:.4f})")
    print(f"  r = (idx_m - {k2:.4f} * sin(θ) - {b2:.4f}) / {k3:.8f}")
    
    # 验证
    sin_theta_pred = (idx_n_arr - b1) / k1
    sin_theta_pred = np.clip(sin_theta_pred, -1, 1)
    theta_pred = np.rad2deg(np.arcsin(sin_theta_pred))
    r_pred = (idx_m_arr - k2 * sin_theta_pred - b2) / k3
    
    theta_errors = np.abs(theta_arr - theta_pred)
    r_errors = np.abs(r_arr - r_pred)
    
    print(f"\n校准精度:")
    print(f"  角度误差: 均值={theta_errors.mean():.2f}°, 最大={theta_errors.max():.2f}°")
    print(f"  距离误差: 均值={r_errors.mean():.1f}m, 最大={r_errors.max():.1f}m")
    
    mapping_params = {
        'k1': k1, 'b1': b1,
        'k2': k2, 'k3': k3, 'b2': b2,
        'fft_size_M': fft_size_M, 'fft_size_N': fft_size_N
    }
    
    return mapping_params


def test_mapping(params, snr_db=10.0):
    """测试映射精度"""
    
    print("\n" + "=" * 70)
    print(f"测试独立数据 (SNR={snr_db}dB)")
    print("=" * 70)
    
    k1, b1 = params['k1'], params['b1']
    k2, k3, b2 = params['k2'], params['k3'], params['b2']
    fft_size_M = params['fft_size_M']
    fft_size_N = params['fft_size_N']
    
    # 随机生成测试数据
    np.random.seed(42)
    n_test = 50
    r_test = np.random.uniform(100, 1900, n_test)
    theta_test = np.random.uniform(-55, 55, n_test)
    
    r_errors = []
    theta_errors = []
    
    for r_true, theta_true in zip(r_test, theta_test):
        Y = generate_echo_signal([(r_true, theta_true)], snr_db=snr_db, L=500)
        R = compute_sample_covariance_matrix(Y)
        signal_vec = R[:, 0].reshape(N, M).T
        
        beamspace = np.fft.fft2(signal_vec, s=(fft_size_M, fft_size_N))
        beamspace = np.fft.fftshift(beamspace)
        mag = np.abs(beamspace)
        idx_m, idx_n = np.unravel_index(np.argmax(mag), mag.shape)
        
        # 反推参数
        sin_theta_est = (idx_n - b1) / k1
        sin_theta_est = np.clip(sin_theta_est, -0.999, 0.999)
        theta_est = np.rad2deg(np.arcsin(sin_theta_est))
        
        r_est = (idx_m - k2 * sin_theta_est - b2) / k3
        r_est = np.clip(r_est, r_min, r_max)
        
        r_errors.append(abs(r_true - r_est))
        theta_errors.append(abs(theta_true - theta_est))
    
    r_errors = np.array(r_errors)
    theta_errors = np.array(theta_errors)
    
    print(f"\n{n_test}个测试样本的粗估计精度:")
    print(f"  距离RMSE: {np.sqrt(np.mean(r_errors**2)):.1f}m")
    print(f"  角度RMSE: {np.sqrt(np.mean(theta_errors**2)):.2f}°")
    print(f"  距离误差: 均值={r_errors.mean():.1f}m, 中位数={np.median(r_errors):.1f}m, 最大={r_errors.max():.1f}m")
    print(f"  角度误差: 均值={theta_errors.mean():.2f}°, 中位数={np.median(theta_errors):.2f}°, 最大={theta_errors.max():.2f}°")
    
    # 误差分布
    print(f"\n误差分布:")
    print(f"  距离误差 < 50m: {(r_errors < 50).sum()}/{n_test} ({(r_errors < 50).sum()/n_test*100:.0f}%)")
    print(f"  距离误差 < 100m: {(r_errors < 100).sum()}/{n_test} ({(r_errors < 100).sum()/n_test*100:.0f}%)")
    print(f"  角度误差 < 5°: {(theta_errors < 5).sum()}/{n_test} ({(theta_errors < 5).sum()/n_test*100:.0f}%)")
    print(f"  角度误差 < 10°: {(theta_errors < 10).sum()}/{n_test} ({(theta_errors < 10).sum()/n_test*100:.0f}%)")
    
    return r_errors, theta_errors


if __name__ == "__main__":
    # 校准
    params = calibrate_fft_mapping(fft_size_M=64, fft_size_N=64)
    
    # 测试不同SNR
    for snr in [20, 10, 0, -5]:
        test_mapping(params, snr_db=snr)
