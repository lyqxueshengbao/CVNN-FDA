# -*- coding: utf-8 -*-
"""
诊断脚本：验证信号生成和协方差矩阵是否正确
检查输入数据是否包含足够的距离-角度信息
"""

import numpy as np
import matplotlib.pyplot as plt
from config import r_min, r_max, theta_min, theta_max, MN, M, N, delta_f, c, wavelength, d
from utils import (
    generate_joint_steering_vector, 
    generate_echo_signal, 
    compute_sample_covariance_matrix,
    complex_normalize
)

def test_steering_vector_sensitivity():
    """测试导向矢量对距离和角度的敏感性"""
    print("=" * 70)
    print("测试1: 导向矢量敏感性")
    print("=" * 70)
    
    # 固定角度，变化距离
    theta = 0
    distances = [0, 100, 500, 1000, 2000]
    
    print(f"\n固定角度 θ={theta}°，变化距离：")
    u_base = generate_joint_steering_vector(distances[0], theta)
    
    for r in distances:
        u = generate_joint_steering_vector(r, theta)
        # 计算与基准的相关性
        corr = np.abs(np.vdot(u_base, u)) / (np.linalg.norm(u_base) * np.linalg.norm(u))
        print(f"  r={r:4d}m: 相关系数={corr:.4f}")
    
    # 固定距离，变化角度
    r = 1000
    angles = [-60, -30, 0, 30, 60]
    
    print(f"\n固定距离 r={r}m，变化角度：")
    u_base = generate_joint_steering_vector(r, angles[2])  # 0度为基准
    
    for theta in angles:
        u = generate_joint_steering_vector(r, theta)
        corr = np.abs(np.vdot(u_base, u)) / (np.linalg.norm(u_base) * np.linalg.norm(u))
        print(f"  θ={theta:3d}°: 相关系数={corr:.4f}")


def test_distance_resolution():
    """测试距离分辨能力"""
    print("\n" + "=" * 70)
    print("测试2: 距离分辨能力")
    print("=" * 70)
    
    # FDA的距离分辨率由频率偏移决定
    # Δr = c / (2 * M * Δf)
    range_resolution = c / (2 * M * delta_f)
    print(f"理论距离分辨率: Δr = c/(2MΔf) = {range_resolution:.2f} m")
    
    theta = 0
    r_base = 1000
    
    # 测试不同距离偏移的可分辨性
    offsets = [1, 5, 10, 50, 100, 500]
    u_base = generate_joint_steering_vector(r_base, theta)
    
    print(f"\n基准: r={r_base}m, θ={theta}°")
    print("距离偏移 vs 相关系数:")
    for dr in offsets:
        u = generate_joint_steering_vector(r_base + dr, theta)
        corr = np.abs(np.vdot(u_base, u)) / (np.linalg.norm(u_base) * np.linalg.norm(u))
        distinguishable = "✓ 可区分" if corr < 0.9 else "✗ 难区分"
        print(f"  Δr={dr:3d}m: corr={corr:.4f} {distinguishable}")


def test_covariance_matrix_structure():
    """测试协方差矩阵结构"""
    print("\n" + "=" * 70)
    print("测试3: 协方差矩阵结构")
    print("=" * 70)
    
    # 生成两个不同目标的协方差矩阵
    snr = 30  # 高SNR
    L = 500
    
    targets = [
        (500, 0),    # 目标1: 500m, 0°
        (1500, 30),  # 目标2: 1500m, 30°
    ]
    
    for r, theta in targets:
        Y = generate_echo_signal([(r, theta)], snr, L)
        R = compute_sample_covariance_matrix(Y)
        R_norm = complex_normalize(R)
        
        # 分析协方差矩阵特性
        eigenvalues = np.linalg.eigvalsh(R)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        print(f"\n目标: r={r}m, θ={theta}°")
        print(f"  协方差矩阵形状: {R.shape}")
        print(f"  最大特征值: {eigenvalues[0]:.4f}")
        print(f"  前5个特征值: {eigenvalues[:5]}")
        print(f"  特征值比 (λ1/λ2): {eigenvalues[0]/eigenvalues[1]:.2f}")
        print(f"  归一化后范数: {np.linalg.norm(R_norm):.4f}")


def test_label_range():
    """测试标签范围是否合理"""
    print("\n" + "=" * 70)
    print("测试4: 标签归一化范围")
    print("=" * 70)
    
    print(f"距离范围: [{r_min}, {r_max}] m")
    print(f"角度范围: [{theta_min}, {theta_max}]°")
    print(f"距离跨度: {r_max - r_min} m")
    print(f"角度跨度: {theta_max - theta_min}°")
    
    # 检查归一化
    test_r = [0, 500, 1000, 1500, 2000]
    test_theta = [-60, -30, 0, 30, 60]
    
    print("\n归一化测试:")
    for r in test_r:
        r_norm = (r - r_min) / (r_max - r_min)
        print(f"  r={r:4d}m -> r_norm={r_norm:.3f}")
    
    for theta in test_theta:
        theta_norm = (theta - theta_min) / (theta_max - theta_min)
        print(f"  θ={theta:3d}° -> θ_norm={theta_norm:.3f}")


def test_input_discriminability():
    """测试不同目标的输入是否可区分"""
    print("\n" + "=" * 70)
    print("测试5: 输入可区分性 (关键测试)")
    print("=" * 70)
    
    snr = 30
    L = 500
    
    # 生成多个目标的协方差矩阵并计算相似度
    test_targets = [
        (500, 0),
        (510, 0),    # 距离差10m
        (550, 0),    # 距离差50m
        (600, 0),    # 距离差100m
        (500, 5),    # 角度差5°
        (500, 10),   # 角度差10°
    ]
    
    # 生成基准
    Y_base = generate_echo_signal([test_targets[0]], snr, L)
    R_base = compute_sample_covariance_matrix(Y_base)
    R_base_norm = complex_normalize(R_base)
    R_base_flat = np.concatenate([R_base_norm.real.flatten(), R_base_norm.imag.flatten()])
    
    print(f"基准目标: r={test_targets[0][0]}m, θ={test_targets[0][1]}°")
    print("\n与基准的余弦相似度:")
    
    for r, theta in test_targets:
        Y = generate_echo_signal([(r, theta)], snr, L)
        R = compute_sample_covariance_matrix(Y)
        R_norm = complex_normalize(R)
        R_flat = np.concatenate([R_norm.real.flatten(), R_norm.imag.flatten()])
        
        # 余弦相似度
        cos_sim = np.dot(R_base_flat, R_flat) / (np.linalg.norm(R_base_flat) * np.linalg.norm(R_flat))
        
        dr = r - test_targets[0][0]
        dtheta = theta - test_targets[0][1]
        distinguishable = "✓" if cos_sim < 0.95 else "✗"
        print(f"  r={r:4d}m (Δr={dr:3d}), θ={theta:2d}° (Δθ={dtheta:2d}): sim={cos_sim:.4f} {distinguishable}")


def test_delta_f_impact():
    """测试6: 不同 delta_f 对分辨率的影响"""
    print("\n" + "="*70)
    print("测试6: 不同 delta_f 对距离分辨率的影响")
    print("="*70)
    
    c_val = 3e8
    M_val = 10
    
    # 测试不同的 delta_f 值
    delta_f_values = [30e3, 100e3, 300e3, 1e6, 3e6, 10e6]
    
    print(f"\n当前参数: M={M_val} 发射天线\n")
    print(f"{'delta_f':>12s} | {'理论分辨率':>12s} | {'10m差异相似度':>15s}")
    print("-" * 50)
    
    for delta_f_val in delta_f_values:
        # 理论分辨率
        resolution = c_val / (2 * M_val * delta_f_val)
        
        # 实际测试: 计算 10m 差异的相似度
        r_base, theta_base = 500, 0
        r_test = r_base + 10
        
        # 生成导向矢量
        f0_val = 1e9
        wavelength_val = c_val / f0_val
        d_val = wavelength_val / 2
        N_val = 10
        
        def steering_vector_custom(r, theta):
            theta_rad = np.deg2rad(theta)
            m_arr = np.arange(M_val)
            n_arr = np.arange(N_val)
            
            # 发射导向矢量
            phase_t = 2 * np.pi * (m_arr * d_val * np.sin(theta_rad) / wavelength_val - 
                                   m_arr * delta_f_val * 2 * r / c_val)
            a_t = np.exp(1j * phase_t)
            
            # 接收导向矢量
            phase_r = 2 * np.pi * n_arr * d_val * np.sin(theta_rad) / wavelength_val
            a_r = np.exp(1j * phase_r)
            
            return np.kron(a_t, a_r)
        
        sv_base = steering_vector_custom(r_base, theta_base)
        sv_test = steering_vector_custom(r_test, theta_base)
        
        # 计算相关系数
        corr = np.abs(np.vdot(sv_base, sv_test)) / (np.linalg.norm(sv_base) * np.linalg.norm(sv_test))
        
        status = "✓ 可区分" if corr < 0.95 else "✗ 难区分"
        print(f"{delta_f_val/1e3:>10.0f}kHz | {resolution:>10.1f}m | {corr:.4f} {status}")
    
    print("\n结论:")
    print("- 要达到 10m 分辨率，需要 delta_f ≈ 1.5 MHz")
    print("- 要达到 50m 分辨率，需要 delta_f ≈ 300 kHz")
    print("- 当前 30kHz 只能达到 500m 分辨率")


def test_with_new_delta_f(new_delta_f=300e3):
    """测试7: 用新的 delta_f 重新测试输入可区分性"""
    print("\n" + "="*70)
    print(f"测试7: 使用 delta_f = {new_delta_f/1e3:.0f}kHz 重新测试")
    print("="*70)
    
    c_val = 3e8
    f0_val = 1e9
    M_val, N_val = 10, 10
    wavelength_val = c_val / f0_val
    d_val = wavelength_val / 2
    L_val = 500
    SNR_dB = 30
    
    resolution = c_val / (2 * M_val * new_delta_f)
    print(f"\n新理论分辨率: {resolution:.1f}m")
    
    def steering_vector_new(r, theta):
        theta_rad = np.deg2rad(theta)
        m_arr = np.arange(M_val)
        n_arr = np.arange(N_val)
        
        phase_t = 2 * np.pi * (m_arr * d_val * np.sin(theta_rad) / wavelength_val - 
                               m_arr * new_delta_f * 2 * r / c_val)
        a_t = np.exp(1j * phase_t)
        
        phase_r = 2 * np.pi * n_arr * d_val * np.sin(theta_rad) / wavelength_val
        a_r = np.exp(1j * phase_r)
        
        return np.kron(a_t, a_r)
    
    def generate_input_new(r, theta):
        a = steering_vector_new(r, theta)
        signal_power = 1.0
        noise_power = signal_power / (10 ** (SNR_dB / 10))
        
        S = np.sqrt(signal_power / 2) * (np.random.randn(1, L_val) + 1j * np.random.randn(1, L_val))
        N_noise = np.sqrt(noise_power / 2) * (np.random.randn(M_val*N_val, L_val) + 1j * np.random.randn(M_val*N_val, L_val))
        
        Y = a.reshape(-1, 1) @ S + N_noise
        R = Y @ Y.conj().T / L_val
        
        R_real = np.stack([R.real, R.imag], axis=0)
        R_real = R_real / (np.linalg.norm(R_real) + 1e-10)
        return R_real.flatten()
    
    # 测试不同距离差异
    r_base, theta_base = 500, 0
    print(f"\n基准: r={r_base}m, θ={theta_base}°")
    print("\n距离差异测试:")
    
    np.random.seed(42)
    input_base = generate_input_new(r_base, theta_base)
    
    for delta_r in [1, 5, 10, 20, 50, 100]:
        np.random.seed(43)  # 不同的种子
        input_test = generate_input_new(r_base + delta_r, theta_base)
        
        sim = np.dot(input_base, input_test) / (np.linalg.norm(input_base) * np.linalg.norm(input_test))
        status = "✓" if sim < 0.95 else "✗"
        print(f"  Δr={delta_r:3d}m: 相似度={sim:.4f} {status}")


if __name__ == "__main__":
    np.random.seed(42)
    
    test_steering_vector_sensitivity()
    test_distance_resolution()
    test_covariance_matrix_structure()
    test_label_range()
    test_input_discriminability()
    
    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)
    print("\n关键问题:")
    print("1. 如果距离差10m的输入相似度>0.99，说明网络很难区分")
    print("2. 理论分辨率如果>10m，那10m以下的误差可能无法达到")
    print("=" * 70)
    
    # 额外测试: delta_f 参数调优
    print("\n\n" + "="*70)
    print("额外测试: delta_f 参数调优")
    print("="*70)
    
    test_delta_f_impact()
    test_with_new_delta_f(300e3)   # 测试 300kHz
    test_with_new_delta_f(1.5e6)  # 测试 1.5MHz
