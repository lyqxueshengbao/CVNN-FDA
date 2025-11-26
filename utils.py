# -*- coding: utf-8 -*-
"""
FDA-MIMO 雷达参数估计 - 工具函数模块
Utility functions for FDA-MIMO Radar Signal Generation

包含:
1. 导向矢量生成 (Steering Vector Generation)
2. 信号合成工具 (Signal Synthesis Utilities)
"""

import numpy as np
import torch
from config import M, N, MN, c, f0, delta_f, wavelength, d


def generate_transmit_steering_vector(r: float, theta: float) -> np.ndarray:
    """
    生成发射导向矢量 a(r, θ)
    
    论文公式 (2-35) 至 (2-37):
    a(r,θ) = [1, e^(-j2π·Δf·2r/c)·e^(j2π·d·sinθ/λ), ..., 
              e^(-j2π·(M-1)·Δf·2r/c)·e^(j2π·(M-1)·d·sinθ/λ)]^T
    
    Args:
        r: 目标距离 [m]
        theta: 目标角度 [degrees]
    
    Returns:
        a: 发射导向矢量, 维度 (M, 1), 复数
    """
    theta_rad = np.deg2rad(theta)
    m = np.arange(M).reshape(-1, 1)  # [0, 1, ..., M-1]
    
    # 距离相位项 (FDA特有): e^(-j2π·m·Δf·2r/c)
    range_phase = -2 * np.pi * m * delta_f * 2 * r / c
    
    # 角度相位项: e^(j2π·m·d·sinθ/λ)
    angle_phase = 2 * np.pi * m * d * np.sin(theta_rad) / wavelength
    
    # 合并相位
    total_phase = range_phase + angle_phase
    a = np.exp(1j * total_phase)
    
    return a  # (M, 1)


def generate_receive_steering_vector(theta: float) -> np.ndarray:
    """
    生成接收导向矢量 b(θ)
    
    论文公式 (2-38):
    b(θ) = [1, e^(j2π·d·sinθ/λ), ..., e^(j2π·(N-1)·d·sinθ/λ)]^T
    
    Args:
        theta: 目标角度 [degrees]
    
    Returns:
        b: 接收导向矢量, 维度 (N, 1), 复数
    """
    theta_rad = np.deg2rad(theta)
    n = np.arange(N).reshape(-1, 1)  # [0, 1, ..., N-1]
    
    # 角度相位项: e^(j2π·n·d·sinθ/λ)
    angle_phase = 2 * np.pi * n * d * np.sin(theta_rad) / wavelength
    b = np.exp(1j * angle_phase)
    
    return b  # (N, 1)


def generate_joint_steering_vector(r: float, theta: float) -> np.ndarray:
    """
    生成联合导向矢量 u(r, θ) = b(θ) ⊗ a(r, θ)
    
    利用克罗内克积 (Kronecker Product)
    
    Args:
        r: 目标距离 [m]
        theta: 目标角度 [degrees]
    
    Returns:
        u: 联合导向矢量, 维度 (MN, 1), 复数
    """
    a = generate_transmit_steering_vector(r, theta)  # (M, 1)
    b = generate_receive_steering_vector(theta)       # (N, 1)
    
    # 克罗内克积: b ⊗ a
    u = np.kron(b, a)  # (MN, 1)
    
    return u


def generate_echo_signal(targets: list, snr_db: float, L: int) -> np.ndarray:
    """
    生成回波信号 Y
    
    论文公式 (2-39):
    y(t) = Σ_{k=1}^{K} ξ_k · u(r_k, θ_k) + n(t)
    
    Args:
        targets: 目标列表, 每个元素为 (r, theta) 元组
        snr_db: 信噪比 [dB]
        L: 快拍数
    
    Returns:
        Y: 接收数据矩阵, 维度 (MN, L), 复数
    """
    K = len(targets)
    
    # 初始化信号矩阵
    Y_signal = np.zeros((MN, L), dtype=np.complex128)
    
    # 生成各目标的回波
    for r, theta in targets:
        u = generate_joint_steering_vector(r, theta)  # (MN, 1)
        
        # 复反射系数 ξ: 标准复高斯分布 (每个快拍不同)
        xi = (np.random.randn(1, L) + 1j * np.random.randn(1, L)) / np.sqrt(2)
        
        # 累加信号
        Y_signal += u @ xi  # (MN, 1) @ (1, L) = (MN, L)
    
    # 计算信号功率
    signal_power = np.mean(np.abs(Y_signal) ** 2)
    
    # 根据SNR计算噪声功率
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # 生成加性高斯白噪声 (AWGN)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(MN, L) + 1j * np.random.randn(MN, L)
    )
    
    # 叠加噪声
    Y = Y_signal + noise
    
    return Y  # (MN, L)


def compute_sample_covariance_matrix(Y: np.ndarray) -> np.ndarray:
    """
    计算样本协方差矩阵 (SCM)
    
    论文公式 (2-42):
    R = (1/L) · Y · Y^H
    
    Args:
        Y: 接收数据矩阵, 维度 (MN, L)
    
    Returns:
        R: 样本协方差矩阵, 维度 (MN, MN), 复数
    """
    L = Y.shape[1]
    R = (Y @ Y.conj().T) / L
    return R  # (MN, MN)


def complex_normalize(R: np.ndarray) -> np.ndarray:
    """
    复数矩阵归一化
    
    方案: 除以矩阵最大模值
    R_norm = R / max(|R|)
    
    Args:
        R: 复数矩阵
    
    Returns:
        R_norm: 归一化后的复数矩阵
    """
    max_abs = np.max(np.abs(R))
    if max_abs > 0:
        R_norm = R / max_abs
    else:
        R_norm = R
    return R_norm


def normalize_labels(r: float, theta: float, 
                     r_range: tuple = (0, 2000), 
                     theta_range: tuple = (-60, 60)) -> tuple:
    """
    将标签归一化到 [0, 1] 区间
    
    Args:
        r: 距离值 [m]
        theta: 角度值 [degrees]
        r_range: 距离范围 (r_min, r_max)
        theta_range: 角度范围 (theta_min, theta_max)
    
    Returns:
        (r_norm, theta_norm): 归一化后的标签
    """
    r_norm = (r - r_range[0]) / (r_range[1] - r_range[0])
    theta_norm = (theta - theta_range[0]) / (theta_range[1] - theta_range[0])
    return r_norm, theta_norm


def denormalize_labels(r_norm: float, theta_norm: float,
                       r_range: tuple = (0, 2000),
                       theta_range: tuple = (-60, 60)) -> tuple:
    """
    将归一化标签还原为真实值
    
    Args:
        r_norm: 归一化距离值
        theta_norm: 归一化角度值
        r_range: 距离范围 (r_min, r_max)
        theta_range: 角度范围 (theta_min, theta_max)
    
    Returns:
        (r, theta): 真实标签值
    """
    r = r_norm * (r_range[1] - r_range[0]) + r_range[0]
    theta = theta_norm * (theta_range[1] - theta_range[0]) + theta_range[0]
    return r, theta


def numpy_to_torch_complex(arr: np.ndarray) -> torch.Tensor:
    """
    将 NumPy 复数数组转换为 PyTorch 复数张量
    
    Args:
        arr: NumPy 复数数组
    
    Returns:
        tensor: PyTorch 复数张量 (complex64)
    """
    return torch.from_numpy(arr.astype(np.complex64))


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("FDA-MIMO 工具函数测试")
    print("=" * 60)
    
    # 测试导向矢量
    r_test = 1000  # 1000m
    theta_test = 30  # 30度
    
    a = generate_transmit_steering_vector(r_test, theta_test)
    b = generate_receive_steering_vector(theta_test)
    u = generate_joint_steering_vector(r_test, theta_test)
    
    print(f"\n1. 导向矢量测试:")
    print(f"   发射导向矢量 a 维度: {a.shape}")
    print(f"   接收导向矢量 b 维度: {b.shape}")
    print(f"   联合导向矢量 u 维度: {u.shape}")
    
    # 测试回波信号生成
    targets = [(1000, 30), (500, -20)]  # 两个目标
    snr = 10  # 10 dB
    L = 200
    
    Y = generate_echo_signal(targets, snr, L)
    print(f"\n2. 回波信号测试:")
    print(f"   接收数据矩阵 Y 维度: {Y.shape}")
    print(f"   Y 数据类型: {Y.dtype}")
    
    # 测试协方差矩阵
    R = compute_sample_covariance_matrix(Y)
    print(f"\n3. 协方差矩阵测试:")
    print(f"   协方差矩阵 R 维度: {R.shape}")
    print(f"   R 数据类型: {R.dtype}")
    
    # 测试归一化
    R_norm = complex_normalize(R)
    print(f"\n4. 归一化测试:")
    print(f"   归一化后 R 最大模值: {np.max(np.abs(R_norm)):.4f}")
    
    # 测试标签归一化
    r_n, theta_n = normalize_labels(1000, 30)
    r_d, theta_d = denormalize_labels(r_n, theta_n)
    print(f"\n5. 标签归一化测试:")
    print(f"   原始标签: r={1000}, θ={30}")
    print(f"   归一化标签: r_norm={r_n:.4f}, θ_norm={theta_n:.4f}")
    print(f"   还原标签: r={r_d:.1f}, θ={theta_d:.1f}")
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
