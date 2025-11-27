# -*- coding: utf-8 -*-
"""
FDA-MIMO 雷达 Beamspace 数据集
基于粗精结合策略: FFT粗定位 + CVNN精修正
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from config import *
from utils import generate_echo_signal, compute_sample_covariance_matrix, complex_normalize


def fft_coarse_estimation(R: np.ndarray, fft_size_M: int = 64, fft_size_N: int = 64) -> Tuple[int, int, np.ndarray]:
    """
    使用2D-FFT进行粗估计
    
    Args:
        R: 协方差矩阵 (MN, MN)
        fft_size_M: M维FFT大小
        fft_size_N: N维FFT大小
    
    Returns:
        idx_m: M维峰值索引
        idx_n: N维峰值索引
        beamspace: FFT后的波束域矩阵
    """
    # 取协方差矩阵第一列作为信号特征，重排成(M, N)
    # 根据 u(r,θ) = b(θ) ⊗ a(r,θ)，Kronecker积结构是 b ⊗ a
    # 因此reshape应该是先N后M的顺序
    signal_proxy = R[:, 0].reshape(N, M)  # (N, M)
    
    # 2D-FFT到波束域
    beamspace = np.fft.fft2(signal_proxy, s=(fft_size_N, fft_size_M))
    beamspace = np.fft.fftshift(beamspace)  # 零频移到中心
    
    # 找到能量峰值
    mag = np.abs(beamspace)
    idx_n, idx_m = np.unravel_index(np.argmax(mag), mag.shape)
    
    return idx_m, idx_n, beamspace


def indices_to_coarse_params(idx_m: int, idx_n: int, 
                             fft_size_M: int = 64, fft_size_N: int = 64) -> Tuple[float, float]:
    """
    将FFT索引转换为粗估计参数
    
    物理原理:
    - N维对应角度theta (ULA相位关系)
    - M维对应距离和角度的耦合 (FDA特性)
    
    Args:
        idx_m: M维索引
        idx_n: N维索引
        fft_size_M: M维FFT大小
        fft_size_N: N维FFT大小
    
    Returns:
        r_coarse: 粗估计距离 [m]
        theta_coarse: 粗估计角度 [degrees]
    """
    # 角度估计 (仅与N维相关)
    # sin(theta) = (k_n / N) * (lambda / d)，其中k_n是归一化频率索引
    k_n = (idx_n - fft_size_N // 2) / fft_size_N  # 归一化到[-0.5, 0.5]
    sin_theta = k_n * (wavelength / d) * N / fft_size_N
    sin_theta = np.clip(sin_theta, -1, 1)  # 限制在[-1, 1]
    theta_coarse = np.rad2deg(np.arcsin(sin_theta))
    
    # 距离估计 (与M维和角度耦合)
    # FDA频率-距离关系: k_m ≈ (4πΔf/c) * r / M
    k_m = (idx_m - fft_size_M // 2) / fft_size_M  # 归一化到[-0.5, 0.5]
    # 考虑角度影响，简化估计
    r_coarse = k_m * fft_size_M * c / (4 * np.pi * delta_f * M)
    r_coarse = np.clip(r_coarse, r_min, r_max)
    
    return r_coarse, theta_coarse


def extract_patch(beamspace: np.ndarray, idx_m: int, idx_n: int, 
                 patch_size: int = 5) -> np.ndarray:
    """
    从波束域提取patch
    
    Args:
        beamspace: 波束域矩阵 (fft_size_N, fft_size_M)
        idx_m: M维中心索引
        idx_n: N维中心索引
        patch_size: Patch大小
    
    Returns:
        patch: 提取的复数patch (patch_size, patch_size)
    """
    pad = patch_size // 2
    H, W = beamspace.shape
    
    # 使用循环边界处理
    patch = np.zeros((patch_size, patch_size), dtype=np.complex64)
    
    for i in range(patch_size):
        for j in range(patch_size):
            # 循环索引
            row = (idx_n - pad + i) % H
            col = (idx_m - pad + j) % W
            patch[i, j] = beamspace[row, col]
    
    return patch


class FDADatasetBeamspace(Dataset):
    """
    FDA-MIMO Beamspace 数据集 (粗精结合)
    
    返回:
    - 5x5 复数patch (归一化)
    - 残差标签 (delta_r, delta_theta)
    - 真实标签 (r_true, theta_true)
    """
    
    def __init__(self,
                 num_samples: int,
                 snr_range: Tuple[float, float] = (SNR_TRAIN_MIN, SNR_TRAIN_MAX),
                 num_targets: int = 1,
                 L: int = L_snapshots,
                 fixed_snr: Optional[float] = None,
                 fft_size_M: int = 64,
                 fft_size_N: int = 64,
                 patch_size: int = 5,
                 seed: Optional[int] = None):
        """
        Args:
            num_samples: 样本数量
            snr_range: SNR范围 [dB]
            num_targets: 目标数量 (当前只支持1)
            L: 快拍数
            fixed_snr: 固定SNR (测试集用)
            fft_size_M: M维FFT大小
            fft_size_N: N维FFT大小
            patch_size: Patch大小
            seed: 随机种子
        """
        self.num_samples = num_samples
        self.snr_range = snr_range
        self.num_targets = num_targets
        self.L = L
        self.fixed_snr = fixed_snr
        self.fft_size_M = fft_size_M
        self.fft_size_N = fft_size_N
        self.patch_size = patch_size
        
        if seed is not None:
            np.random.seed(seed)
        
        # 预生成目标参数
        self.targets_params = []
        for _ in range(num_samples):
            targets = []
            for _ in range(num_targets):
                r = np.random.uniform(r_min, r_max)
                theta = np.random.uniform(theta_min, theta_max)
                targets.append((r, theta))
            self.targets_params.append(targets)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            patch_tensor: (2, patch_size, patch_size), float32
            residual_label: (2,), float32 - [delta_r_norm, delta_theta_norm]
            true_label: (2,), float32 - [r_true, theta_true]
        """
        # 1. 获取真实参数
        targets = self.targets_params[idx]
        r_true, theta_true = targets[0]
        
        # 2. 确定SNR
        if self.fixed_snr is not None:
            snr = self.fixed_snr
        else:
            snr = np.random.uniform(self.snr_range[0], self.snr_range[1])
        
        # 3. 生成回波信号和协方差矩阵
        Y = generate_echo_signal(targets, snr, self.L)
        R = compute_sample_covariance_matrix(Y)
        
        # 4. FFT粗估计
        idx_m, idx_n, beamspace = fft_coarse_estimation(
            R, self.fft_size_M, self.fft_size_N
        )
        r_coarse, theta_coarse = indices_to_coarse_params(
            idx_m, idx_n, self.fft_size_M, self.fft_size_N
        )
        
        # 5. 提取patch
        patch = extract_patch(beamspace, idx_m, idx_n, self.patch_size)
        
        # 6. 归一化patch
        patch_norm = complex_normalize(patch, method='max')
        
        # 7. 转换为PyTorch张量 (2通道实数格式)
        patch_tensor = torch.stack([
            torch.from_numpy(patch_norm.real),
            torch.from_numpy(patch_norm.imag)
        ], dim=0).float()  # (2, patch_size, patch_size)
        
        # 8. 计算残差标签 (归一化)
        delta_r = r_true - r_coarse
        delta_theta = theta_true - theta_coarse
        
        # 归一化: 假设残差范围约为 ±200m, ±10度
        delta_r_norm = delta_r / 200.0  # 归一化到约[-1, 1]
        delta_theta_norm = delta_theta / 10.0
        
        residual_label = torch.tensor([delta_r_norm, delta_theta_norm], dtype=torch.float32)
        true_label = torch.tensor([r_true, theta_true], dtype=torch.float32)
        
        return patch_tensor, residual_label, true_label


def create_beamspace_dataloaders(
    train_size: int = 20000,
    val_size: int = 3000,
    test_size: int = 2000,
    batch_size: int = 64,
    num_workers: int = 4,
    L: int = L_snapshots,
    snr_train_range: Tuple[float, float] = (SNR_TRAIN_MIN, SNR_TRAIN_MAX),
    patch_size: int = 5,
    fft_size_M: int = 64,
    fft_size_N: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建Beamspace数据加载器
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("=" * 60)
    print("创建 Beamspace 数据集 (粗精结合)")
    print("=" * 60)
    print(f"FFT大小: {fft_size_M} x {fft_size_N}")
    print(f"Patch大小: {patch_size} x {patch_size}")
    
    # 训练集
    train_dataset = FDADatasetBeamspace(
        num_samples=train_size,
        snr_range=snr_train_range,
        num_targets=1,
        L=L,
        fixed_snr=None,
        fft_size_M=fft_size_M,
        fft_size_N=fft_size_N,
        patch_size=patch_size,
        seed=42
    )
    
    # 验证集
    val_dataset = FDADatasetBeamspace(
        num_samples=val_size,
        snr_range=snr_train_range,
        num_targets=1,
        L=L,
        fixed_snr=None,
        fft_size_M=fft_size_M,
        fft_size_N=fft_size_N,
        patch_size=patch_size,
        seed=43
    )
    
    # 测试集 (固定10dB SNR)
    test_dataset = FDADatasetBeamspace(
        num_samples=test_size,
        snr_range=(10, 10),
        num_targets=1,
        L=L,
        fixed_snr=10.0,
        fft_size_M=fft_size_M,
        fft_size_N=fft_size_N,
        patch_size=patch_size,
        seed=44
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"训练集: {train_size} 样本, {len(train_loader)} batches")
    print(f"验证集: {val_size} 样本, {len(val_loader)} batches")
    print(f"测试集: {test_size} 样本, {len(test_loader)} batches")
    print("=" * 60)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集
    print("测试 Beamspace 数据集...")
    
    dataset = FDADatasetBeamspace(
        num_samples=10,
        patch_size=5,
        fft_size_M=64,
        fft_size_N=64
    )
    
    patch, residual, true_label = dataset[0]
    
    print(f"\nPatch shape: {patch.shape}")
    print(f"Patch range: [{patch.min():.4f}, {patch.max():.4f}]")
    print(f"Residual label: {residual.numpy()}")
    print(f"True label: {true_label.numpy()}")
    print("\n测试通过!")
