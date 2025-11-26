# -*- coding: utf-8 -*-
"""
FDA-MIMO 雷达参数估计 - 数据集模块
Dataset Module for FDA-MIMO Radar Range-Angle Estimation

包含:
1. FDADataset 类: 动态生成训练/验证/测试数据
2. 数据加载器创建函数
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional

from config import (
    M, N, MN, L_snapshots, 
    r_min, r_max, theta_min, theta_max,
    SNR_TRAIN_MIN, SNR_TRAIN_MAX,
    BATCH_SIZE, DEVICE
)
from utils import (
    generate_echo_signal,
    compute_sample_covariance_matrix,
    complex_normalize,
    normalize_labels,
    numpy_to_torch_complex
)


class FDADataset(Dataset):
    """
    FDA-MIMO 数据集类
    
    动态生成:
    1. 随机目标参数 (r, θ)
    2. 回波信号 Y
    3. 样本协方差矩阵 R
    4. 归一化处理
    
    Attributes:
        num_samples: 数据集样本数量
        snr_range: SNR 范围 (min, max) [dB]
        num_targets: 目标数量 (1 或 2)
        L: 快拍数
        fixed_snr: 是否使用固定 SNR (用于测试)
    """
    
    def __init__(self, 
                 num_samples: int,
                 snr_range: Tuple[float, float] = (SNR_TRAIN_MIN, SNR_TRAIN_MAX),
                 num_targets: int = 1,
                 L: int = L_snapshots,
                 fixed_snr: Optional[float] = None,
                 seed: Optional[int] = None):
        """
        初始化数据集
        
        Args:
            num_samples: 样本数量
            snr_range: SNR 范围 (min_snr, max_snr) [dB]
            num_targets: 目标数量 (默认为1,简化问题)
            L: 快拍数
            fixed_snr: 固定SNR值 (如果不为None,则使用固定SNR)
            seed: 随机种子 (用于可重复性)
        """
        self.num_samples = num_samples
        self.snr_range = snr_range
        self.num_targets = num_targets
        self.L = L
        self.fixed_snr = fixed_snr
        
        if seed is not None:
            np.random.seed(seed)
        
        # 预生成目标参数 (用于确定性评估)
        self._pregenerate_targets()
    
    def _pregenerate_targets(self):
        """预生成所有样本的目标参数"""
        self.targets_params = []
        
        for _ in range(self.num_samples):
            targets = []
            for _ in range(self.num_targets):
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
            R_tensor: 归一化协方差矩阵, shape (1, MN, MN), complex64
            label_tensor: 归一化标签, shape (2,), float32
            raw_label_tensor: 原始标签, shape (2,), float32
        """
        # 获取目标参数
        targets = self.targets_params[idx]
        
        # 确定 SNR
        if self.fixed_snr is not None:
            snr = self.fixed_snr
        else:
            snr = np.random.uniform(self.snr_range[0], self.snr_range[1])
        
        # 生成回波信号
        Y = generate_echo_signal(targets, snr, self.L)  # (MN, L)
        
        # 计算样本协方差矩阵
        R = compute_sample_covariance_matrix(Y)  # (MN, MN)
        
        # 复数归一化
        R_norm = complex_normalize(R)
        
        # 转换为 PyTorch 张量
        # 添加通道维度: (MN, MN) -> (1, MN, MN)
        R_complex = numpy_to_torch_complex(R_norm).unsqueeze(0)
        
        # DataParallel 兼容性修复: 将复数张量转换为2通道实数张量 [real, imag]
        # shape: (1, MN, MN) complex -> (2, MN, MN) float
        R_tensor = torch.stack([R_complex.real, R_complex.imag], dim=0).squeeze(1)
        
        # 处理标签 (对于单目标,直接使用第一个目标的参数)
        # 对于多目标问题,可以选择主目标或进行其他处理
        r, theta = targets[0]
        
        # 归一化标签
        r_norm, theta_norm = normalize_labels(r, theta)
        label_tensor = torch.tensor([r_norm, theta_norm], dtype=torch.float32)
        
        # 原始标签 (用于评估)
        raw_label_tensor = torch.tensor([r, theta], dtype=torch.float32)
        
        return R_tensor, label_tensor, raw_label_tensor


class FDADatasetMultiTarget(FDADataset):
    """
    多目标 FDA-MIMO 数据集
    
    继承自 FDADataset,但返回所有目标的标签
    """
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本 (多目标版本)
        
        Returns:
            R_tensor: 归一化协方差矩阵, shape (1, MN, MN), complex64
            label_tensor: 归一化标签, shape (num_targets, 2), float32
            raw_label_tensor: 原始标签, shape (num_targets, 2), float32
        """
        targets = self.targets_params[idx]
        
        if self.fixed_snr is not None:
            snr = self.fixed_snr
        else:
            snr = np.random.uniform(self.snr_range[0], self.snr_range[1])
        
        Y = generate_echo_signal(targets, snr, self.L)
        R = compute_sample_covariance_matrix(Y)
        R_norm = complex_normalize(R)
        R_tensor = numpy_to_torch_complex(R_norm).unsqueeze(0)
        
        # 处理所有目标的标签
        labels_norm = []
        labels_raw = []
        for r, theta in targets:
            r_norm, theta_norm = normalize_labels(r, theta)
            labels_norm.append([r_norm, theta_norm])
            labels_raw.append([r, theta])
        
        label_tensor = torch.tensor(labels_norm, dtype=torch.float32)
        raw_label_tensor = torch.tensor(labels_raw, dtype=torch.float32)
        
        return R_tensor, label_tensor, raw_label_tensor


def create_dataloaders(train_size: int,
                       val_size: int,
                       test_size: int,
                       batch_size: int = BATCH_SIZE,
                       num_workers: int = 0,
                       snr_range: Tuple[float, float] = (SNR_TRAIN_MIN, SNR_TRAIN_MAX),
                       test_snr: Optional[float] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试数据加载器
    
    Args:
        train_size: 训练集大小
        val_size: 验证集大小
        test_size: 测试集大小
        batch_size: 批大小
        num_workers: 数据加载线程数
        snr_range: 训练/验证 SNR 范围
        test_snr: 测试固定 SNR (如果不为None)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建数据集
    train_dataset = FDADataset(
        num_samples=train_size,
        snr_range=snr_range,
        num_targets=1,
        seed=42
    )
    
    val_dataset = FDADataset(
        num_samples=val_size,
        snr_range=snr_range,
        num_targets=1,
        seed=123
    )
    
    test_dataset = FDADataset(
        num_samples=test_size,
        snr_range=snr_range,
        num_targets=1,
        fixed_snr=test_snr,
        seed=456
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    return train_loader, val_loader, test_loader


def create_test_loader_with_snr(test_size: int,
                                 snr: float,
                                 batch_size: int = BATCH_SIZE) -> DataLoader:
    """
    创建指定 SNR 的测试数据加载器
    
    Args:
        test_size: 测试集大小
        snr: 固定 SNR 值 [dB]
        batch_size: 批大小
    
    Returns:
        test_loader: 测试数据加载器
    """
    test_dataset = FDADataset(
        num_samples=test_size,
        num_targets=1,
        fixed_snr=snr,
        seed=int(snr * 100 + 1000)  # 不同SNR使用不同种子
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return test_loader


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("FDA-MIMO 数据集测试")
    print("=" * 60)
    
    # 创建小型数据集进行测试
    dataset = FDADataset(num_samples=100, num_targets=1, seed=42)
    
    print(f"\n1. 数据集基本信息:")
    print(f"   样本数量: {len(dataset)}")
    print(f"   MN (虚拟阵元数): {MN}")
    
    # 获取一个样本
    R, label, raw_label = dataset[0]
    
    print(f"\n2. 单个样本信息:")
    print(f"   协方差矩阵 R 形状: {R.shape}")
    print(f"   协方差矩阵 R 数据类型: {R.dtype}")
    print(f"   归一化标签: {label}")
    print(f"   原始标签 (r, θ): ({raw_label[0]:.2f} m, {raw_label[1]:.2f}°)")
    
    # 测试数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        train_size=100,
        val_size=20,
        test_size=20,
        batch_size=16
    )
    
    print(f"\n3. 数据加载器信息:")
    print(f"   训练集批次数: {len(train_loader)}")
    print(f"   验证集批次数: {len(val_loader)}")
    print(f"   测试集批次数: {len(test_loader)}")
    
    # 获取一个批次
    for batch_R, batch_label, batch_raw in train_loader:
        print(f"\n4. 批次数据信息:")
        print(f"   批次 R 形状: {batch_R.shape}")
        print(f"   批次 label 形状: {batch_label.shape}")
        print(f"   批次 raw_label 形状: {batch_raw.shape}")
        break
    
    # 测试指定 SNR 的数据加载器
    test_loader_snr = create_test_loader_with_snr(test_size=50, snr=10.0)
    print(f"\n5. 指定 SNR 测试加载器:")
    print(f"   SNR = 10 dB, 批次数: {len(test_loader_snr)}")
    
    print("\n" + "=" * 60)
    print("数据集测试完成!")
    print("=" * 60)
