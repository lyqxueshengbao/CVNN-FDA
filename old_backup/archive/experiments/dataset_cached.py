# -*- coding: utf-8 -*-
"""
FDA-MIMO 雷达参数估计 - 缓存数据集模块
Cached Dataset Module - 预生成数据到内存，减少CPU负担

性能优化:
1. 启动时预生成所有数据到内存
2. __getitem__ 只做索引查询，无计算
3. CPU占用从100%降到~20%
4. GPU利用率提升到90%+
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from tqdm import tqdm

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


class FDADatasetCached(Dataset):
    """
    FDA-MIMO 缓存数据集 - 预生成所有数据到内存
    
    优点:
    - CPU占用低 (训练时不再实时计算)
    - 数据加载速度快 (直接内存读取)
    - GPU利用率高 (无等待)
    
    缺点:
    - 启动时间长 (~1-2分钟生成30K样本)
    - 内存占用大 (~4-6GB for 30K samples)
    
    使用场景: 多轮训练，服务器资源充足
    """
    
    def __init__(self, 
                 num_samples: int,
                 snr_range: Tuple[float, float] = (SNR_TRAIN_MIN, SNR_TRAIN_MAX),
                 num_targets: int = 1,
                 L: int = L_snapshots,
                 fixed_snr: Optional[float] = None,
                 seed: Optional[int] = None,
                 verbose: bool = True):
        """
        初始化缓存数据集
        
        Args:
            num_samples: 样本数量
            snr_range: SNR 范围 (min_snr, max_snr) [dB]
            num_targets: 目标数量
            L: 快拍数
            fixed_snr: 固定SNR (用于测试集)
            seed: 随机种子
            verbose: 是否显示生成进度
        """
        self.num_samples = num_samples
        self.snr_range = snr_range
        self.num_targets = num_targets
        self.L = L
        self.fixed_snr = fixed_snr
        self.verbose = verbose
        
        if seed is not None:
            np.random.seed(seed)
        
        # 预生成所有数据
        if verbose:
            print(f"\n预生成 {num_samples} 个样本到内存...")
        
        self._pregenerate_all_data()
        
        if verbose:
            memory_mb = (self.data_R.nbytes + self.data_labels.nbytes + 
                        self.data_labels_raw.nbytes) / 1024 / 1024
            print(f"数据生成完成! 内存占用: {memory_mb:.1f} MB")
    
    def _pregenerate_all_data(self):
        """预生成所有样本的数据"""
        # 预分配内存
        # R_tensor: (N, 2, MN, MN) - 2通道实数格式
        self.data_R = np.zeros((self.num_samples, 2, MN, MN), dtype=np.float32)
        self.data_labels = np.zeros((self.num_samples, 2), dtype=np.float32)
        self.data_labels_raw = np.zeros((self.num_samples, 2), dtype=np.float32)
        
        # 逐个生成样本
        iterator = range(self.num_samples)
        if self.verbose:
            iterator = tqdm(iterator, desc="生成数据")
        
        for idx in iterator:
            # 生成目标参数
            targets = []
            for _ in range(self.num_targets):
                r = np.random.uniform(r_min, r_max)
                theta = np.random.uniform(theta_min, theta_max)
                targets.append((r, theta))
            
            # 确定 SNR
            if self.fixed_snr is not None:
                snr = self.fixed_snr
            else:
                snr = np.random.uniform(self.snr_range[0], self.snr_range[1])
            
            # 生成回波信号
            Y = generate_echo_signal(targets, snr, self.L)
            
            # 计算协方差矩阵
            R = compute_sample_covariance_matrix(Y)
            
            # 归一化
            R_norm = complex_normalize(R)
            
            # 转换为2通道实数: (MN, MN) complex -> (2, MN, MN) float
            self.data_R[idx, 0] = R_norm.real
            self.data_R[idx, 1] = R_norm.imag
            
            # 标签
            r, theta = targets[0]  # 单目标
            r_norm, theta_norm = normalize_labels(r, theta)
            self.data_labels[idx] = [r_norm, theta_norm]
            self.data_labels_raw[idx] = [r, theta]
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本 (纯内存读取,无计算)
        
        Returns:
            R_tensor: shape (2, MN, MN), float32
            label_tensor: shape (2,), float32
            raw_label_tensor: shape (2,), float32
        """
        R_tensor = torch.from_numpy(self.data_R[idx])
        label_tensor = torch.from_numpy(self.data_labels[idx])
        raw_label_tensor = torch.from_numpy(self.data_labels_raw[idx])
        
        return R_tensor, label_tensor, raw_label_tensor


def create_dataloaders_cached(
    train_size: int = 20000,
    val_size: int = 3000,
    test_size: int = 2000,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    L: int = L_snapshots,
    snr_train_range: Tuple[float, float] = (SNR_TRAIN_MIN, SNR_TRAIN_MAX),
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建缓存数据加载器
    
    Args:
        train_size: 训练集大小
        val_size: 验证集大小
        test_size: 测试集大小
        batch_size: 批大小
        num_workers: 数据加载workers (缓存模式建议设为2-4)
        L: 快拍数
        snr_train_range: 训练SNR范围
        verbose: 显示生成进度
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if verbose:
        print("=" * 60)
        print("创建缓存数据集")
        print("=" * 60)
    
    # 训练集
    train_dataset = FDADatasetCached(
        num_samples=train_size,
        snr_range=snr_train_range,
        num_targets=1,
        L=L,
        fixed_snr=None,
        seed=42,
        verbose=verbose
    )
    
    # 验证集
    val_dataset = FDADatasetCached(
        num_samples=val_size,
        snr_range=snr_train_range,
        num_targets=1,
        L=L,
        fixed_snr=None,
        seed=43,
        verbose=verbose
    )
    
    # 测试集 (固定10dB SNR)
    test_dataset = FDADatasetCached(
        num_samples=test_size,
        snr_range=(10, 10),
        num_targets=1,
        L=L,
        fixed_snr=10.0,
        seed=44,
        verbose=verbose
    )
    
    # 创建 DataLoader
    # 缓存模式: num_workers 设置为2-4即可,不需要16
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
    
    if verbose:
        print(f"\n训练集: {len(train_dataset)} 样本, {len(train_loader)} batches")
        print(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} batches")
        print(f"测试集: {len(test_dataset)} 样本, {len(test_loader)} batches")
        print("=" * 60)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试缓存数据集
    print("测试缓存数据集...")
    
    train_loader, val_loader, test_loader = create_dataloaders_cached(
        train_size=1000,
        val_size=200,
        test_size=200,
        batch_size=32,
        num_workers=2,
        verbose=True
    )
    
    print("\n测试数据加载速度...")
    import time
    start = time.time()
    for i, (batch_R, batch_label, batch_raw) in enumerate(train_loader):
        if i >= 10:
            break
    elapsed = time.time() - start
    print(f"加载10个batch用时: {elapsed:.2f}秒")
    print(f"平均速度: {elapsed/10*1000:.1f}ms/batch")
