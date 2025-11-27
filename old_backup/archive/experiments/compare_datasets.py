#!/usr/bin/env python3
"""
比较动态数据集和缓存数据集的差异
"""

import torch
import numpy as np
from dataset import create_dataloaders
from dataset_cached import create_dataloaders_cached

def compare_datasets():
    """比较两种数据集的数据特征"""
    
    print("=" * 60)
    print("动态数据集 vs 缓存数据集 对比分析")
    print("=" * 60)
    
    # 创建小规模数据集进行对比
    num_samples = 500
    batch_size = 32
    
    # 动态数据集
    print("\n[1] 创建动态数据集...")
    train_dyn, _, _ = create_dataloaders(
        train_size=num_samples,
        val_size=100,
        test_size=100,
        batch_size=batch_size,
        num_workers=0
    )
    
    # 缓存数据集
    print("\n[2] 创建缓存数据集...")
    train_cached, _, _ = create_dataloaders_cached(
        train_size=num_samples,
        val_size=100,
        test_size=100,
        batch_size=batch_size,
        num_workers=0
    )
    
    # 收集数据
    print("\n[3] 收集数据...")
    
    # 动态数据集第一个batch
    batch_dyn = next(iter(train_dyn))
    R_dyn, labels_dyn, raw_dyn = batch_dyn
    
    # 缓存数据集第一个batch
    batch_cached = next(iter(train_cached))
    R_cached, labels_cached, raw_cached = batch_cached
    
    print("\n" + "=" * 60)
    print("数据形状对比")
    print("=" * 60)
    print(f"动态数据集 R_tensor shape: {R_dyn.shape}")
    print(f"缓存数据集 R_tensor shape: {R_cached.shape}")
    print(f"动态数据集 labels shape: {labels_dyn.shape}")
    print(f"缓存数据集 labels shape: {labels_cached.shape}")
    
    print("\n" + "=" * 60)
    print("R_tensor 数据范围对比")
    print("=" * 60)
    
    # 动态数据集
    R_dyn_np = R_dyn.numpy()
    print(f"\n动态数据集:")
    print(f"  Real部分: min={R_dyn_np[:, 0].min():.6f}, max={R_dyn_np[:, 0].max():.6f}")
    print(f"  Imag部分: min={R_dyn_np[:, 1].min():.6f}, max={R_dyn_np[:, 1].max():.6f}")
    print(f"  Real均值={R_dyn_np[:, 0].mean():.6f}, std={R_dyn_np[:, 0].std():.6f}")
    print(f"  Imag均值={R_dyn_np[:, 1].mean():.6f}, std={R_dyn_np[:, 1].std():.6f}")
    
    # 缓存数据集
    R_cached_np = R_cached.numpy()
    print(f"\n缓存数据集:")
    print(f"  Real部分: min={R_cached_np[:, 0].min():.6f}, max={R_cached_np[:, 0].max():.6f}")
    print(f"  Imag部分: min={R_cached_np[:, 1].min():.6f}, max={R_cached_np[:, 1].max():.6f}")
    print(f"  Real均值={R_cached_np[:, 0].mean():.6f}, std={R_cached_np[:, 0].std():.6f}")
    print(f"  Imag均值={R_cached_np[:, 1].mean():.6f}, std={R_cached_np[:, 1].std():.6f}")
    
    print("\n" + "=" * 60)
    print("标签范围对比")
    print("=" * 60)
    
    print(f"\n动态数据集标签 (归一化后):")
    print(f"  r_norm: min={labels_dyn[:, 0].min():.4f}, max={labels_dyn[:, 0].max():.4f}, mean={labels_dyn[:, 0].mean():.4f}")
    print(f"  θ_norm: min={labels_dyn[:, 1].min():.4f}, max={labels_dyn[:, 1].max():.4f}, mean={labels_dyn[:, 1].mean():.4f}")
    
    print(f"\n缓存数据集标签 (归一化后):")
    print(f"  r_norm: min={labels_cached[:, 0].min():.4f}, max={labels_cached[:, 0].max():.4f}, mean={labels_cached[:, 0].mean():.4f}")
    print(f"  θ_norm: min={labels_cached[:, 1].min():.4f}, max={labels_cached[:, 1].max():.4f}, mean={labels_cached[:, 1].mean():.4f}")
    
    print("\n" + "=" * 60)
    print("原始标签范围对比")
    print("=" * 60)
    
    print(f"\n动态数据集原始标签:")
    print(f"  r: min={raw_dyn[:, 0].min():.2f}m, max={raw_dyn[:, 0].max():.2f}m")
    print(f"  θ: min={raw_dyn[:, 1].min():.2f}°, max={raw_dyn[:, 1].max():.2f}°")
    
    print(f"\n缓存数据集原始标签:")
    print(f"  r: min={raw_cached[:, 0].min():.2f}m, max={raw_cached[:, 0].max():.2f}m")
    print(f"  θ: min={raw_cached[:, 1].min():.2f}°, max={raw_cached[:, 1].max():.2f}°")
    
    # 检查数据是否有NaN或Inf
    print("\n" + "=" * 60)
    print("数据健康检查")
    print("=" * 60)
    
    print(f"\n动态数据集:")
    print(f"  R_tensor 含NaN: {torch.isnan(R_dyn).any()}")
    print(f"  R_tensor 含Inf: {torch.isinf(R_dyn).any()}")
    
    print(f"\n缓存数据集:")
    print(f"  R_tensor 含NaN: {torch.isnan(R_cached).any()}")
    print(f"  R_tensor 含Inf: {torch.isinf(R_cached).any()}")
    
    # 检查dtype
    print("\n" + "=" * 60)
    print("数据类型检查")
    print("=" * 60)
    
    print(f"\n动态数据集:")
    print(f"  R_tensor dtype: {R_dyn.dtype}")
    print(f"  labels dtype: {labels_dyn.dtype}")
    
    print(f"\n缓存数据集:")
    print(f"  R_tensor dtype: {R_cached.dtype}")
    print(f"  labels dtype: {labels_cached.dtype}")
    
    # 检查第一个样本的详细值
    print("\n" + "=" * 60)
    print("第一个样本详细对比")
    print("=" * 60)
    
    print(f"\n动态数据集 R[0] 对角线元素 (real):")
    diag_dyn = torch.diag(R_dyn[0, 0])[:5]
    print(f"  {diag_dyn.numpy()}")
    
    print(f"\n缓存数据集 R[0] 对角线元素 (real):")
    diag_cached = torch.diag(R_cached[0, 0])[:5]
    print(f"  {diag_cached.numpy()}")
    
    print("\n" + "=" * 60)
    print("对比完成")
    print("=" * 60)


if __name__ == "__main__":
    compare_datasets()
