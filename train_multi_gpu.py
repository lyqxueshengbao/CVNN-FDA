#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDA-MIMO CVNN 多GPU训练脚本 (预缓存数据版本)
解决CPU瓶颈问题：预先生成所有数据到内存

使用方法:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train_multi_gpu.py --epochs 100
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
from tqdm import tqdm

# 项目模块
from model import CVNN_Improved, RealCNN_Estimator, count_parameters
from config import (
    r_min, r_max, theta_min, theta_max,
    SNR_TRAIN_MIN, SNR_TRAIN_MAX, M, N, MN, L_snapshots
)
from utils import (
    generate_echo_signal, compute_sample_covariance_matrix,
    complex_normalize, normalize_labels
)


def setup_environment():
    """设置环境"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
    
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    return num_gpus


def generate_dataset_cached(num_samples, snr_range, seed=42, desc="生成数据"):
    """
    预生成数据集到内存 (解决CPU瓶颈)
    
    Returns:
        data: (N, 2, MN, MN) float32 tensor [real, imag channels]
        labels: (N, 2) float32 tensor [r_norm, theta_norm]
        raw_labels: (N, 2) float32 tensor [r, theta]
    """
    np.random.seed(seed)
    
    data_list = []
    label_list = []
    raw_label_list = []
    
    for i in tqdm(range(num_samples), desc=desc, ncols=80):
        # 随机生成目标参数
        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(theta_min, theta_max)
        snr = np.random.uniform(snr_range[0], snr_range[1])
        
        # 生成信号和协方差矩阵
        targets = [(r, theta)]
        Y = generate_echo_signal(targets, snr, L_snapshots)
        R = compute_sample_covariance_matrix(Y)
        R_norm = complex_normalize(R)
        
        # 转换为2通道实数格式 [real, imag]
        R_real = np.real(R_norm).astype(np.float32)
        R_imag = np.imag(R_norm).astype(np.float32)
        R_2ch = np.stack([R_real, R_imag], axis=0)  # (2, MN, MN)
        
        # 归一化标签
        r_norm, theta_norm = normalize_labels(r, theta)
        
        data_list.append(R_2ch)
        label_list.append([r_norm, theta_norm])
        raw_label_list.append([r, theta])
    
    data = torch.from_numpy(np.stack(data_list, axis=0))  # (N, 2, MN, MN)
    labels = torch.tensor(label_list, dtype=torch.float32)  # (N, 2)
    raw_labels = torch.tensor(raw_label_list, dtype=torch.float32)  # (N, 2)
    
    return data, labels, raw_labels


def create_cached_dataloaders(train_size, val_size, test_size, batch_size, num_workers=4):
    """创建预缓存的数据加载器"""
    
    print("\n预生成数据集到内存...")
    
    # 生成数据
    train_data, train_labels, train_raw = generate_dataset_cached(
        train_size, (SNR_TRAIN_MIN, SNR_TRAIN_MAX), seed=42, desc="训练集"
    )
    val_data, val_labels, val_raw = generate_dataset_cached(
        val_size, (SNR_TRAIN_MIN, SNR_TRAIN_MAX), seed=123, desc="验证集"
    )
    test_data, test_labels, test_raw = generate_dataset_cached(
        test_size, (SNR_TRAIN_MIN, SNR_TRAIN_MAX), seed=456, desc="测试集"
    )
    
    print(f"\n数据集大小: 训练={train_data.shape}, 验证={val_data.shape}, 测试={test_data.shape}")
    mem_gb = (train_data.nbytes + val_data.nbytes + test_data.nbytes) / 1024**3
    print(f"内存占用: {mem_gb:.2f} GB")
    
    # 创建TensorDataset
    train_dataset = TensorDataset(train_data, train_labels, train_raw)
    val_dataset = TensorDataset(val_data, val_labels, val_raw)
    test_dataset = TensorDataset(test_data, test_labels, test_raw)
    
    # 创建DataLoader (num_workers=0 因为数据已在内存中)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_model(model_name='cvnn', dropout_rate=0.2, num_gpus=1):
    """创建模型"""
    if model_name in ['cvnn', 'cvnn_improved']:
        model = CVNN_Improved(dropout_rate=dropout_rate)
    else:
        model = RealCNN_Estimator(dropout_rate=dropout_rate)
    
    if num_gpus > 1:
        print(f"使用 DataParallel ({num_gpus} GPUs)")
        model = nn.DataParallel(model)
    
    model = model.cuda()
    return model


def train_epoch(model, train_loader, criterion, optimizer, accumulation_steps=1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    
    for batch_idx, (x, y, _) in enumerate(train_loader):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        
        out = model(x)
        loss = criterion(out, y) / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
    
    # 处理剩余的batch
    if num_batches % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, val_loader):
    """评估模型"""
    model.eval()
    r_errors, theta_errors = [], []
    total_loss = 0
    criterion = nn.MSELoss()
    
    for x, y, raw in val_loader:
        x, y, raw = x.cuda(non_blocking=True), y.cuda(non_blocking=True), raw.cuda(non_blocking=True)
        
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item()
        
        r_pred = out[:, 0] * (r_max - r_min) + r_min
        theta_pred = out[:, 1] * (theta_max - theta_min) + theta_min
        
        r_errors.extend((r_pred - raw[:, 0]).abs().cpu().numpy())
        theta_errors.extend((theta_pred - raw[:, 1]).abs().cpu().numpy())
    
    return {
        'loss': total_loss / len(val_loader),
        'rmse_r': np.sqrt(np.mean(np.array(r_errors)**2)),
        'rmse_theta': np.sqrt(np.mean(np.array(theta_errors)**2)),
        'mae_r': np.mean(r_errors),
        'mae_theta': np.mean(theta_errors)
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """保存检查点"""
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def main():
    parser = argparse.ArgumentParser(description='FDA-MIMO CVNN 多GPU训练 (预缓存版)')
    
    parser.add_argument('--model', type=str, default='cvnn', choices=['cvnn', 'real'])
    parser.add_argument('--dropout', type=float, default=0.2)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64, help='每GPU批大小')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='梯度累积')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    parser.add_argument('--train_size', type=int, default=50000)
    parser.add_argument('--val_size', type=int, default=10000)
    parser.add_argument('--test_size', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # ==================== 环境 ====================
    print("=" * 70)
    print("FDA-MIMO CVNN 多GPU训练 (预缓存版)")
    print("=" * 70)
    
    num_gpus = setup_environment()
    if num_gpus == 0:
        print("错误: 未检测到GPU!")
        sys.exit(1)
    
    effective_batch_size = args.batch_size * num_gpus * args.accumulation_steps
    print(f"\n有效批大小: {args.batch_size} × {num_gpus} GPU × {args.accumulation_steps} 累积 = {effective_batch_size}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ==================== 数据 (预缓存) ====================
    train_loader, val_loader, test_loader = create_cached_dataloaders(
        args.train_size, args.val_size, args.test_size,
        batch_size=args.batch_size * num_gpus,
        num_workers=args.num_workers
    )
    
    # ==================== 模型 ====================
    print(f"\n创建模型: {args.model}")
    model = create_model(args.model, args.dropout, num_gpus)
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    print(f"  参数量: {count_parameters(actual_model):,}")
    
    # ==================== 优化器 ====================
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, args.epochs // 4), T_mult=2, eta_min=1e-6)
    
    # ==================== 恢复训练 ====================
    start_epoch = 0
    best_rmse_r = float('inf')
    
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume)
        actual_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_rmse_r = checkpoint['metrics'].get('rmse_r', float('inf'))
    
    # ==================== 训练 ====================
    print(f"\n开始训练 (Epochs: {start_epoch} -> {args.epochs})...")
    print("-" * 70)
    
    history = []
    total_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.accumulation_steps)
        val_metrics = evaluate(model, val_loader)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'rmse_r': val_metrics['rmse_r'],
            'rmse_theta': val_metrics['rmse_theta'],
            'lr': current_lr
        })
        
        is_best = val_metrics['rmse_r'] < best_rmse_r
        if is_best:
            best_rmse_r = val_metrics['rmse_r']
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics,
                          os.path.join(args.save_dir, 'best_model.pth'))
        
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics,
                          os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        if (epoch + 1) % args.log_interval == 0:
            marker = " *" if is_best else ""
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"Loss={train_loss:.4f}, "
                  f"RMSE_r={val_metrics['rmse_r']:6.1f}m, "
                  f"RMSE_θ={val_metrics['rmse_theta']:5.2f}°, "
                  f"LR={current_lr:.2e}, "
                  f"Time={epoch_time:.1f}s{marker}")
    
    # ==================== 完成 ====================
    total_time = time.time() - total_start_time
    
    print("-" * 70)
    print(f"训练完成! 总用时: {total_time/3600:.2f} 小时, 最佳 RMSE_r: {best_rmse_r:.1f}m")
    
    # 测试
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    actual_model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader)
    
    print(f"\n测试集: RMSE_r={test_metrics['rmse_r']:.1f}m, RMSE_θ={test_metrics['rmse_theta']:.2f}°")
    
    # 保存历史
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump({'args': vars(args), 'num_gpus': num_gpus, 'best_rmse_r': best_rmse_r,
                   'test_metrics': test_metrics, 'history': history}, f, indent=2)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
