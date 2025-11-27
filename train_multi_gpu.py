#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDA-MIMO CVNN 多GPU训练脚本
支持 DataParallel 多卡训练

使用方法:
    # 使用所有可用GPU
    python train_multi_gpu.py
    
    # 指定GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train_multi_gpu.py
    
    # 自定义参数
    python train_multi_gpu.py --epochs 100 --train_size 50000 --batch_size 256
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import json
from datetime import datetime

# 项目模块
from model import CVNN_Improved, RealCNN_Estimator, count_parameters
from dataset import create_dataloaders
from config import r_min, r_max, theta_min, theta_max


def setup_environment():
    """设置环境"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # CUDA设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # 检测GPU
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    return num_gpus


def create_model(model_name='cvnn', dropout_rate=0.2, num_gpus=1):
    """创建模型并包装为DataParallel"""
    if model_name in ['cvnn', 'cvnn_improved']:
        model = CVNN_Improved(dropout_rate=dropout_rate)
    else:
        model = RealCNN_Estimator(dropout_rate=dropout_rate)
    
    # 多GPU包装
    if num_gpus > 1:
        print(f"使用 DataParallel 包装模型 ({num_gpus} GPUs)")
        model = nn.DataParallel(model)
    
    model = model.cuda()
    return model


def train_epoch(model, train_loader, criterion, optimizer, scaler=None, accumulation_steps=1):
    """训练一个epoch (支持梯度累积)"""
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    
    for batch_idx, (x, y, _) in enumerate(train_loader):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        
        # 混合精度训练 (可选)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y) / accumulation_steps
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            out = model(x)
            loss = criterion(out, y) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
    
    # 处理最后不足accumulation_steps的批次
    if num_batches % accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
        
        # 反归一化
        r_pred = out[:, 0] * (r_max - r_min) + r_min
        theta_pred = out[:, 1] * (theta_max - theta_min) + theta_min
        
        r_errors.extend((r_pred - raw[:, 0]).abs().cpu().numpy())
        theta_errors.extend((theta_pred - raw[:, 1]).abs().cpu().numpy())
    
    rmse_r = np.sqrt(np.mean(np.array(r_errors)**2))
    rmse_theta = np.sqrt(np.mean(np.array(theta_errors)**2))
    mae_r = np.mean(r_errors)
    mae_theta = np.mean(theta_errors)
    
    return {
        'loss': total_loss / len(val_loader),
        'rmse_r': rmse_r,
        'rmse_theta': rmse_theta,
        'mae_r': mae_r,
        'mae_theta': mae_theta
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """保存检查点"""
    # 处理DataParallel
    if isinstance(model, nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def main():
    parser = argparse.ArgumentParser(description='FDA-MIMO CVNN 多GPU训练')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='cvnn', choices=['cvnn', 'real'],
                        help='模型类型')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='每GPU批大小')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='预热轮数')
    
    # 数据参数
    parser.add_argument('--train_size', type=int, default=50000, help='训练集大小')
    parser.add_argument('--val_size', type=int, default=10000, help='验证集大小')
    parser.add_argument('--test_size', type=int, default=5000, help='测试集大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--log_interval', type=int, default=1, help='日志打印间隔')
    parser.add_argument('--save_interval', type=int, default=10, help='模型保存间隔')
    parser.add_argument('--amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # ==================== 环境设置 ====================
    print("=" * 70)
    print("FDA-MIMO CVNN 多GPU训练")
    print("=" * 70)
    
    num_gpus = setup_environment()
    if num_gpus == 0:
        print("错误: 未检测到GPU!")
        sys.exit(1)
    
    # 根据GPU数量调整batch_size
    effective_batch_size = args.batch_size * num_gpus * args.accumulation_steps
    print(f"\n有效批大小: {args.batch_size} × {num_gpus} GPU × {args.accumulation_steps} 累积 = {effective_batch_size}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ==================== 数据加载 ====================
    print(f"\n创建数据集...")
    print(f"  训练集: {args.train_size}")
    print(f"  验证集: {args.val_size}")
    print(f"  测试集: {args.test_size}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=effective_batch_size,
        num_workers=args.num_workers
    )
    
    # ==================== 模型创建 ====================
    print(f"\n创建模型: {args.model}")
    model = create_model(args.model, args.dropout, num_gpus)
    
    # 获取实际模型（处理DataParallel）
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    print(f"  参数量: {count_parameters(actual_model):,}")
    
    # ==================== 优化器和调度器 ====================
    criterion = nn.MSELoss()
    
    # 学习率根据batch_size缩放 (线性缩放规则)
    scaled_lr = args.lr * (effective_batch_size / 64)
    print(f"\n学习率: {args.lr} × ({effective_batch_size}/64) = {scaled_lr:.6f}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器: Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.epochs // 4,  # 第一次重启周期
        T_mult=2,              # 周期倍增因子
        eta_min=1e-6
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.amp:
        print("启用混合精度训练 (AMP)")
    
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
        print(f"  恢复自 Epoch {start_epoch}, 最佳 RMSE_r: {best_rmse_r:.1f}m")
    
    # ==================== 训练循环 ====================
    print(f"\n开始训练 (Epochs: {start_epoch} -> {args.epochs})...")
    print("-" * 70)
    
    history = []
    total_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # 训练 (带梯度累积)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, args.accumulation_steps)
        
        # 验证
        val_metrics = evaluate(model, val_loader)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start_time
        
        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'rmse_r': val_metrics['rmse_r'],
            'rmse_theta': val_metrics['rmse_theta'],
            'lr': current_lr
        })
        
        # 保存最佳模型
        is_best = val_metrics['rmse_r'] < best_rmse_r
        if is_best:
            best_rmse_r = val_metrics['rmse_r']
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                os.path.join(args.save_dir, 'best_model.pth')
            )
        
        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # 打印日志
        if (epoch + 1) % args.log_interval == 0:
            marker = " *" if is_best else ""
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"Loss={train_loss:.4f}/{val_metrics['loss']:.4f}, "
                  f"RMSE_r={val_metrics['rmse_r']:6.1f}m, "
                  f"RMSE_θ={val_metrics['rmse_theta']:5.2f}°, "
                  f"LR={current_lr:.2e}, "
                  f"Time={epoch_time:.1f}s{marker}")
    
    # ==================== 训练完成 ====================
    total_time = time.time() - total_start_time
    
    print("-" * 70)
    print(f"训练完成!")
    print(f"  总用时: {total_time/3600:.2f} 小时")
    print(f"  最佳 RMSE_r: {best_rmse_r:.1f}m")
    
    # 测试集评估
    print(f"\n测试集评估...")
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    actual_model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader)
    print(f"  RMSE_r: {test_metrics['rmse_r']:.1f}m (MAE: {test_metrics['mae_r']:.1f}m)")
    print(f"  RMSE_θ: {test_metrics['rmse_theta']:.2f}° (MAE: {test_metrics['mae_theta']:.2f}°)")
    
    # 保存训练历史
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'num_gpus': num_gpus,
            'total_time_hours': total_time / 3600,
            'best_rmse_r': best_rmse_r,
            'test_metrics': test_metrics,
            'history': history
        }, f, indent=2)
    print(f"\n训练历史已保存: {history_path}")
    
    # 保存最终模型
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, test_metrics,
        os.path.join(args.save_dir, 'final_model.pth')
    )
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
