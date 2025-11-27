#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证修复后的CVNN模型能否正常学习
"""

import torch
import torch.nn as nn
from model import CVNN_Estimator_Light, get_model
from dataset import create_dataloaders
from config import r_min, r_max, theta_min, theta_max
import time


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for R, labels, _ in loader:
        R, labels = R.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(R)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_rmse_r, total_rmse_theta = 0, 0
    n_samples = 0
    
    with torch.no_grad():
        for R, labels, raw_labels in loader:
            R, labels = R.to(device), labels.to(device)
            raw_labels = raw_labels.to(device)
            
            out = model(R)
            loss = criterion(out, labels)
            total_loss += loss.item()
            
            # 反归一化
            r_pred = out[:, 0] * (r_max - r_min) + r_min
            theta_pred = out[:, 1] * (theta_max - theta_min) + theta_min
            
            rmse_r = torch.sqrt(torch.mean((r_pred - raw_labels[:, 0])**2))
            rmse_theta = torch.sqrt(torch.mean((theta_pred - raw_labels[:, 1])**2))
            
            total_rmse_r += rmse_r.item() * R.size(0)
            total_rmse_theta += rmse_theta.item() * R.size(0)
            n_samples += R.size(0)
    
    return (total_loss / len(loader), 
            total_rmse_r / n_samples, 
            total_rmse_theta / n_samples)


def main():
    print("=" * 70)
    print("修复后的 CVNN 模型验证")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 创建数据集
    train_size, val_size = 5000, 1000
    batch_size = 64
    epochs = 15
    
    print(f"\n创建数据集 (训练={train_size}, 验证={val_size})...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_size=train_size,
        val_size=val_size,
        test_size=500,
        batch_size=batch_size,
        num_workers=0
    )
    
    # 创建模型
    print("\n创建 CVNN_Estimator_Light 模型...")
    model = CVNN_Estimator_Light().to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )
    
    print(f"\n开始训练 ({epochs} epochs)...")
    print("-" * 70)
    
    best_rmse_r = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse_r, val_rmse_theta = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        lr = optimizer.param_groups[0]['lr']
        marker = " *" if val_rmse_r < best_rmse_r else ""
        if val_rmse_r < best_rmse_r:
            best_rmse_r = val_rmse_r
        
        print(f"Epoch {epoch+1:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"RMSE_r={val_rmse_r:6.1f}m, RMSE_theta={val_rmse_theta:5.2f}deg, "
              f"LR={lr:.2e}{marker}")
    
    total_time = time.time() - start_time
    
    # 测试集评估
    test_loss, test_rmse_r, test_rmse_theta = evaluate(model, test_loader, criterion, device)
    
    print("-" * 70)
    print(f"训练完成，耗时: {total_time:.1f}s")
    print(f"最佳验证 RMSE_r: {best_rmse_r:.1f}m")
    print(f"测试集结果: RMSE_r={test_rmse_r:.1f}m, RMSE_theta={test_rmse_theta:.2f}deg")
    
    # 与目标对比
    print("\n" + "=" * 70)
    print("性能评估")
    print("=" * 70)
    print(f"目标 RMSE_r @ 10dB:  < 5m (优秀 < 2m)")
    print(f"达到 RMSE_r:         {test_rmse_r:.1f}m")
    print(f"目标 RMSE_theta:     < 0.5deg (优秀 < 0.1deg)")
    print(f"达到 RMSE_theta:     {test_rmse_theta:.2f}deg")
    
    # 判断模型是否在学习
    random_guess_rmse = 2000 / 12**0.5  # 均匀分布的随机猜测RMSE约577m
    if test_rmse_r < random_guess_rmse * 0.8:
        print(f"\n✓ 模型正在学习！(RMSE_r={test_rmse_r:.1f}m < 随机猜测={random_guess_rmse:.0f}m)")
    else:
        print(f"\n✗ 模型可能未有效学习 (RMSE_r={test_rmse_r:.1f}m vs 随机猜测={random_guess_rmse:.0f}m)")
    
    if test_rmse_r < 100:
        print("\n>>> 非常好的进展！继续训练和调参可以进一步提升性能。")
    elif test_rmse_r < 300:
        print("\n>>> 模型正在学习，需要更多训练或更大数据集。")
    else:
        print("\n>>> 模型性能需要改进，检查数据归一化或模型结构。")


if __name__ == "__main__":
    main()
