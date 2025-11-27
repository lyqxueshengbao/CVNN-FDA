#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整训练测试 - 使用集成到model.py的CVNN_Improved
"""

import torch
import torch.nn as nn
import numpy as np
import time
from model import CVNN_Improved
from dataset import create_dataloaders
from config import r_min, r_max, theta_min, theta_max


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for R, labels, _ in loader:
        R, labels = R.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(R)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    r_errors, theta_errors = [], []
    
    with torch.no_grad():
        for R, labels, raw_labels in loader:
            R, raw_labels = R.to(device), raw_labels.to(device)
            
            out = model(R)
            r_pred = out[:, 0] * (r_max - r_min) + r_min
            theta_pred = out[:, 1] * (theta_max - theta_min) + theta_min
            
            r_errors.extend((r_pred - raw_labels[:, 0]).abs().cpu().numpy())
            theta_errors.extend((theta_pred - raw_labels[:, 1]).abs().cpu().numpy())
    
    rmse_r = np.sqrt(np.mean(np.array(r_errors)**2))
    rmse_theta = np.sqrt(np.mean(np.array(theta_errors)**2))
    mae_r = np.mean(r_errors)
    mae_theta = np.mean(theta_errors)
    
    return rmse_r, rmse_theta, mae_r, mae_theta


def main():
    print("=" * 70)
    print("CVNN_Improved 完整训练测试")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 配置
    train_size = 10000
    val_size = 2000
    test_size = 1000
    batch_size = 64
    epochs = 50
    
    print(f"\n配置:")
    print(f"  训练集: {train_size}")
    print(f"  验证集: {val_size}")
    print(f"  测试集: {test_size}")
    print(f"  批大小: {batch_size}")
    print(f"  训练轮数: {epochs}")
    
    # 数据
    print("\n创建数据集...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_size=train_size, 
        val_size=val_size, 
        test_size=test_size,
        batch_size=batch_size, 
        num_workers=0
    )
    
    # 模型
    model = CVNN_Improved(dropout_rate=0.3).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"模型: CVNN_Improved, 参数量: {params:,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练
    print(f"\n开始训练 ({epochs} epochs)...")
    print("-" * 70)
    
    best_rmse_r = float('inf')
    best_epoch = 0
    best_state = None
    history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_rmse_r, val_rmse_theta, val_mae_r, val_mae_theta = evaluate(model, val_loader, device)
        
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        marker = ""
        if val_rmse_r < best_rmse_r:
            best_rmse_r = val_rmse_r
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
            # 保存最佳模型
            torch.save(model.state_dict(), 'checkpoints/cvnn_improved_best.pth')
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'rmse_r': val_rmse_r,
            'rmse_theta': val_rmse_theta,
            'lr': lr
        })
        
        print(f"Epoch {epoch+1:2d}/{epochs}: Loss={train_loss:.4f}, "
              f"RMSE_r={val_rmse_r:6.1f}m, RMSE_θ={val_rmse_theta:5.2f}°, "
              f"LR={lr:.2e}, Time={epoch_time:.1f}s{marker}")
    
    total_time = time.time() - start_time
    
    print("-" * 70)
    print(f"训练完成! 用时: {total_time/60:.1f}分钟")
    print(f"最佳验证RMSE_r: {best_rmse_r:.1f}m (Epoch {best_epoch})")
    
    # 测试
    print("\n" + "=" * 70)
    print("测试集评估 (加载最佳模型)")
    print("=" * 70)
    
    # 加载最佳模型
    model.load_state_dict(best_state)
    test_rmse_r, test_rmse_theta, test_mae_r, test_mae_theta = evaluate(model, test_loader, device)
    
    print(f"\n测试集结果:")
    print(f"  RMSE_r: {test_rmse_r:.1f}m (MAE: {test_mae_r:.1f}m)")
    print(f"  RMSE_θ: {test_rmse_theta:.2f}° (MAE: {test_mae_theta:.2f}°)")
    
    # 性能分析
    print("\n" + "=" * 70)
    print("性能分析")
    print("=" * 70)
    
    random_rmse_r = 2000 / np.sqrt(12)  # 均匀分布随机猜测
    random_rmse_theta = 120 / np.sqrt(12)
    
    print(f"随机猜测基准: RMSE_r={random_rmse_r:.0f}m, RMSE_θ={random_rmse_theta:.1f}°")
    print(f"相对随机提升: r={(1-test_rmse_r/random_rmse_r)*100:.1f}%, θ={(1-test_rmse_theta/random_rmse_theta)*100:.1f}%")
    
    print(f"\n目标精度 @ 10dB: RMSE_r < 5m, RMSE_θ < 0.5°")
    print(f"实现精度:        RMSE_r = {test_rmse_r:.1f}m, RMSE_θ = {test_rmse_theta:.2f}°")
    
    if test_rmse_r < 50:
        status_r = "✅ 良好"
    elif test_rmse_r < 100:
        status_r = "⚠️ 一般"
    else:
        status_r = "❌ 需改进"
    
    if test_rmse_theta < 2:
        status_theta = "✅ 良好"
    elif test_rmse_theta < 5:
        status_theta = "⚠️ 一般"
    else:
        status_theta = "❌ 需改进"
    
    print(f"\n状态: 距离{status_r}, 角度{status_theta}")


if __name__ == "__main__":
    main()
