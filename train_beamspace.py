#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练和验证 ComplexBeamRefineNet (粗精结合方案)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Tuple

from model_beamspace import ComplexBeamRefineNet, ComplexBeamRefineNet_Deep, count_parameters
from dataset_beamspace import create_beamspace_dataloaders


def train_epoch(model: nn.Module, 
                loader: DataLoader, 
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for patch, residual_label, _ in loader:
        patch = patch.to(device)
        residual_label = residual_label.to(device)
        
        optimizer.zero_grad()
        output = model(patch)
        loss = criterion(output, residual_label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model: nn.Module,
            loader: DataLoader,
            criterion: nn.Module,
            device: torch.device,
            delta_r_scale: float = 200.0,
            delta_theta_scale: float = 10.0) -> Tuple[float, float, float]:
    """
    评估模型
    
    Returns:
        loss: MSE损失
        rmse_r: 距离RMSE [m]
        rmse_theta: 角度RMSE [degrees]
    """
    model.eval()
    total_loss = 0
    total_error_r = 0
    total_error_theta = 0
    n_samples = 0
    
    with torch.no_grad():
        for patch, residual_label, true_label in loader:
            patch = patch.to(device)
            residual_label = residual_label.to(device)
            true_label = true_label.to(device)
            
            # 预测残差
            pred_residual = model(patch)
            
            # 计算损失
            loss = criterion(pred_residual, residual_label)
            total_loss += loss.item()
            
            # 反归一化残差
            delta_r_pred = pred_residual[:, 0] * delta_r_scale
            delta_theta_pred = pred_residual[:, 1] * delta_theta_scale
            
            # 反归一化标签
            delta_r_true = residual_label[:, 0] * delta_r_scale
            delta_theta_true = residual_label[:, 1] * delta_theta_scale
            
            # 计算误差
            error_r = torch.abs(delta_r_pred - delta_r_true)
            error_theta = torch.abs(delta_theta_pred - delta_theta_true)
            
            total_error_r += error_r.sum().item()
            total_error_theta += error_theta.sum().item()
            n_samples += patch.size(0)
    
    avg_loss = total_loss / len(loader)
    rmse_r = total_error_r / n_samples
    rmse_theta = total_error_theta / n_samples
    
    return avg_loss, rmse_r, rmse_theta


def main():
    print("=" * 70)
    print("训练 ComplexBeamRefineNet - 粗精结合方案")
    print("=" * 70)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 超参数
    train_size = 20000
    val_size = 3000
    test_size = 2000
    batch_size = 128  # 可以用大batch，因为输入很小
    epochs = 50
    lr = 1e-3
    
    print(f"\n训练配置:")
    print(f"  训练集: {train_size}")
    print(f"  验证集: {val_size}")
    print(f"  测试集: {test_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    
    # 创建数据集
    print("\n" + "=" * 70)
    train_loader, val_loader, test_loader = create_beamspace_dataloaders(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        batch_size=batch_size,
        num_workers=0,  # Windows下建议设为0
        patch_size=5,
        fft_size_M=64,
        fft_size_N=64
    )
    
    # 创建模型
    print("\n创建模型...")
    model = ComplexBeamRefineNet(patch_size=5).to(device)
    print(f"  模型: ComplexBeamRefineNet")
    print(f"  参数量: {count_parameters(model):,}")
    
    # 优化器和损失函数
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # 训练
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_rmse_r, val_rmse_theta = evaluate(
            model, val_loader, criterion, device
        )
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 记录最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_beamspace_model.pth')
            marker = " *"
        else:
            marker = ""
        
        # 打印进度
        lr_current = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d}/{epochs}: "
              f"Train={train_loss:.6f}, Val={val_loss:.6f}, "
              f"RMSE_r={val_rmse_r:6.2f}m, RMSE_θ={val_rmse_theta:5.2f}°, "
              f"LR={lr_current:.6f}{marker}")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"训练完成! 总用时: {total_time:.1f}s")
    print(f"最佳验证loss: {best_val_loss:.6f} (Epoch {best_epoch+1})")
    
    # 测试最佳模型
    print("\n" + "=" * 70)
    print("测试集评估")
    print("=" * 70)
    
    model.load_state_dict(torch.load('best_beamspace_model.pth'))
    test_loss, test_rmse_r, test_rmse_theta = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n测试结果:")
    print(f"  Loss: {test_loss:.6f}")
    print(f"  RMSE_r: {test_rmse_r:.2f} m")
    print(f"  RMSE_θ: {test_rmse_theta:.2f} °")
    
    print("\n" + "=" * 70)
    print("性能分析")
    print("=" * 70)
    print(f"残差估计误差:")
    print(f"  距离残差误差: {test_rmse_r:.2f} m")
    print(f"  角度残差误差: {test_rmse_theta:.2f} °")
    print(f"\n注意: 这是相对于FFT粗估计的修正误差")
    print(f"最终精度取决于: 粗估计误差 + 残差估计误差")
    
    # 计算训练速度
    samples_per_sec = (train_size * epochs) / total_time
    print(f"\n训练效率: {samples_per_sec:.0f} 样本/秒")


if __name__ == "__main__":
    main()
