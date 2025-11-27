#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断模型问题 - 检查数据的可学习性
"""

import torch
import torch.nn as nn
import numpy as np
from config import M, N, MN, r_min, r_max, theta_min, theta_max
from utils import generate_echo_signal, compute_sample_covariance_matrix, complex_normalize
from dataset import create_dataloaders

def test_simple_models():
    """测试简化模型的学习能力"""
    
    print("=" * 70)
    print("测试简化模型的学习能力")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成数据
    print("\n生成数据集...")
    train_loader, val_loader, _ = create_dataloaders(
        train_size=2000, val_size=400, test_size=100,
        batch_size=32, num_workers=0
    )
    
    # 测试1：相位感知CNN
    print("\n--- 测试: 相位感知CNN ---")
    
    class PhaseAwareCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # 输入：2通道 [real, imag]
            # 先计算额外特征：幅度和相位
            self.conv_amp = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.conv_phase = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.conv_shared = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 25
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            # x: (batch, 2, 100, 100) - [real, imag]
            real = x[:, 0:1]
            imag = x[:, 1:2]
            
            # 计算幅度和相位
            amp = torch.sqrt(real**2 + imag**2 + 1e-8)
            phase = torch.atan2(imag, real)
            
            # 分别处理
            feat_amp = self.conv_amp(amp)
            feat_phase = self.conv_phase(phase)
            
            # 合并
            feat = torch.cat([feat_amp, feat_phase], dim=1)
            feat = self.conv_shared(feat)
            out = self.fc(feat)
            return out
    
    model = PhaseAwareCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params:,}")
    
    best_rmse_r = float('inf')
    
    for epoch in range(30):
        # 训练
        model.train()
        train_loss = 0
        for batch in train_loader:
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        r_errors, theta_errors = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y, raw = batch
                x, y = x.to(device), y.to(device)
                raw = raw.to(device)
                
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                
                # 计算RMSE
                r_pred = out[:, 0] * (r_max - r_min) + r_min
                theta_pred = out[:, 1] * (theta_max - theta_min) + theta_min
                
                r_errors.extend((r_pred - raw[:, 0]).abs().cpu().numpy())
                theta_errors.extend((theta_pred - raw[:, 1]).abs().cpu().numpy())
        
        val_loss /= len(val_loader)
        rmse_r = np.sqrt(np.mean(np.array(r_errors)**2))
        rmse_theta = np.sqrt(np.mean(np.array(theta_errors)**2))
        
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        
        marker = " *" if rmse_r < best_rmse_r else ""
        if rmse_r < best_rmse_r:
            best_rmse_r = rmse_r
        
        print(f"Epoch {epoch+1:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"RMSE_r={rmse_r:.1f}m, RMSE_θ={rmse_theta:.2f}°, LR={lr:.6f}{marker}")
    
    print(f"\n最佳RMSE_r: {best_rmse_r:.1f}m")


if __name__ == "__main__":
    test_simple_models()
