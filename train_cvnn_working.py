#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用验证过能学习的TestCVNN进行更长时间训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from complex_layers import (
    ComplexConv2d, ComplexLinear, ComplexBatchNorm2d, 
    ComplexAvgPool2d, ComplexFlatten
)
from dataset import create_dataloaders
from config import r_min, r_max, theta_min, theta_max
import time


class ModReLU_Fixed(nn.Module):
    """修复版ModReLU"""
    def __init__(self, num_features: Optional[int] = None, bias_init: float = -0.1):
        super().__init__()
        if num_features is not None:
            self.bias = nn.Parameter(torch.ones(num_features) * bias_init)
        else:
            self.bias = nn.Parameter(torch.tensor(bias_init))
        self.num_features = num_features
    
    def forward(self, z: Tensor) -> Tensor:
        abs_z = torch.abs(z)
        eps = 1e-7
        phase = z / (abs_z + eps)
        
        if self.num_features is not None and z.dim() == 4:
            bias = self.bias.view(1, -1, 1, 1)
        elif self.num_features is not None and z.dim() == 2:
            bias = self.bias.view(1, -1)
        else:
            bias = self.bias
        
        activated_magnitude = F.relu(abs_z + bias)
        return activated_magnitude * phase


class CVNN_Working(nn.Module):
    """经过验证的可工作CVNN模型"""
    
    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        
        # 卷积层 + BN + 激活
        self.conv1 = ComplexConv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(16)
        self.act1 = ModReLU_Fixed(16, bias_init=-0.1)
        self.pool1 = ComplexAvgPool2d(2)  # 100 -> 50
        
        self.conv2 = ComplexConv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(32)
        self.act2 = ModReLU_Fixed(32, bias_init=-0.1)
        self.pool2 = ComplexAvgPool2d(2)  # 50 -> 25
        
        self.conv3 = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm2d(64)
        self.act3 = ModReLU_Fixed(64, bias_init=-0.1)
        self.pool3 = ComplexAvgPool2d(5)  # 25 -> 5
        
        # 全连接层
        self.flatten = ComplexFlatten()
        self.fc1 = ComplexLinear(64 * 5 * 5, 128)
        self.bn_fc = ComplexBatchNorm2d(128)  # 注意：需要1d版本
        self.act_fc = ModReLU_Fixed(128, bias_init=-0.1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 输出层: 使用实值线性层
        self.fc_out = nn.Linear(256, 2)  # 128 real + 128 imag = 256
    
    def forward(self, x: Tensor) -> Tensor:
        # 输入转换
        if not x.is_complex() and x.shape[1] == 2:
            x = torch.complex(x[:, 0:1], x[:, 1:2])
        
        # 卷积块
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # 全连接
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act_fc(x)
        
        # 输出: 拼接实部虚部，加dropout
        x_cat = torch.cat([x.real, x.imag], dim=-1)
        x_cat = self.dropout(x_cat)
        out = torch.sigmoid(self.fc_out(x_cat))
        
        return out


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for R, labels, _ in loader:
        R, labels = R.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(R)
        loss = criterion(out, labels)
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    print("CVNN_Working 长时间训练测试")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 更大的数据集
    train_size, val_size = 10000, 2000
    batch_size = 64
    epochs = 30
    
    print(f"\n创建数据集 (训练={train_size}, 验证={val_size})...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_size=train_size,
        val_size=val_size,
        test_size=1000,
        batch_size=batch_size,
        num_workers=0
    )
    
    print("\n创建 CVNN_Working 模型...")
    model = CVNN_Working(dropout_rate=0.3).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\n开始训练 ({epochs} epochs)...")
    print("-" * 70)
    
    best_rmse_r = float('inf')
    best_state = None
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse_r, val_rmse_theta = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        marker = ""
        if val_rmse_r < best_rmse_r:
            best_rmse_r = val_rmse_r
            best_state = model.state_dict().copy()
            marker = " *"
        
        print(f"Epoch {epoch+1:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"RMSE_r={val_rmse_r:6.1f}m, RMSE_theta={val_rmse_theta:5.2f}deg, "
              f"LR={lr:.2e}{marker}")
    
    total_time = time.time() - start_time
    
    # 加载最佳模型
    model.load_state_dict(best_state)
    
    # 测试集评估
    test_loss, test_rmse_r, test_rmse_theta = evaluate(model, test_loader, criterion, device)
    
    print("-" * 70)
    print(f"训练完成，耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"最佳验证 RMSE_r: {best_rmse_r:.1f}m")
    print(f"测试集结果: RMSE_r={test_rmse_r:.1f}m, RMSE_theta={test_rmse_theta:.2f}deg")
    
    # 判断
    random_rmse = 2000 / 12**0.5  # ~577m
    print(f"\n随机猜测 RMSE_r: ~{random_rmse:.0f}m")
    if test_rmse_r < random_rmse * 0.6:
        print(f"✓ 模型学习有效！比随机猜测好 {(1 - test_rmse_r/random_rmse)*100:.1f}%")
    elif test_rmse_r < random_rmse * 0.8:
        print(f"△ 模型在学习，但效果有限")
    else:
        print(f"✗ 模型未有效学习")


if __name__ == "__main__":
    main()
