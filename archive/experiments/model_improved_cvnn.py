#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的CVNN模型 - 基于相位感知CNN的成功经验

关键改进：
1. ModReLU使用bias=0（避免正偏置导致的恒等映射）
2. 更小的网络结构（避免过拟合）
3. 输出层使用实值线性层
"""

import torch
import torch.nn as nn
import numpy as np
from config import M, N, MN, r_min, r_max, theta_min, theta_max
from complex_layers import ComplexConv2d, ComplexBatchNorm2d, ComplexLinear, ComplexFlatten

class ModReLU_Fixed(nn.Module):
    """修复后的ModReLU"""
    def __init__(self, num_features, bias_init=0.0):
        super().__init__()
        # 偏置初始化为0或小负值
        self.bias = nn.Parameter(torch.full((num_features,), bias_init))
    
    def forward(self, z):
        # z: complex tensor
        mag = torch.abs(z)
        phase = torch.angle(z)
        
        # ReLU(|z| + b) * exp(j*phase)
        # 当bias=0时，这就是标准的模值ReLU
        bias = self.bias.view(1, -1, 1, 1) if z.dim() == 4 else self.bias.view(1, -1)
        new_mag = torch.relu(mag + bias)
        
        return new_mag * torch.exp(1j * phase)


class CVNN_Improved(nn.Module):
    """
    改进的CVNN模型
    
    关键改进：
    - ModReLU bias=0
    - 更简洁的结构
    - 实值输出层
    """
    
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        
        # Block 1: 1 -> 16
        self.conv1 = ComplexConv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(16)
        self.act1 = ModReLU_Fixed(16, bias_init=0.0)
        self.pool1 = nn.AdaptiveAvgPool2d(50)  # 100 -> 50
        
        # Block 2: 16 -> 32
        self.conv2 = ComplexConv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(32)
        self.act2 = ModReLU_Fixed(32, bias_init=0.0)
        self.pool2 = nn.AdaptiveAvgPool2d(25)  # 50 -> 25
        
        # Block 3: 32 -> 64
        self.conv3 = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm2d(64)
        self.act3 = ModReLU_Fixed(64, bias_init=0.0)
        self.pool3 = nn.AdaptiveAvgPool2d(1)  # Global pool
        
        # 输出层：实值
        self.flatten = ComplexFlatten()
        self.fc = nn.Linear(64 * 2, 2)  # 实部+虚部
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, ComplexConv2d):
                nn.init.kaiming_normal_(m.conv_real.weight, mode='fan_out')
                nn.init.kaiming_normal_(m.conv_imag.weight, mode='fan_out')
    
    def forward(self, x):
        # 转换为复数
        if not x.is_complex():
            if x.shape[1] == 2:
                x = torch.complex(x[:, 0:1], x[:, 1:2])
            else:
                raise ValueError(f"Expected 2 channels, got {x.shape[1]}")
        
        # Block 1
        x = self.act1(self.bn1(self.conv1(x)))
        x = torch.complex(self.pool1(x.real), self.pool1(x.imag))
        
        # Block 2
        x = self.act2(self.bn2(self.conv2(x)))
        x = torch.complex(self.pool2(x.real), self.pool2(x.imag))
        
        # Block 3
        x = self.act3(self.bn3(self.conv3(x)))
        x = torch.complex(self.pool3(x.real), self.pool3(x.imag))
        
        # 展平并转为实数
        x = self.flatten(x)  # (batch, 64) complex
        x_real = torch.cat([x.real, x.imag], dim=1)  # (batch, 128) real
        
        x_real = self.dropout(x_real)
        out = torch.sigmoid(self.fc(x_real))
        
        return out


class CVNN_Improved_Deep(nn.Module):
    """更深的改进CVNN"""
    
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        
        # Block 1: 1 -> 32
        self.conv1a = ComplexConv2d(1, 32, kernel_size=3, padding=1)
        self.bn1a = ComplexBatchNorm2d(32)
        self.act1a = ModReLU_Fixed(32, bias_init=0.0)
        self.conv1b = ComplexConv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = ComplexBatchNorm2d(32)
        self.act1b = ModReLU_Fixed(32, bias_init=0.0)
        self.pool1 = nn.MaxPool2d(2)  # 100 -> 50
        
        # Block 2: 32 -> 64
        self.conv2a = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = ComplexBatchNorm2d(64)
        self.act2a = ModReLU_Fixed(64, bias_init=0.0)
        self.conv2b = ComplexConv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = ComplexBatchNorm2d(64)
        self.act2b = ModReLU_Fixed(64, bias_init=0.0)
        self.pool2 = nn.MaxPool2d(2)  # 50 -> 25
        
        # Block 3: 64 -> 128
        self.conv3a = ComplexConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3a = ComplexBatchNorm2d(128)
        self.act3a = ModReLU_Fixed(128, bias_init=0.0)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        
        # 输出
        self.flatten = ComplexFlatten()
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        if not x.is_complex():
            if x.shape[1] == 2:
                x = torch.complex(x[:, 0:1], x[:, 1:2])
        
        # Block 1
        x = self.act1a(self.bn1a(self.conv1a(x)))
        x = self.act1b(self.bn1b(self.conv1b(x)))
        x = torch.complex(self.pool1(x.real), self.pool1(x.imag))
        
        # Block 2
        x = self.act2a(self.bn2a(self.conv2a(x)))
        x = self.act2b(self.bn2b(self.conv2b(x)))
        x = torch.complex(self.pool2(x.real), self.pool2(x.imag))
        
        # Block 3
        x = self.act3a(self.bn3a(self.conv3a(x)))
        x = torch.complex(self.pool3(x.real), self.pool3(x.imag))
        
        # 输出
        x = self.flatten(x)
        x_real = torch.cat([x.real, x.imag], dim=1)
        x_real = self.dropout(torch.relu(self.fc1(x_real)))
        out = torch.sigmoid(self.fc2(x_real))
        
        return out


def test_improved_cvnn():
    """测试改进的CVNN"""
    
    print("=" * 70)
    print("测试改进的CVNN模型")
    print("=" * 70)
    
    from dataset import create_dataloaders
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 数据
    train_loader, val_loader, _ = create_dataloaders(
        train_size=3000, val_size=500, test_size=200,
        batch_size=32, num_workers=0
    )
    
    # 测试两个模型
    models = [
        ("CVNN_Improved", CVNN_Improved()),
        ("CVNN_Improved_Deep", CVNN_Improved_Deep()),
    ]
    
    for name, model in models:
        print(f"\n{'='*70}")
        print(f"测试: {name}")
        print(f"{'='*70}")
        
        model = model.to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {params:,}")
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
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
    test_improved_cvnn()
