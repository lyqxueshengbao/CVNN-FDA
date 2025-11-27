#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同激活函数对CVNN学习能力的影响
找到能让CVNN正常工作的配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import numpy as np

# 导入基础复数层
from complex_layers import (
    ComplexConv2d, ComplexLinear, ComplexBatchNorm2d, 
    ComplexAvgPool2d, ComplexFlatten
)
from dataset import create_dataloaders
from config import r_min, r_max, theta_min, theta_max


# ============================================================================
# 修复版激活函数
# ============================================================================

class ModReLU_Fixed(nn.Module):
    """
    修复版 ModReLU
    
    关键修复: 偏置初始化为小负值，这样ReLU才会起作用
    
    ModReLU(z) = ReLU(|z| + b) * (z / |z|)
    当 b < 0 时，小幅度的复数会被置零，实现非线性
    """
    
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
        
        # 关键: bias为负值时，小幅度复数会被ReLU置零
        activated_magnitude = F.relu(abs_z + bias)
        return activated_magnitude * phase


class CReLU(nn.Module):
    """分别对实部虚部应用ReLU"""
    def forward(self, z: Tensor) -> Tensor:
        return torch.complex(F.relu(z.real), F.relu(z.imag))


class ComplexLeakyReLU(nn.Module):
    """分别对实部虚部应用LeakyReLU - 避免死神经元"""
    def __init__(self, negative_slope: float = 0.1):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, z: Tensor) -> Tensor:
        return torch.complex(
            F.leaky_relu(z.real, self.negative_slope),
            F.leaky_relu(z.imag, self.negative_slope)
        )


class ComplexTanh(nn.Module):
    """分别对实部虚部应用Tanh - 有界激活"""
    def forward(self, z: Tensor) -> Tensor:
        return torch.complex(torch.tanh(z.real), torch.tanh(z.imag))


# ============================================================================
# 简化的测试用CVNN
# ============================================================================

class TestCVNN(nn.Module):
    """用于测试不同激活函数的简化CVNN"""
    
    def __init__(self, activation_type: str = 'modrelu_fixed'):
        super().__init__()
        
        # 选择激活函数
        self.act_type = activation_type
        
        # 卷积层
        self.conv1 = ComplexConv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(16)
        self.pool1 = ComplexAvgPool2d(2)  # 100 -> 50
        
        self.conv2 = ComplexConv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(32)
        self.pool2 = ComplexAvgPool2d(2)  # 50 -> 25
        
        self.conv3 = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm2d(64)
        self.pool3 = ComplexAvgPool2d(5)  # 25 -> 5
        
        # 创建激活函数
        self.act1 = self._make_activation(16)
        self.act2 = self._make_activation(32)
        self.act3 = self._make_activation(64)
        
        # 全连接层
        self.flatten = ComplexFlatten()
        self.fc1 = ComplexLinear(64 * 5 * 5, 128)
        self.act_fc = self._make_activation(128)
        
        # 输出层: 使用实值线性层
        self.fc_out = nn.Linear(256, 2)  # 128 real + 128 imag = 256
    
    def _make_activation(self, num_features):
        if self.act_type == 'modrelu_fixed':
            return ModReLU_Fixed(num_features, bias_init=-0.1)
        elif self.act_type == 'modrelu_zero':
            return ModReLU_Fixed(num_features, bias_init=0.0)
        elif self.act_type == 'crelu':
            return CReLU()
        elif self.act_type == 'leaky':
            return ComplexLeakyReLU(0.1)
        elif self.act_type == 'tanh':
            return ComplexTanh()
        else:
            raise ValueError(f"Unknown activation: {self.act_type}")
    
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
        x = self.act_fc(self.fc1(x))
        
        # 输出: 拼接实部虚部
        x_cat = torch.cat([x.real, x.imag], dim=-1)
        out = torch.sigmoid(self.fc_out(x_cat))
        
        return out


def train_and_evaluate(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    """训练并评估模型"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    
    train_losses = []
    val_metrics = []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        epoch_loss = 0
        for R, labels, _ in train_loader:
            R, labels = R.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(R)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        total_rmse_r = 0
        n_samples = 0
        with torch.no_grad():
            for R, labels, raw_labels in val_loader:
                R, raw_labels = R.to(device), raw_labels.to(device)
                out = model(R)
                r_pred = out[:, 0] * (r_max - r_min) + r_min
                rmse_r = torch.sqrt(torch.mean((r_pred - raw_labels[:, 0])**2))
                total_rmse_r += rmse_r.item() * R.size(0)
                n_samples += R.size(0)
        
        rmse_r = total_rmse_r / n_samples
        val_metrics.append(rmse_r)
        
        print(f"  Epoch {epoch+1:2d}: Loss={train_loss:.4f}, RMSE_r={rmse_r:.1f}m")
    
    return train_losses, val_metrics


def main():
    print("=" * 70)
    print("CVNN 激活函数对比测试")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 创建数据
    print("\n创建数据集...")
    train_loader, val_loader, _ = create_dataloaders(
        train_size=3000,
        val_size=500,
        test_size=100,
        batch_size=64,
        num_workers=0
    )
    
    # 测试不同激活函数
    activations = [
        ('modrelu_fixed', 'ModReLU (bias=-0.1)'),
        ('modrelu_zero', 'ModReLU (bias=0)'),
        ('crelu', 'CReLU'),
        ('leaky', 'Complex LeakyReLU'),
        ('tanh', 'Complex Tanh'),
    ]
    
    results = {}
    
    for act_type, act_name in activations:
        print(f"\n{'='*60}")
        print(f"测试: {act_name}")
        print('='*60)
        
        model = TestCVNN(activation_type=act_type)
        params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {params:,}")
        
        train_losses, val_metrics = train_and_evaluate(
            model, train_loader, val_loader, device, epochs=10, lr=1e-3
        )
        
        results[act_name] = {
            'final_loss': train_losses[-1],
            'final_rmse': val_metrics[-1],
            'best_rmse': min(val_metrics),
            'loss_improved': train_losses[-1] < train_losses[0],
            'rmse_improved': val_metrics[-1] < val_metrics[0],
        }
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    print(f"{'激活函数':<25} {'最终Loss':<12} {'最终RMSE_r':<12} {'最佳RMSE_r':<12} {'学习?'}")
    print("-" * 70)
    
    for name, res in results.items():
        learning = "Y" if res['rmse_improved'] else "N"
        print(f"{name:<25} {res['final_loss']:<12.4f} {res['final_rmse']:<12.1f} {res['best_rmse']:<12.1f} {learning}")
    
    # 找出最佳配置
    best_name = min(results.keys(), key=lambda k: results[k]['best_rmse'])
    print(f"\n最佳配置: {best_name} (RMSE_r = {results[best_name]['best_rmse']:.1f}m)")


if __name__ == "__main__":
    main()
