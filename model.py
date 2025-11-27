# -*- coding: utf-8 -*-
"""
FDA-MIMO 雷达参数估计 - CVNN 网络模型模块
Complex-Valued Neural Network Model for FDA-MIMO Radar Range-Angle Estimation

包含:
1. CVNN_Improved: 改进的CVNN模型 (主要模型，已验证有效)
2. RealCNN_Estimator: 实值CNN对比模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from complex_layers import (
    ComplexConv2d,
    ComplexLinear,
    ComplexBatchNorm2d,
    ComplexBatchNorm1d,
    ComplexAvgPool2d,
    ComplexDropout,
    ComplexFlatten,
)
from config import MN


# ============================================================================
# 核心组件
# ============================================================================

class ModReLU(nn.Module):
    """
    ModReLU 复数激活函数
    
    公式: ModReLU(z) = ReLU(|z| + b) * e^(j*phase(z))
    
    关键: bias初始化为0或小负数，避免正偏置导致ReLU恒为正(无非线性效果)
    """
    def __init__(self, num_features: int, bias_init: float = 0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.full((num_features,), bias_init))
    
    def forward(self, z: Tensor) -> Tensor:
        mag = torch.abs(z)
        phase = torch.angle(z)
        
        # 根据输入维度调整bias形状
        if z.dim() == 4:  # (B, C, H, W)
            bias = self.bias.view(1, -1, 1, 1)
        else:  # (B, C)
            bias = self.bias.view(1, -1)
        
        new_mag = F.relu(mag + bias)
        return new_mag * torch.exp(1j * phase)


# ============================================================================
# 主要模型: CVNN_Improved (推荐使用)
# ============================================================================

class CVNN_Improved(nn.Module):
    """
    改进的CVNN模型 - FDA-MIMO雷达参数估计
    
    经验证有效的网络结构:
    - 3个卷积块 (32 -> 64 -> 128 channels)
    - ModReLU激活 (bias=0)
    - 池化分离处理实部虚部
    - 实值输出层
    
    性能: 10000样本50epochs训练后
    - RMSE_r ≈ 32m
    - RMSE_θ ≈ 1.7°
    """
    
    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        
        # Block 1: 1 -> 32 channels
        self.conv1a = ComplexConv2d(1, 32, kernel_size=3, padding=1)
        self.bn1a = ComplexBatchNorm2d(32)
        self.act1a = ModReLU(32, bias_init=0.0)
        self.conv1b = ComplexConv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = ComplexBatchNorm2d(32)
        self.act1b = ModReLU(32, bias_init=0.0)
        self.pool1 = nn.MaxPool2d(2)  # 100 -> 50
        
        # Block 2: 32 -> 64 channels
        self.conv2a = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = ComplexBatchNorm2d(64)
        self.act2a = ModReLU(64, bias_init=0.0)
        self.conv2b = ComplexConv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = ComplexBatchNorm2d(64)
        self.act2b = ModReLU(64, bias_init=0.0)
        self.pool2 = nn.MaxPool2d(2)  # 50 -> 25
        
        # Block 3: 64 -> 128 channels
        self.conv3a = ComplexConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3a = ComplexBatchNorm2d(128)
        self.act3a = ModReLU(128, bias_init=0.0)
        self.pool3 = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        
        # 实值输出层
        self.flatten = ComplexFlatten()
        self.fc1 = nn.Linear(128 * 2, 64)  # 128 real + 128 imag = 256
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入，支持两种格式:
               - 复数: (batch, 1, 100, 100) complex64
               - 2通道实数: (batch, 2, 100, 100) float32 [real, imag]
        
        Returns:
            out: 归一化预测值 (batch, 2) [r_norm, θ_norm]
        """
        # 输入转换: 2通道实数 -> 复数
        if not x.is_complex():
            if x.shape[1] == 2:
                x = torch.complex(x[:, 0:1], x[:, 1:2])
        
        # Block 1
        x = self.act1a(self.bn1a(self.conv1a(x)))
        x = self.act1b(self.bn1b(self.conv1b(x)))
        # 池化: 分离处理实部虚部 (PyTorch池化不直接支持复数)
        x = torch.complex(self.pool1(x.real), self.pool1(x.imag))
        
        # Block 2
        x = self.act2a(self.bn2a(self.conv2a(x)))
        x = self.act2b(self.bn2b(self.conv2b(x)))
        x = torch.complex(self.pool2(x.real), self.pool2(x.imag))
        
        # Block 3
        x = self.act3a(self.bn3a(self.conv3a(x)))
        x = torch.complex(self.pool3(x.real), self.pool3(x.imag))
        
        # 输出层: 复数 -> 实数
        x = self.flatten(x)  # (batch, 128) complex
        x_real = torch.cat([x.real, x.imag], dim=1)  # (batch, 256) real
        x_real = self.dropout(F.relu(self.fc1(x_real)))
        out = torch.sigmoid(self.fc2(x_real))
        
        return out


# ============================================================================
# 对比模型: RealCNN (作为基线对比)
# ============================================================================

class ComplexResidualBlock(nn.Module):
    """复数残差块"""
    def __init__(self, channels: int, use_batchnorm: bool = True):
        super().__init__()
        self.conv1 = ComplexConv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.act1 = ModReLU(channels, bias_init=0.0)
        self.conv2 = ComplexConv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.act2 = ModReLU(channels, bias_init=0.0)
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity  # 残差连接
        out = self.act2(out)
        return out


class CVNN_Medium(nn.Module):
    """
    中等规模 CVNN - 平衡参数量和性能
    
    结构:
    - 4个卷积块 + 2个残差块
    - 通道数: 32 -> 64 -> 128 -> 256
    - 全局平均池化 + 实值全连接输出
    
    参数量: ~1.5M (vs CVNN_Improved ~300K, CVNN_Pro ~19M)
    """
    
    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        
        # Block 1: 1 -> 32 channels, 100x100
        self.conv1 = ComplexConv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(32)
        self.act1 = ModReLU(32, bias_init=0.0)
        self.res1 = ComplexResidualBlock(32)
        self.pool1 = nn.MaxPool2d(2)  # 100 -> 50
        
        # Block 2: 32 -> 64 channels, 50x50
        self.conv2 = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(64)
        self.act2 = ModReLU(64, bias_init=0.0)
        self.res2 = ComplexResidualBlock(64)
        self.pool2 = nn.MaxPool2d(2)  # 50 -> 25
        
        # Block 3: 64 -> 128 channels, 25x25
        self.conv3 = ComplexConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm2d(128)
        self.act3 = ModReLU(128, bias_init=0.0)
        self.res3 = ComplexResidualBlock(128)
        self.pool3 = nn.MaxPool2d(5)  # 25 -> 5
        
        # Block 4: 128 -> 256 channels, 5x5
        self.conv4 = ComplexConv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = ComplexBatchNorm2d(256)
        self.act4 = ModReLU(256, bias_init=0.0)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 5 -> 1
        
        # 输出层: 实值全连接 (避免ComplexLinear的4倍参数)
        self.flatten = ComplexFlatten()
        self.fc1 = nn.Linear(256 * 2, 128)  # 256 complex -> 512 real -> 128
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: Tensor) -> Tensor:
        # 输入转换: 2通道实数 -> 复数
        if not x.is_complex():
            if x.shape[1] == 2:
                x = torch.complex(x[:, 0:1], x[:, 1:2])
        
        # Block 1
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = torch.complex(self.pool1(x.real), self.pool1(x.imag))
        
        # Block 2
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.res2(x)
        x = torch.complex(self.pool2(x.real), self.pool2(x.imag))
        
        # Block 3
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.res3(x)
        x = torch.complex(self.pool3(x.real), self.pool3(x.imag))
        
        # Block 4
        x = self.act4(self.bn4(self.conv4(x)))
        x = torch.complex(self.global_pool(x.real), self.global_pool(x.imag))
        
        # 输出层
        x = self.flatten(x)
        x_real = torch.cat([x.real, x.imag], dim=1)
        x_real = self.dropout(F.relu(self.fc1(x_real)))
        out = torch.sigmoid(self.fc2(x_real))
        
        return out


class CVNN_Pro(nn.Module):
    """
    增强版 CVNN 估计器 - 带残差连接的深度网络
    
    专为高精度参数估计设计:
    - 更深的网络结构 (7个残差块)
    - 残差连接防止梯度消失
    - 更多的特征通道 (64->128->256->512)
    - 全局平均池化减少参数
    
    参数量: ~6.8M (vs CVNN_Improved ~300K)
    """
    
    def __init__(self, dropout_rate: float = 0.2, use_batchnorm: bool = True):
        super().__init__()
        
        self.use_batchnorm = use_batchnorm
        
        # ==================== Stage 1: 初始卷积 ====================
        # 100x100 -> 100x100, 1 -> 64 channels
        self.conv_in = ComplexConv2d(1, 64, kernel_size=7, padding=3)
        self.bn_in = ComplexBatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.act_in = ModReLU(64, bias_init=0.0)
        
        # ==================== Stage 2: 残差块组 ====================
        # 100x100, 64 channels
        self.res_block1 = ComplexResidualBlock(64, use_batchnorm)
        self.res_block2 = ComplexResidualBlock(64, use_batchnorm)
        
        # 下采样: 100x100 -> 50x50, 64 -> 128 channels
        self.down1 = ComplexConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn_down1 = ComplexBatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.act_down1 = ModReLU(128, bias_init=0.0)
        
        # 50x50, 128 channels
        self.res_block3 = ComplexResidualBlock(128, use_batchnorm)
        self.res_block4 = ComplexResidualBlock(128, use_batchnorm)
        
        # 下采样: 50x50 -> 25x25, 128 -> 256 channels
        self.down2 = ComplexConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn_down2 = ComplexBatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.act_down2 = ModReLU(256, bias_init=0.0)
        
        # 25x25, 256 channels
        self.res_block5 = ComplexResidualBlock(256, use_batchnorm)
        self.res_block6 = ComplexResidualBlock(256, use_batchnorm)
        
        # 下采样: 25x25 -> 13x13, 256 -> 512 channels
        self.down3 = ComplexConv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn_down3 = ComplexBatchNorm2d(512) if use_batchnorm else nn.Identity()
        self.act_down3 = ModReLU(512, bias_init=0.0)
        
        # 13x13, 512 channels
        self.res_block7 = ComplexResidualBlock(512, use_batchnorm)
        
        # ==================== Stage 3: 全局平均池化 ====================
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ==================== Stage 4: 全连接头 ====================
        self.flatten = ComplexFlatten()
        
        self.fc1 = ComplexLinear(512, 256)
        self.bn_fc1 = ComplexBatchNorm1d(256) if use_batchnorm else nn.Identity()
        self.act_fc1 = ModReLU(256, bias_init=0.0)
        self.dropout1 = ComplexDropout(p=dropout_rate)
        
        self.fc2 = ComplexLinear(256, 64)
        self.bn_fc2 = ComplexBatchNorm1d(64) if use_batchnorm else nn.Identity()
        self.act_fc2 = ModReLU(64, bias_init=0.0)
        self.dropout2 = ComplexDropout(p=dropout_rate)
        
        # 输出层: 64复数 -> 128实数 -> 2
        self.fc_out = nn.Linear(128, 2)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, ComplexConv2d):
                nn.init.kaiming_normal_(m.conv_real.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.conv_imag.weight, mode='fan_out', nonlinearity='relu')
                if m.conv_real.bias is not None:
                    nn.init.zeros_(m.conv_real.bias)
                    nn.init.zeros_(m.conv_imag.bias)
            elif isinstance(m, ComplexLinear):
                nn.init.xavier_uniform_(m.linear_real.weight)
                nn.init.xavier_uniform_(m.linear_imag.weight)
                if m.linear_real.bias is not None:
                    nn.init.zeros_(m.linear_real.bias)
                    nn.init.zeros_(m.linear_imag.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        # 输入转换: 2通道实数 -> 复数
        if not x.is_complex():
            if x.shape[1] == 2:
                x = torch.complex(x[:, 0:1], x[:, 1:2])
        
        # Stage 1: 初始卷积
        x = self.act_in(self.bn_in(self.conv_in(x)))
        
        # Stage 2: 残差块 + 下采样
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.act_down1(self.bn_down1(self.down1(x)))
        
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.act_down2(self.bn_down2(self.down2(x)))
        
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.act_down3(self.bn_down3(self.down3(x)))
        
        x = self.res_block7(x)
        
        # Stage 3: 全局平均池化 (分别处理实部虚部)
        x = torch.complex(self.global_pool(x.real), self.global_pool(x.imag))
        
        # Stage 4: 全连接头
        x = self.flatten(x)
        x = self.dropout1(self.act_fc1(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(self.act_fc2(self.bn_fc2(self.fc2(x))))
        
        # 输出层: 拼接实部和虚部
        x_concat = torch.cat([x.real, x.imag], dim=-1)
        out = torch.sigmoid(self.fc_out(x_concat))
        
        return out


class RealCNN_Estimator(nn.Module):
    """
    实值CNN估计器 (作为基线对比)
    
    使用标准实值CNN处理2通道输入(实部+虚部)
    """
    
    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 2 -> 32
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 100 -> 50
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 50 -> 25
            
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# 工厂函数
# ============================================================================

def get_model(model_name: str = 'cvnn', **kwargs) -> nn.Module:
    """
    获取模型实例
    
    Args:
        model_name: 模型名称
            - 'cvnn' / 'cvnn_improved': 改进的CVNN (~300K参数)
            - 'medium': 中等规模CVNN (~1.5M参数, 推荐)
            - 'pro' / 'cvnn_pro': 深度残差CVNN (~19M参数, 大模型)
            - 'real' / 'real_cnn': 实值CNN (基线对比)
        **kwargs: 传递给模型的额外参数
    
    Returns:
        model: 模型实例
    """
    models = {
        'cvnn': CVNN_Improved,
        'cvnn_improved': CVNN_Improved,
        'medium': CVNN_Medium,
        'pro': CVNN_Pro,
        'cvnn_pro': CVNN_Pro,
        'real': RealCNN_Estimator,
        'real_cnn': RealCNN_Estimator,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("模型测试")
    print("=" * 60)
    
    # CVNN_Improved
    model = CVNN_Improved()
    x = torch.randn(4, 2, 100, 100)  # 2通道实数输入
    y = model(x)
    print(f"CVNN_Improved:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {y.shape}")
    print(f"  参数量: {count_parameters(model):,}")
    
    # RealCNN
    model2 = RealCNN_Estimator()
    y2 = model2(x)
    print(f"\nRealCNN_Estimator:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {y2.shape}")
    print(f"  参数量: {count_parameters(model2):,}")
