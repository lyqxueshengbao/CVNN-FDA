# -*- coding: utf-8 -*-
"""
FDA-MIMO 雷达参数估计 - CVNN 网络模型模块
Complex-Valued Neural Network Model for FDA-MIMO Radar Range-Angle Estimation

包含:
1. CVNN_Estimator: 主要的 CVNN 估计器模型
2. CVNN_Estimator_Light: 轻量级版本
"""

import torch
import torch.nn as nn
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
    ComplexToReal,
    ModReLU
)
from config import MN


class CVNN_Estimator(nn.Module):
    """
    CVNN 参数估计器
    
    网络结构:
    1. 输入层: 100x100 复数矩阵 (Channel=1)
    2. 特征提取 (Backbone):
       - ComplexConv2d(32) + ModReLU
       - ComplexConv2d(64) + ModReLU
       - ComplexAvgPool2d(2x2) -> 50x50
       - ComplexConv2d(128) + ModReLU
       - ComplexAvgPool2d(2x2) -> 25x25
    3. 展平: (128, 25, 25) -> 80000
    4. 全连接层:
       - ComplexLinear(1024) + ModReLU
    5. 输出层:
       - ComplexLinear(2) + 取模 -> [r, θ]
    """
    
    def __init__(self, 
                 input_size: int = MN,
                 dropout_rate: float = 0.3,
                 use_batchnorm: bool = True):
        """
        初始化 CVNN 估计器
        
        Args:
            input_size: 输入矩阵大小 (MN x MN)
            dropout_rate: Dropout 概率
            use_batchnorm: 是否使用批归一化
        """
        super(CVNN_Estimator, self).__init__()
        
        self.input_size = input_size
        self.use_batchnorm = use_batchnorm
        
        # ==================== 特征提取层 ====================
        # Block 1: Conv(1->32) + BN + ModReLU
        self.conv1 = ComplexConv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.act1 = ModReLU(num_features=32)
        
        # Block 2: Conv(32->64) + BN + ModReLU
        self.conv2 = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.act2 = ModReLU(num_features=64)
        
        # Pool 1: 100x100 -> 50x50
        self.pool1 = ComplexAvgPool2d(kernel_size=2, stride=2)
        
        # Block 3: Conv(64->128) + BN + ModReLU
        self.conv3 = ComplexConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.act3 = ModReLU(num_features=128)
        
        # Pool 2: 50x50 -> 25x25
        self.pool2 = ComplexAvgPool2d(kernel_size=2, stride=2)
        
        # ==================== 展平层 ====================
        self.flatten = ComplexFlatten()
        
        # 计算展平后的特征维度
        # 经过两次 2x2 池化: 100 -> 50 -> 25
        # 特征维度: 128 * 25 * 25 = 80000
        self.flat_features = 128 * 25 * 25
        
        # ==================== 全连接层 ====================
        # FC1: 80000 -> 1024
        self.fc1 = ComplexLinear(self.flat_features, 1024)
        self.bn_fc1 = ComplexBatchNorm1d(1024) if use_batchnorm else nn.Identity()
        self.act_fc1 = ModReLU(num_features=1024)
        self.dropout1 = ComplexDropout(p=dropout_rate)
        
        # FC2: 1024 -> 256
        self.fc2 = ComplexLinear(1024, 256)
        self.bn_fc2 = ComplexBatchNorm1d(256) if use_batchnorm else nn.Identity()
        self.act_fc2 = ModReLU(num_features=256)
        self.dropout2 = ComplexDropout(p=dropout_rate)
        
        # ==================== 输出层 ====================
        # 输出层: 使用实值线性层 (从复数特征提取实部和虚部拼接后映射到2维输出)
        # 256复数 -> 512实数 (concat real+imag) -> 2
        self.fc_out = nn.Linear(512, 2)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, (ComplexConv2d, ComplexLinear)):
                # Xavier 初始化
                for sub_m in [m.conv_real, m.conv_imag] if hasattr(m, 'conv_real') else [m.linear_real, m.linear_imag]:
                    if hasattr(sub_m, 'weight'):
                        nn.init.xavier_uniform_(sub_m.weight)
                    if hasattr(sub_m, 'bias') and sub_m.bias is not None:
                        nn.init.zeros_(sub_m.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入复数协方差矩阵, shape (batch, 1, 100, 100), dtype=complex64
                或 2通道实数 (batch, 2, 100, 100) for DataParallel
        
        Returns:
            out: 预测的归一化参数, shape (batch, 2), dtype=float32
                 [0]: 归一化距离 r_norm
                 [1]: 归一化角度 θ_norm
        """
        # DataParallel 兼容性: 转换2通道实数为复数
        if not x.is_complex() and x.shape[1] == 2:
            x = torch.complex(x[:, 0:1], x[:, 1:2])
        
        # 特征提取
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        # Pool 1
        x = self.pool1(x)  # 100 -> 50
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        
        # Pool 2
        x = self.pool2(x)  # 50 -> 25
        
        # 展平
        x = self.flatten(x)  # (batch, 128*25*25)
        
        # 全连接层
        # FC1
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.act_fc1(x)
        x = self.dropout1(x)
        
        # FC2
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.act_fc2(x)
        x = self.dropout2(x)
        
        # 输出层: 拼接实部和虚部，然后用实值线性层映射
        x_concat = torch.cat([x.real, x.imag], dim=-1)  # (batch, 512)
        out = torch.sigmoid(self.fc_out(x_concat))  # (batch, 2) real, range [0,1]
        
        return out


class CVNN_Estimator_Light(nn.Module):
    """
    轻量级 CVNN 估计器
    
    参数更少,训练更快,适合快速实验
    """
    
    def __init__(self,
                 input_size: int = MN,
                 dropout_rate: float = 0.2,
                 use_batchnorm: bool = True):
        super(CVNN_Estimator_Light, self).__init__()
        self.use_batchnorm = use_batchnorm

        # 特征提取
        self.features = nn.Sequential()
        
        # Block 1: Conv(1->16) + BN + ModReLU
        self.conv1 = ComplexConv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(16) if use_batchnorm else nn.Identity()
        self.act1 = ModReLU(num_features=16)
        self.pool1 = ComplexAvgPool2d(kernel_size=2, stride=2)  # 100 -> 50
        
        # Block 2: Conv(16->32) + BN + ModReLU
        self.conv2 = ComplexConv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.act2 = ModReLU(num_features=32)
        self.pool2 = ComplexAvgPool2d(kernel_size=2, stride=2)  # 50 -> 25
        
        # Block 3: Conv(32->64) + BN + ModReLU
        self.conv3 = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.act3 = ModReLU(num_features=64)
        self.pool3 = ComplexAvgPool2d(kernel_size=5, stride=5)  # 25 -> 5
        
        # 展平
        self.flatten = ComplexFlatten()
        
        # 特征维度: 64 * 5 * 5 = 1600
        self.flat_features = 64 * 5 * 5
        
        # 全连接层
        self.fc1 = ComplexLinear(self.flat_features, 256)
        self.act_fc1 = ModReLU(num_features=256)
        self.dropout = ComplexDropout(p=dropout_rate)
        
        # 输出层: 使用实值线性层 (从复数特征提取实部和虚部拼接后映射到2维输出)
        # 256复数 -> 512实数 (concat real+imag) -> 2
        self.fc_out = nn.Linear(512, 2)
    
    def forward(self, x: Tensor) -> Tensor:
        # DataParallel 兼容性: 转换2通道实数为复数
        if not x.is_complex() and x.shape[1] == 2:
            x = torch.complex(x[:, 0:1], x[:, 1:2])
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)
        
        # 展平和全连接
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act_fc1(x)
        x = self.dropout(x)
        
        # 输出: 拼接实部和虚部，然后用实值线性层映射
        x_concat = torch.cat([x.real, x.imag], dim=-1)  # (batch, 512)
        out = torch.sigmoid(self.fc_out(x_concat))
        
        return out


class CVNN_Estimator_Deep(nn.Module):
    """
    深度 CVNN 估计器
    
    更多层,更强特征提取能力,适合复杂场景
    """
    
    def __init__(self,
                 input_size: int = MN,
                 dropout_rate: float = 0.4,
                 use_batchnorm: bool = True):
        super(CVNN_Estimator_Deep, self).__init__()
        self.use_batchnorm = use_batchnorm
        
        # Block 1: 1 -> 32
        self.conv1a = ComplexConv2d(1, 32, kernel_size=3, padding=1)
        self.bn1a = ComplexBatchNorm2d(32)
        self.act1a = ModReLU(num_features=32)
        self.conv1b = ComplexConv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = ComplexBatchNorm2d(32)
        self.act1b = ModReLU(num_features=32)
        self.pool1 = ComplexAvgPool2d(kernel_size=2, stride=2)  # 100 -> 50
        
        # Block 2: 32 -> 64
        self.conv2a = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = ComplexBatchNorm2d(64)
        self.act2a = ModReLU(num_features=64)
        self.conv2b = ComplexConv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = ComplexBatchNorm2d(64)
        self.act2b = ModReLU(num_features=64)
        self.pool2 = ComplexAvgPool2d(kernel_size=2, stride=2)  # 50 -> 25
        
        # Block 3: 64 -> 128
        self.conv3a = ComplexConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3a = ComplexBatchNorm2d(128)
        self.act3a = ModReLU(num_features=128)
        self.conv3b = ComplexConv2d(128, 128, kernel_size=3, padding=1)
        self.bn3b = ComplexBatchNorm2d(128)
        self.act3b = ModReLU(num_features=128)
        self.pool3 = ComplexAvgPool2d(kernel_size=5, stride=5)  # 25 -> 5
        
        # 展平
        self.flatten = ComplexFlatten()
        self.flat_features = 128 * 5 * 5  # 3200
        
        # 全连接层
        self.fc1 = ComplexLinear(self.flat_features, 512)
        self.bn_fc1 = ComplexBatchNorm1d(512)
        self.act_fc1 = ModReLU(num_features=512)
        self.dropout1 = ComplexDropout(p=dropout_rate)
        
        self.fc2 = ComplexLinear(512, 128)
        self.bn_fc2 = ComplexBatchNorm1d(128)
        self.act_fc2 = ModReLU(num_features=128)
        self.dropout2 = ComplexDropout(p=dropout_rate)
        
        # 输出层: 使用实值线性层
        # 128复数 -> 256实数 (concat real+imag) -> 2
        self.fc_out = nn.Linear(256, 2)
    
    def forward(self, x: Tensor) -> Tensor:
        # DataParallel 兼容性: 转换2通道实数为复数
        if not x.is_complex() and x.shape[1] == 2:
            x = torch.complex(x[:, 0:1], x[:, 1:2])
        
        # Block 1
        x = self.act1a(self.bn1a(self.conv1a(x)))
        x = self.act1b(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        
        # Block 2
        x = self.act2a(self.bn2a(self.conv2a(x)))
        x = self.act2b(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        
        # Block 3
        x = self.act3a(self.bn3a(self.conv3a(x)))
        x = self.act3b(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        
        # 全连接
        x = self.flatten(x)
        x = self.dropout1(self.act_fc1(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(self.act_fc2(self.bn_fc2(self.fc2(x))))
        
        # 输出层: 拼接实部和虚部，然后用实值线性层映射
        x_concat = torch.cat([x.real, x.imag], dim=-1)  # (batch, 256)
        out = torch.sigmoid(self.fc_out(x_concat))
        
        return out


def get_model(model_name: str = 'standard', **kwargs) -> nn.Module:
    """
    获取模型实例
    
    Args:
        model_name: 模型名称 ('standard', 'light', 'deep', 'pro')
        **kwargs: 传递给模型的额外参数
    
    Returns:
        model: 模型实例
    """
    models = {
        'standard': CVNN_Estimator,
        'light': CVNN_Estimator_Light,
        'deep': CVNN_Estimator_Deep,
        'pro': CVNN_Estimator_Pro
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](**kwargs)


class ComplexResidualBlock(nn.Module):
    """复数残差块"""
    def __init__(self, channels: int, use_batchnorm: bool = True):
        super().__init__()
        self.conv1 = ComplexConv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.act1 = ModReLU(num_features=channels)
        self.conv2 = ComplexConv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.act2 = ModReLU(num_features=channels)
    
    def forward(self, x: Tensor) -> Tensor:
        # DataParallel 兼容性: 转换2通道实数为复数
        if not x.is_complex() and x.shape[1] == 2:
            x = torch.complex(x[:, 0:1], x[:, 1:2])
        
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity  # 残差连接
        out = self.act2(out)
        return out


class CVNN_Estimator_Pro(nn.Module):
    """
    增强版 CVNN 估计器 - 带残差连接的深度网络
    
    专为高精度参数估计设计:
    - 更深的网络结构
    - 残差连接防止梯度消失
    - 更多的特征通道
    - 全局平均池化减少参数
    """
    
    def __init__(self,
                 input_size: int = MN,
                 dropout_rate: float = 0.2,
                 use_batchnorm: bool = True):
        super(CVNN_Estimator_Pro, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        
        # ==================== Stage 1: 初始卷积 ====================
        # 100x100 -> 100x100, 1 -> 64 channels
        self.conv_in = ComplexConv2d(1, 64, kernel_size=7, padding=3)
        self.bn_in = ComplexBatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.act_in = ModReLU(num_features=64)
        
        # ==================== Stage 2: 残差块组 ====================
        # 100x100, 64 channels
        self.res_block1 = ComplexResidualBlock(64, use_batchnorm)
        self.res_block2 = ComplexResidualBlock(64, use_batchnorm)
        
        # 下采样: 100x100 -> 50x50, 64 -> 128 channels
        self.down1 = ComplexConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn_down1 = ComplexBatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.act_down1 = ModReLU(num_features=128)
        
        # 50x50, 128 channels
        self.res_block3 = ComplexResidualBlock(128, use_batchnorm)
        self.res_block4 = ComplexResidualBlock(128, use_batchnorm)
        
        # 下采样: 50x50 -> 25x25, 128 -> 256 channels
        self.down2 = ComplexConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn_down2 = ComplexBatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.act_down2 = ModReLU(num_features=256)
        
        # 25x25, 256 channels
        self.res_block5 = ComplexResidualBlock(256, use_batchnorm)
        self.res_block6 = ComplexResidualBlock(256, use_batchnorm)
        
        # 下采样: 25x25 -> 12x12, 256 -> 512 channels
        self.down3 = ComplexConv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn_down3 = ComplexBatchNorm2d(512) if use_batchnorm else nn.Identity()
        self.act_down3 = ModReLU(num_features=512)
        
        # 12x12, 512 channels (实际是12x12因为100/2/2/2=12.5向下取整)
        self.res_block7 = ComplexResidualBlock(512, use_batchnorm)
        
        # ==================== Stage 3: 全局平均池化 ====================
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ==================== Stage 4: 全连接头 ====================
        self.flatten = ComplexFlatten()
        
        self.fc1 = ComplexLinear(512, 256)
        self.bn_fc1 = ComplexBatchNorm1d(256) if use_batchnorm else nn.Identity()
        self.act_fc1 = ModReLU(num_features=256)
        self.dropout1 = ComplexDropout(p=dropout_rate)
        
        self.fc2 = ComplexLinear(256, 64)
        self.bn_fc2 = ComplexBatchNorm1d(64) if use_batchnorm else nn.Identity()
        self.act_fc2 = ModReLU(num_features=64)
        self.dropout2 = ComplexDropout(p=dropout_rate)
        
        # 输出层: 使用实值线性层
        # 64复数 -> 128实数 (concat real+imag) -> 2
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
        # DataParallel 兼容性修复: 将2通道实数输入转换为复数
        # 输入: (batch, 2, H, W) float -> (batch, 1, H, W) complex
        if not x.is_complex():
            if x.shape[1] == 2:  # [real, imag] 格式
                x = torch.complex(x[:, 0:1], x[:, 1:2])
            else:
                raise TypeError(f"Input must be complex or 2-channel real, got shape={x.shape}")
        
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
        
        # Stage 3: 全局平均池化
        # 分别对实部和虚部进行池化
        x_real = self.global_pool(x.real)
        x_imag = self.global_pool(x.imag)
        x = torch.complex(x_real, x_imag)
        
        # Stage 4: 全连接头
        x = self.flatten(x)
        x = self.dropout1(self.act_fc1(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(self.act_fc2(self.bn_fc2(self.fc2(x))))
        
        # 输出层: 拼接实部和虚部，然后用实值线性层映射
        x_concat = torch.cat([x.real, x.imag], dim=-1)  # (batch, 128)
        out = torch.sigmoid(self.fc_out(x_concat))
        
        return out


def count_parameters(model: nn.Module) -> int:
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 实值CNN模型 (已验证有效)
# ============================================================================

class RealCNN_Estimator(nn.Module):
    """
    实值CNN估计器
    
    使用标准实值CNN处理2通道输入(实部+虚部)
    经验证可以有效学习FDA-MIMO数据
    """
    
    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        
        # 特征提取: 输入 (batch, 2, 100, 100)
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
            nn.MaxPool2d(5),  # 25 -> 5
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class RealCNN_Estimator_Deep(nn.Module):
    """
    深度实值CNN估计器
    
    更深的网络结构，更强的特征提取能力
    """
    
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 2 -> 32
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 100 -> 50
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 50 -> 25
            
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class RealCNN_Estimator_Pro(nn.Module):
    """
    专业级实值CNN估计器
    
    带残差连接的深度网络，适合高精度估计任务
    """
    
    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        
        # 初始卷积
        self.conv_in = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 残差块组 1: 64 channels, 100x100
        self.res1 = self._make_res_block(64)
        self.res2 = self._make_res_block(64)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # 100 -> 50
        
        # 残差块组 2: 128 channels, 50x50
        self.res3 = self._make_res_block(128)
        self.res4 = self._make_res_block(128)
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )  # 50 -> 25
        
        # 残差块组 3: 256 channels, 25x25
        self.res5 = self._make_res_block(256)
        self.res6 = self._make_res_block(256)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
    
    def _make_res_block(self, channels):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        
        # 残差块组 1
        identity = x
        x = self.res1(x) + identity
        x = nn.functional.relu(x, inplace=True)
        identity = x
        x = self.res2(x) + identity
        x = nn.functional.relu(x, inplace=True)
        x = self.down1(x)
        
        # 残差块组 2
        identity = x
        x = self.res3(x) + identity
        x = nn.functional.relu(x, inplace=True)
        identity = x
        x = self.res4(x) + identity
        x = nn.functional.relu(x, inplace=True)
        x = self.down2(x)
        
        # 残差块组 3
        identity = x
        x = self.res5(x) + identity
        x = nn.functional.relu(x, inplace=True)
        identity = x
        x = self.res6(x) + identity
        x = nn.functional.relu(x, inplace=True)
        
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ============================================================================
# 改进的CVNN模型 (已验证有效)
# ============================================================================

class ModReLU_Fixed(nn.Module):
    """修复后的ModReLU - bias初始化为0"""
    def __init__(self, num_features, bias_init=0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.full((num_features,), bias_init))
    
    def forward(self, z):
        mag = torch.abs(z)
        phase = torch.angle(z)
        bias = self.bias.view(1, -1, 1, 1) if z.dim() == 4 else self.bias.view(1, -1)
        new_mag = torch.relu(mag + bias)
        return new_mag * torch.exp(1j * phase)


class CVNN_Improved(nn.Module):
    """
    改进的CVNN模型 (已验证有效)
    
    关键改进：
    - ModReLU bias=0 (避免正偏置导致的恒等映射)
    - 池化操作分离实部虚部
    - 实值输出层
    """
    
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        
        # Block 1: 1 -> 32
        self.conv1a = ComplexConv2d(1, 32, kernel_size=3, padding=1)
        self.bn1a = ComplexBatchNorm2d(32)
        self.act1a = ModReLU_Fixed(32, bias_init=0.0)
        self.conv1b = ComplexConv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = ComplexBatchNorm2d(32)
        self.act1b = ModReLU_Fixed(32, bias_init=0.0)
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2: 32 -> 64
        self.conv2a = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = ComplexBatchNorm2d(64)
        self.act2a = ModReLU_Fixed(64, bias_init=0.0)
        self.conv2b = ComplexConv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = ComplexBatchNorm2d(64)
        self.act2b = ModReLU_Fixed(64, bias_init=0.0)
        self.pool2 = nn.MaxPool2d(2)
        
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


def get_model(model_name: str = 'cvnn_improved', **kwargs) -> nn.Module:
    """
    获取模型实例
    
    Args:
        model_name: 模型名称
            - 'cvnn_improved': 改进的CVNN (推荐，默认)
            - 'standard', 'light', 'deep', 'pro': 旧CVNN模型
            - 'real', 'real_deep', 'real_pro': 实值CNN模型
        **kwargs: 传递给模型的额外参数
    
    Returns:
        model: 模型实例
    """
    models = {
        # 改进的CVNN (推荐)
        'cvnn_improved': CVNN_Improved,
        # 旧CVNN模型
        'standard': CVNN_Estimator,
        'light': CVNN_Estimator_Light,
        'deep': CVNN_Estimator_Deep,
        'pro': CVNN_Estimator_Pro,
        # 实值CNN模型
        'real': RealCNN_Estimator,
        'real_light': RealCNN_Estimator,
        'real_deep': RealCNN_Estimator_Deep,
        'real_pro': RealCNN_Estimator_Pro,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](**kwargs)
