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
        # FC_out: 256 -> 2 (复数)
        self.fc_out = ComplexLinear(256, 2)
        
        # 复数转实数 (取模)
        self.to_real = ComplexToReal(mode='abs')
        
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
        
        Returns:
            out: 预测的归一化参数, shape (batch, 2), dtype=float32
                 [0]: 归一化距离 r_norm
                 [1]: 归一化角度 θ_norm
        """
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
        
        # 输出层
        x = self.fc_out(x)  # (batch, 2) complex
        
        # 转为实数 (取模)
        out = self.to_real(x)  # (batch, 2) real
        
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
        
        # Block 1: Conv(1->16)
        self.conv1 = ComplexConv2d(1, 16, kernel_size=3, padding=1)
        self.act1 = ModReLU(num_features=16)
        self.pool1 = ComplexAvgPool2d(kernel_size=2, stride=2)  # 100 -> 50
        
        # Block 2: Conv(16->32)
        self.conv2 = ComplexConv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = ModReLU(num_features=32)
        self.pool2 = ComplexAvgPool2d(kernel_size=2, stride=2)  # 50 -> 25
        
        # Block 3: Conv(32->64)
        self.conv3 = ComplexConv2d(32, 64, kernel_size=3, padding=1)
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
        
        self.fc_out = ComplexLinear(256, 2)
        self.to_real = ComplexToReal(mode='abs')
    
    def forward(self, x: Tensor) -> Tensor:
        # Block 1
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)
        
        # 展平和全连接
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act_fc1(x)
        x = self.dropout(x)
        
        # 输出
        x = self.fc_out(x)
        out = self.to_real(x)
        
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
        
        self.fc_out = ComplexLinear(128, 2)
        self.to_real = ComplexToReal(mode='abs')
    
    def forward(self, x: Tensor) -> Tensor:
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
        
        # 输出
        x = self.fc_out(x)
        out = self.to_real(x)
        
        return out


def get_model(model_name: str = 'standard', **kwargs) -> nn.Module:
    """
    获取模型实例
    
    Args:
        model_name: 模型名称 ('standard', 'light', 'deep')
        **kwargs: 传递给模型的额外参数
    
    Returns:
        model: 模型实例
    """
    models = {
        'standard': CVNN_Estimator,
        'light': CVNN_Estimator_Light,
        'deep': CVNN_Estimator_Deep
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("CVNN 模型测试")
    print("=" * 60)
    
    # 创建测试输入
    batch_size = 4
    x = torch.randn(batch_size, 1, MN, MN) + 1j * torch.randn(batch_size, 1, MN, MN)
    x = x.to(torch.complex64)
    
    print(f"\n输入张量形状: {x.shape}, 数据类型: {x.dtype}")
    
    # 测试标准模型
    print("\n1. 标准模型 (CVNN_Estimator):")
    model_std = CVNN_Estimator()
    out_std = model_std(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {out_std.shape}, dtype={out_std.dtype}")
    print(f"   参数量: {count_parameters(model_std):,}")
    
    # 测试轻量模型
    print("\n2. 轻量模型 (CVNN_Estimator_Light):")
    model_light = CVNN_Estimator_Light()
    out_light = model_light(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {out_light.shape}, dtype={out_light.dtype}")
    print(f"   参数量: {count_parameters(model_light):,}")
    
    # 测试深度模型
    print("\n3. 深度模型 (CVNN_Estimator_Deep):")
    model_deep = CVNN_Estimator_Deep()
    out_deep = model_deep(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {out_deep.shape}, dtype={out_deep.dtype}")
    print(f"   参数量: {count_parameters(model_deep):,}")
    
    # 测试模型工厂函数
    print("\n4. 模型工厂函数测试:")
    for name in ['standard', 'light', 'deep']:
        model = get_model(name)
        print(f"   {name}: {count_parameters(model):,} parameters")
    
    # 验证输出范围
    print("\n5. 输出验证:")
    print(f"   输出示例: {out_std[0].detach().numpy()}")
    print(f"   输出范围: [{out_std.min().item():.4f}, {out_std.max().item():.4f}]")
    
    print("\n" + "=" * 60)
    print("模型测试完成!")
    print("=" * 60)
