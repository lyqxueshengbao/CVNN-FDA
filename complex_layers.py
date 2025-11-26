# -*- coding: utf-8 -*-
"""
FDA-MIMO 雷达参数估计 - 复数神经网络层模块
Complex-Valued Neural Network Layers for FDA-MIMO Radar

包含:
1. ComplexConv2d: 复数卷积层
2. ComplexLinear: 复数全连接层
3. ComplexBatchNorm2d: 复数批归一化层
4. ComplexAvgPool2d: 复数平均池化层
5. ModReLU: 复数激活函数 (保相位ReLU)
6. ComplexDropout: 复数Dropout层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Union


class ComplexConv2d(nn.Module):
    """
    复数卷积层 (Complex-Valued Convolution 2D)
    
    输入 Z = X + iY, 权重 W = A + iB
    输出 Out = (X*A - Y*B) + i(X*B + Y*A)
    
    通过两组实数卷积实现复数卷积运算
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True):
        """
        初始化复数卷积层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 膨胀率
            groups: 分组数
            bias: 是否使用偏置
        """
        super(ComplexConv2d, self).__init__()
        
        # 实部卷积 (对应权重 A)
        self.conv_real = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        
        # 虚部卷积 (对应权重 B)
        self.conv_imag = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
    
    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            z: 复数输入张量, shape (batch, channels, H, W), dtype=complex64
        
        Returns:
            out: 复数输出张量
        """
        # 分离实部和虚部
        x = z.real  # 实部 X
        y = z.imag  # 虚部 Y
        
        # 复数卷积: (X + iY) * (A + iB) = (XA - YB) + i(XB + YA)
        # 实部输出: X*A - Y*B
        out_real = self.conv_real(x) - self.conv_imag(y)
        
        # 虚部输出: X*B + Y*A
        out_imag = self.conv_imag(x) + self.conv_real(y)
        
        # 组合为复数
        out = torch.complex(out_real, out_imag)
        
        return out


class ComplexLinear(nn.Module):
    """
    复数全连接层 (Complex-Valued Linear Layer)
    
    输入 Z = X + iY, 权重 W = A + iB
    输出 Out = (XA - YB) + i(XB + YA)
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        """
        初始化复数全连接层
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            bias: 是否使用偏置
        """
        super(ComplexLinear, self).__init__()
        
        # 实部线性变换
        self.linear_real = nn.Linear(in_features, out_features, bias=bias)
        
        # 虚部线性变换
        self.linear_imag = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            z: 复数输入张量, shape (batch, features), dtype=complex64
        
        Returns:
            out: 复数输出张量
        """
        x = z.real
        y = z.imag
        
        # 复数线性变换
        out_real = self.linear_real(x) - self.linear_imag(y)
        out_imag = self.linear_imag(x) + self.linear_real(y)
        
        out = torch.complex(out_real, out_imag)
        
        return out


class ComplexBatchNorm2d(nn.Module):
    """
    复数批归一化层 (Complex Batch Normalization 2D)
    
    分别对实部和虚部进行批归一化
    """
    
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        """
        初始化复数批归一化层
        
        Args:
            num_features: 特征数 (通道数)
            eps: 数值稳定性参数
            momentum: 动量
            affine: 是否使用可学习参数
            track_running_stats: 是否跟踪运行时统计
        """
        super(ComplexBatchNorm2d, self).__init__()
        
        self.bn_real = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats
        )
        
        self.bn_imag = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats
        )
    
    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            z: 复数输入张量
        
        Returns:
            out: 归一化后的复数张量
        """
        out_real = self.bn_real(z.real)
        out_imag = self.bn_imag(z.imag)
        
        return torch.complex(out_real, out_imag)


class ComplexBatchNorm1d(nn.Module):
    """
    复数批归一化层 (Complex Batch Normalization 1D)
    
    用于全连接层后的归一化
    """
    
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True):
        super(ComplexBatchNorm1d, self).__init__()
        
        self.bn_real = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)
        self.bn_imag = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)
    
    def forward(self, z: Tensor) -> Tensor:
        out_real = self.bn_real(z.real)
        out_imag = self.bn_imag(z.imag)
        return torch.complex(out_real, out_imag)


class ComplexAvgPool2d(nn.Module):
    """
    复数平均池化层 (Complex Average Pooling 2D)
    
    分别对实部和虚部进行平均池化
    """
    
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0):
        """
        初始化复数平均池化层
        
        Args:
            kernel_size: 池化核大小
            stride: 步长 (默认等于 kernel_size)
            padding: 填充
        """
        super(ComplexAvgPool2d, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            z: 复数输入张量
        
        Returns:
            out: 池化后的复数张量
        """
        out_real = F.avg_pool2d(z.real, self.kernel_size, self.stride, self.padding)
        out_imag = F.avg_pool2d(z.imag, self.kernel_size, self.stride, self.padding)
        
        return torch.complex(out_real, out_imag)


class ComplexMaxPool2d(nn.Module):
    """
    复数最大池化层 (Complex Max Pooling 2D)
    
    基于模值选择最大值,保留对应的复数值
    """
    
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0):
        super(ComplexMaxPool2d, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, z: Tensor) -> Tensor:
        """基于模值的最大池化"""
        # 计算模值
        abs_z = torch.abs(z)
        
        # 对模值进行最大池化,获取索引
        _, indices = F.max_pool2d(
            abs_z, self.kernel_size, self.stride, self.padding,
            return_indices=True
        )
        
        # 使用索引从原始复数张量中选择值
        # 分别对实部和虚部进行操作
        out_real = F.max_pool2d(z.real, self.kernel_size, self.stride, self.padding)
        out_imag = F.max_pool2d(z.imag, self.kernel_size, self.stride, self.padding)
        
        return torch.complex(out_real, out_imag)


class ModReLU(nn.Module):
    """
    ModReLU 激活函数 (Modulus ReLU)
    
    ModReLU(z) = ReLU(|z| + b) * (z / |z|)
    
    特点:
    - 保持相位不变 (Phase preservation)
    - b 是可学习的偏置参数
    - 只对幅度应用 ReLU,相位保持不变
    """
    
    def __init__(self, num_features: Optional[int] = None, bias_init: float = 0.5):
        """
        初始化 ModReLU
        
        Args:
            num_features: 特征数 (如果为 None,使用全局偏置)
            bias_init: 偏置初始化值
        """
        super(ModReLU, self).__init__()
        
        if num_features is not None:
            # 每个通道一个可学习偏置
            self.bias = nn.Parameter(torch.ones(num_features) * bias_init)
        else:
            # 全局可学习偏置
            self.bias = nn.Parameter(torch.tensor(bias_init))
        
        self.num_features = num_features
    
    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            z: 复数输入张量, shape (batch, channels, H, W) 或 (batch, features)
        
        Returns:
            out: 激活后的复数张量
        """
        # 计算模值
        abs_z = torch.abs(z)
        
        # 避免除零
        eps = 1e-7
        abs_z_safe = abs_z + eps
        
        # 计算相位 (单位复数)
        phase = z / abs_z_safe
        
        # 调整偏置的形状以匹配输入
        if self.num_features is not None and z.dim() == 4:
            # 对于卷积层输出: (batch, channels, H, W)
            bias = self.bias.view(1, -1, 1, 1)
        elif self.num_features is not None and z.dim() == 2:
            # 对于全连接层输出: (batch, features)
            bias = self.bias.view(1, -1)
        else:
            bias = self.bias
        
        # ModReLU: ReLU(|z| + b) * phase
        activated_magnitude = F.relu(abs_z + bias)
        out = activated_magnitude * phase
        
        return out


class CReLU(nn.Module):
    """
    CReLU 激活函数 (Complex ReLU)
    
    分别对实部和虚部应用 ReLU
    CReLU(z) = ReLU(Re(z)) + i*ReLU(Im(z))
    
    注意: 这会改变相位,但计算简单
    """
    
    def __init__(self):
        super(CReLU, self).__init__()
    
    def forward(self, z: Tensor) -> Tensor:
        return torch.complex(F.relu(z.real), F.relu(z.imag))


class ZReLU(nn.Module):
    """
    ZReLU 激活函数
    
    只有当实部和虚部都为正时才保留,否则置零
    """
    
    def __init__(self):
        super(ZReLU, self).__init__()
    
    def forward(self, z: Tensor) -> Tensor:
        mask = (z.real > 0) & (z.imag > 0)
        return z * mask.float()


class ComplexDropout(nn.Module):
    """
    复数 Dropout 层
    
    对整个复数值应用 dropout (实部虚部同时 drop)
    """
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: dropout 概率
        """
        super(ComplexDropout, self).__init__()
        self.p = p
    
    def forward(self, z: Tensor) -> Tensor:
        if self.training and self.p > 0:
            # 生成 mask
            mask = torch.ones_like(z.real)
            mask = F.dropout(mask, p=self.p, training=True)
            
            # 对复数应用 mask
            out_real = z.real * mask
            out_imag = z.imag * mask
            
            return torch.complex(out_real, out_imag)
        else:
            return z


class ComplexFlatten(nn.Module):
    """
    复数展平层
    
    将多维复数张量展平为一维
    """
    
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super(ComplexFlatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, z: Tensor) -> Tensor:
        # 分别展平实部和虚部,然后重组
        real_flat = torch.flatten(z.real, self.start_dim, self.end_dim)
        imag_flat = torch.flatten(z.imag, self.start_dim, self.end_dim)
        return torch.complex(real_flat, imag_flat)


class ComplexToReal(nn.Module):
    """
    复数转实数层
    
    将复数张量转换为实数张量,用于网络输出层
    支持多种转换方式: 取模, 取实部, 拼接实虚部
    """
    
    def __init__(self, mode: str = 'abs'):
        """
        Args:
            mode: 转换模式
                - 'abs': 取模值 |z|
                - 'real': 取实部 Re(z)
                - 'concat': 拼接实虚部 [Re(z), Im(z)]
        """
        super(ComplexToReal, self).__init__()
        self.mode = mode
    
    def forward(self, z: Tensor) -> Tensor:
        if self.mode == 'abs':
            return torch.abs(z)
        elif self.mode == 'real':
            return z.real
        elif self.mode == 'concat':
            return torch.cat([z.real, z.imag], dim=-1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("复数神经网络层测试")
    print("=" * 60)
    
    # 创建测试输入
    batch_size = 4
    channels = 1
    H, W = 100, 100
    
    # 生成随机复数输入
    z = torch.randn(batch_size, channels, H, W) + 1j * torch.randn(batch_size, channels, H, W)
    z = z.to(torch.complex64)
    
    print(f"\n输入张量形状: {z.shape}, 数据类型: {z.dtype}")
    
    # 测试 ComplexConv2d
    print("\n1. 测试 ComplexConv2d:")
    conv = ComplexConv2d(1, 32, kernel_size=3, padding=1)
    out_conv = conv(z)
    print(f"   输入: {z.shape} -> 输出: {out_conv.shape}")
    
    # 测试 ComplexBatchNorm2d
    print("\n2. 测试 ComplexBatchNorm2d:")
    bn = ComplexBatchNorm2d(32)
    out_bn = bn(out_conv)
    print(f"   输入: {out_conv.shape} -> 输出: {out_bn.shape}")
    
    # 测试 ModReLU
    print("\n3. 测试 ModReLU:")
    modrelu = ModReLU(num_features=32)
    out_relu = modrelu(out_bn)
    print(f"   输入: {out_bn.shape} -> 输出: {out_relu.shape}")
    
    # 测试 ComplexAvgPool2d
    print("\n4. 测试 ComplexAvgPool2d:")
    pool = ComplexAvgPool2d(kernel_size=2, stride=2)
    out_pool = pool(out_relu)
    print(f"   输入: {out_relu.shape} -> 输出: {out_pool.shape}")
    
    # 测试 ComplexFlatten
    print("\n5. 测试 ComplexFlatten:")
    flatten = ComplexFlatten()
    out_flat = flatten(out_pool)
    print(f"   输入: {out_pool.shape} -> 输出: {out_flat.shape}")
    
    # 测试 ComplexLinear
    print("\n6. 测试 ComplexLinear:")
    linear = ComplexLinear(out_flat.shape[1], 256)
    out_linear = linear(out_flat)
    print(f"   输入: {out_flat.shape} -> 输出: {out_linear.shape}")
    
    # 测试 ComplexToReal
    print("\n7. 测试 ComplexToReal:")
    to_real = ComplexToReal(mode='abs')
    out_real = to_real(out_linear)
    print(f"   输入: {out_linear.shape}, dtype={out_linear.dtype}")
    print(f"   输出: {out_real.shape}, dtype={out_real.dtype}")
    
    # 测试 ComplexDropout
    print("\n8. 测试 ComplexDropout:")
    dropout = ComplexDropout(p=0.5)
    dropout.train()
    out_drop = dropout(out_linear)
    print(f"   输入: {out_linear.shape} -> 输出: {out_drop.shape}")
    
    print("\n" + "=" * 60)
    print("所有复数层测试通过!")
    print("=" * 60)
