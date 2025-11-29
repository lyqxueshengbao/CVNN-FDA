"""
FDA-CVNN 网络模型
端到端回归：输入协方差矩阵，输出距离和角度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_complex import (
    ComplexConv2d, 
    ComplexBatchNorm2d, 
    ModReLU, 
    ComplexAvgPool2d,
    ComplexAdaptiveAvgPool2d
)
import config as cfg


# ==========================================
# 复数注意力模块
# ==========================================
class ComplexSEBlock(nn.Module):
    """
    复数 Squeeze-and-Excitation (SE) 通道注意力
    
    核心思想：
    1. Squeeze: 全局平均池化压缩空间维度
    2. Excitation: 两层 FC 学习通道间关系
    3. Scale: 用学到的权重重新加权各通道
    
    对于复数：使用模值来计算注意力权重，然后同时缩放实部和虚部
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = ComplexAdaptiveAvgPool2d(1)
        
        # Excitation: 两层 FC (作用于模值)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        x: [B, 2, C, H, W]
        """
        b, _, c, h, w = x.shape
        
        # Squeeze: 全局平均池化 -> [B, 2, C, 1, 1]
        y = self.avg_pool(x)
        
        # 计算模值作为注意力输入 -> [B, C]
        real = y[:, 0, :, 0, 0]  # [B, C]
        imag = y[:, 1, :, 0, 0]  # [B, C]
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        
        # Excitation: 学习通道权重 -> [B, C]
        attn = self.fc(mag)
        
        # Scale: 重新加权 -> [B, 1, C, 1, 1]
        attn = attn.view(b, 1, c, 1, 1)
        
        return x * attn


class ComplexCBAM(nn.Module):
    """
    复数 CBAM (Convolutional Block Attention Module)
    = 通道注意力 + 空间注意力
    
    在低 SNR 下，空间注意力可以聚焦于协方差矩阵中的关键区域
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        
        # 通道注意力 (SE)
        self.channel_attn = ComplexSEBlock(channels, reduction)
        
        # 空间注意力
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        x: [B, 2, C, H, W]
        """
        # 1. 通道注意力
        x = self.channel_attn(x)
        
        # 2. 空间注意力
        # 沿通道维度取平均和最大的模值
        real = x[:, 0]  # [B, C, H, W]
        imag = x[:, 1]
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)  # [B, C, H, W]
        
        # 通道维度的平均和最大
        avg_mag = mag.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        max_mag = mag.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
        
        # 拼接并生成空间注意力图
        spatial_input = torch.cat([avg_mag, max_mag], dim=1)  # [B, 2, H, W]
        spatial_attn = self.spatial_conv(spatial_input)  # [B, 1, H, W]
        spatial_attn = spatial_attn.unsqueeze(1)  # [B, 1, 1, H, W]
        
        return x * spatial_attn


class FDA_CVNN(nn.Module):
    """
    FDA-MIMO 复数卷积神经网络
    
    输入: [Batch, 2, 100, 100] - 协方差矩阵 (实部通道, 虚部通道)
    输出: [Batch, 2] - 归一化的 (距离, 角度)
    
    架构特点:
    1. 使用复数卷积保持相位信息
    2. ModReLU激活函数 (负偏置创造非线性)
    3. 平均池化 (不破坏相位)
    """
    def __init__(self):
        super().__init__()
        
        # 输入: [B, 2, 1, 100, 100] -> 需要调整为 [B, 2, 1, H, W]
        # 通道数翻倍: 32 -> 64 -> 128，增强特征提取能力
        
        # Block 1: 100 -> 50
        self.conv1 = ComplexConv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(32)
        self.act1 = ModReLU(32, bias_init=-0.5)
        self.pool1 = ComplexAvgPool2d(2)
        
        # Block 2: 50 -> 25
        self.conv2 = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(64)
        self.act2 = ModReLU(64, bias_init=-0.5)
        self.pool2 = ComplexAvgPool2d(2)
        
        # Block 3: 25 -> 5
        self.conv3 = ComplexConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm2d(128)
        self.act3 = ModReLU(128, bias_init=-0.5)
        self.pool3 = ComplexAvgPool2d(5)
        
        # 全连接层
        # 特征图大小: 5x5, 通道128, 实部+虚部
        self.fc_in_dim = 128 * 5 * 5 * 2  # 6400
        
        self.fc1 = nn.Linear(self.fc_in_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # 输出 r 和 theta
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        x: [B, 2, 100, 100] - 实部和虚部
        """
        # 调整维度: [B, 2, H, W] -> [B, 2, 1, H, W]
        x = x.unsqueeze(2)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)  # [B, 2, 16, 50, 50]
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)  # [B, 2, 32, 25, 25]
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)  # [B, 2, 64, 5, 5]
        
        # 展平: 将复数维度和空间维度合并
        b = x.shape[0]
        x = x.view(b, -1)  # [B, 2*64*5*5]
        
        # 全连接回归
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 归一化到 [0, 1]
        
        return x
    
    def count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FDA_CVNN_Attention(nn.Module):
    """
    带注意力机制的 FDA-CVNN (增强低SNR性能)
    
    改进点:
    1. 每个卷积块后加入 SE 通道注意力
    2. 残差连接帮助梯度流动
    3. 更深的网络结构
    
    在低 SNR 下，注意力机制可以:
    - 自适应放大包含信号特征的通道
    - 抑制噪声主导的通道
    """
    def __init__(self, use_cbam=False):
        super().__init__()
        self.use_cbam = use_cbam
        
        # Block 1: 100 -> 50
        self.conv1 = ComplexConv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(32)
        self.act1 = ModReLU(32, bias_init=-0.5)
        self.attn1 = ComplexCBAM(32) if use_cbam else ComplexSEBlock(32)
        self.pool1 = ComplexAvgPool2d(2)
        
        # Block 2: 50 -> 25
        self.conv2 = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(64)
        self.act2 = ModReLU(64, bias_init=-0.5)
        self.attn2 = ComplexCBAM(64) if use_cbam else ComplexSEBlock(64)
        self.pool2 = ComplexAvgPool2d(2)
        
        # Block 3: 25 -> 12
        self.conv3 = ComplexConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm2d(128)
        self.act3 = ModReLU(128, bias_init=-0.5)
        self.attn3 = ComplexCBAM(128) if use_cbam else ComplexSEBlock(128)
        self.pool3 = ComplexAvgPool2d(2)
        
        # Block 4: 12 -> 6 (新增一层，更深的网络)
        self.conv4 = ComplexConv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = ComplexBatchNorm2d(256)
        self.act4 = ModReLU(256, bias_init=-0.5)
        self.attn4 = ComplexCBAM(256) if use_cbam else ComplexSEBlock(256)
        self.pool4 = ComplexAvgPool2d(2)
        
        # 全局平均池化
        self.global_pool = ComplexAdaptiveAvgPool2d(1)
        
        # 全连接层
        self.fc_in_dim = 256 * 2  # 256通道 * 实部虚部
        
        self.fc1 = nn.Linear(self.fc_in_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        """
        x: [B, 2, 100, 100]
        """
        x = x.unsqueeze(2)  # [B, 2, 1, 100, 100]
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.attn1(x)  # 注意力
        x = self.pool1(x)  # [B, 2, 32, 50, 50]
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.attn2(x)
        x = self.pool2(x)  # [B, 2, 64, 25, 25]
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.attn3(x)
        x = self.pool3(x)  # [B, 2, 128, 12, 12]
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.attn4(x)
        x = self.pool4(x)  # [B, 2, 256, 6, 6]
        
        # 全局池化
        x = self.global_pool(x)  # [B, 2, 256, 1, 1]
        
        # 展平
        b = x.shape[0]
        x = x.view(b, -1)  # [B, 512]
        
        # 全连接回归
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FDA_CVNN_Light(nn.Module):
    """
    轻量级版本，适合快速测试
    """
    def __init__(self):
        super().__init__()
        
        # Block 1: 100 -> 25
        self.conv1 = ComplexConv2d(1, 16, kernel_size=3, padding=1)
        self.act1 = ModReLU(16, bias_init=-0.5)
        self.pool1 = ComplexAvgPool2d(4)
        
        # Block 2: 25 -> 5
        self.conv2 = ComplexConv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = ModReLU(32, bias_init=-0.5)
        self.pool2 = ComplexAvgPool2d(5)
        
        # 全连接
        self.fc_in_dim = 32 * 5 * 5 * 2
        self.fc1 = nn.Linear(self.fc_in_dim, 128)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = x.unsqueeze(2)
        
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        
        b = x.shape[0]
        x = x.view(b, -1)
        
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        return x
    
    def count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("测试 FDA_CVNN 模型 (原始版)")
    print("=" * 60)
    
    model = FDA_CVNN()
    print(f"模型参数量: {model.count_parameters():,}")
    
    # 模拟输入
    x = torch.randn(4, 2, 100, 100)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        y = model(x)
    print(f"输出形状: {y.shape}")
    print(f"输出范围: [{y.min().item():.4f}, {y.max().item():.4f}]")
    
    # 测试带注意力的模型
    print("\n" + "=" * 60)
    print("测试 FDA_CVNN_Attention 模型 (SE注意力)")
    print("=" * 60)
    
    model_attn = FDA_CVNN_Attention(use_cbam=False)
    print(f"模型参数量: {model_attn.count_parameters():,}")
    
    with torch.no_grad():
        y_attn = model_attn(x)
    print(f"输出形状: {y_attn.shape}")
    print(f"输出范围: [{y_attn.min().item():.4f}, {y_attn.max().item():.4f}]")
    
    # 测试 CBAM 版本
    print("\n" + "=" * 60)
    print("测试 FDA_CVNN_Attention 模型 (CBAM注意力)")
    print("=" * 60)
    
    model_cbam = FDA_CVNN_Attention(use_cbam=True)
    print(f"模型参数量: {model_cbam.count_parameters():,}")
    
    with torch.no_grad():
        y_cbam = model_cbam(x)
    print(f"输出形状: {y_cbam.shape}")
    print(f"输出范围: [{y_cbam.min().item():.4f}, {y_cbam.max().item():.4f}]")
    
    # 测试轻量级模型
    print("\n" + "=" * 60)
    print("测试 FDA_CVNN_Light 模型")
    print("=" * 60)
    model_light = FDA_CVNN_Light()
    print(f"轻量级模型参数量: {model_light.count_parameters():,}")
    
    with torch.no_grad():
        y_light = model_light(x)
    print(f"输出形状: {y_light.shape}")

