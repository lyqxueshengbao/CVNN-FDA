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
        
        # Block 1: 100 -> 50
        self.conv1 = ComplexConv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(16)
        self.act1 = ModReLU(16, bias_init=-0.5)
        self.pool1 = ComplexAvgPool2d(2)
        
        # Block 2: 50 -> 25
        self.conv2 = ComplexConv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(32)
        self.act2 = ModReLU(32, bias_init=-0.5)
        self.pool2 = ComplexAvgPool2d(2)
        
        # Block 3: 25 -> 5
        self.conv3 = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm2d(64)
        self.act3 = ModReLU(64, bias_init=-0.5)
        self.pool3 = ComplexAvgPool2d(5)
        
        # 全连接层
        # 特征图大小: 5x5, 通道64, 实部+虚部
        self.fc_in_dim = 64 * 5 * 5 * 2
        
        self.fc1 = nn.Linear(self.fc_in_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # 输出 r 和 theta
        
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
    print("测试 FDA_CVNN 模型...")
    
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
    print(f"输出示例:\n{y}")
    
    # 测试轻量级模型
    print("\n测试 FDA_CVNN_Light 模型...")
    model_light = FDA_CVNN_Light()
    print(f"轻量级模型参数量: {model_light.count_parameters():,}")
    
    with torch.no_grad():
        y_light = model_light(x)
    print(f"输出形状: {y_light.shape}")
