"""
基线模型：Real-CNN（实数卷积神经网络）
用于消融实验，对比复数网络的优势
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RealCNN(nn.Module):
    """
    实数卷积神经网络 (消融实验用)
    
    与 FDA_CVNN 保持相同的层数和参数规模，但使用实数卷积
    输入: [B, 2, 100, 100] - 将复数矩阵的实部虚部作为2个通道
    输出: [B, 2] - 归一化的 (距离, 角度)
    """
    def __init__(self):
        super().__init__()
        
        # Block 1: 100 -> 50
        # 输入2通道(实部+虚部)，与CVNN的复数1通道对应
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)  # 2->64 (对应CVNN的1->32复数)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(2)
        
        # Block 2: 50 -> 25
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.AvgPool2d(2)
        
        # Block 3: 25 -> 5
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(5)
        
        # 全连接层
        self.fc_in_dim = 256 * 5 * 5
        self.fc1 = nn.Linear(self.fc_in_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        x: [B, 2, 100, 100] - 实部和虚部作为2个通道
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RealCNN_Light(nn.Module):
    """轻量级Real-CNN"""
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(4)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(5)
        
        self.fc_in_dim = 64 * 5 * 5
        self.fc1 = nn.Linear(self.fc_in_dim, 128)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    print("测试 RealCNN...")
    model = RealCNN()
    print(f"参数量: {model.count_parameters():,}")
    
    x = torch.randn(4, 2, 100, 100)
    with torch.no_grad():
        y = model(x)
    print(f"输入: {x.shape} -> 输出: {y.shape}")
    
    print("\n测试 RealCNN_Light...")
    model_light = RealCNN_Light()
    print(f"参数量: {model_light.count_parameters():,}")
