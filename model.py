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


class ComplexFARBlock(nn.Module):
    """
    复数版 FAR (Feature Attention Refinement) Block
    
    与 SE 的核心区别：
    - SE: 全局池化 → 通道级注意力 [B, 1, C, 1, 1]
    - FAR: 局部池化 → 空间+通道级注意力 [B, 1, C, H, W]
    
    优势：保留空间位置信息，更适合协方差矩阵这种空间结构有意义的输入
    """
    def __init__(self, channels, kernel_size=3, reduction=4):
        super().__init__()
        
        features = max(channels // reduction, 8)  # 确保至少8个特征
        padding = (kernel_size - 1) // 2
        
        # 1. 局部平均池化 (LAP) - 获取局部上下文，不改变尺寸
        self.local_avg_pool = ComplexAvgPool2d(
            kernel_size=kernel_size, stride=1, padding=padding
        )
        
        # 2. 特征重加权网络
        # Layer 1: 降维 (1x1 Conv)
        self.conv1 = ComplexConv2d(channels, features, kernel_size=1)
        self.bn1 = ComplexBatchNorm2d(features)
        self.act1 = ModReLU(features, bias_init=-0.5)
        
        # Layer 2: 升维 (1x1 Conv)
        self.conv2 = ComplexConv2d(features, channels, kernel_size=1)
        
        # Sigmoid 用于生成注意力权重
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        x: [B, 2, C, H, W]
        """
        # 1. 局部平均池化获取上下文
        y = self.local_avg_pool(x)  # [B, 2, C, H, W]
        
        # 2. 生成注意力权重
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.conv2(y)  # [B, 2, C, H, W]
        
        # 3. 基于模值生成注意力图
        real = y[:, 0]  # [B, C, H, W]
        imag = y[:, 1]
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        attn = self.sigmoid(mag)  # [B, C, H, W]
        
        # 扩展维度: [B, 1, C, H, W]
        attn = attn.unsqueeze(1)
        
        # 4. 重加权
        return x * attn


class ComplexCBAM(nn.Module):
    """
    复数 CBAM (Convolutional Block Attention Module)
    = 通道注意力 + 空间注意力
    
    注意：使用了 Max Pooling，可能破坏相位干涉特征
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


class ComplexDualAttention(nn.Module):
    """
    【创新点核心模块】相位保持双尺度注意力 (Phase-Preserving Dual-Scale Attention, PP-DSA)
    
    结合了：
    1. SE (Global Path): 全局平均池化 → 捕捉全孔径相位依赖，保证角度分辨率
    2. FAR (Local Path): 局部平均池化 → 软阈值去噪，抑制非相干噪声
    
    优势：
    - 全程无 Max Pooling，完美保留复数相位线性叠加特性
    - SE 提供全局通道校准，FAR 提供局部空间去噪
    - 串联结构：先全局统筹，再局部精修
    
    与 CBAM 的区别：
    - CBAM 空间注意力使用 Max Pooling，可能破坏相位干涉条纹
    - 本模块全程使用 Average Pooling，保持相位安全
    """
    def __init__(self, channels, reduction=4, far_kernel=3):
        super().__init__()
        
        # 1. 全局路径 (SE Block) - 通道级全局校准
        self.global_attn = ComplexSEBlock(channels, reduction)
        
        # 2. 局部路径 (FAR Block) - 空间级局部去噪
        self.local_attn = ComplexFARBlock(channels, kernel_size=far_kernel, reduction=reduction)
        
    def forward(self, x):
        """
        串联结构：先全局统筹 (SE)，再局部精修 (FAR)
        
        x: [B, 2, C, H, W]
        """
        # 第一步：全局校准 (SE) - 通道重加权
        x = self.global_attn(x)
        
        # 第二步：局部去噪 (FAR) - 空间+通道精修
        x = self.local_attn(x)
        
        return x


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
    带注意力机制的 FDA-CVNN
    
    设计原则：完全保持原始 FDA_CVNN 的架构，只加入轻量级注意力模块
    - 保持 3 层卷积结构（不加深）
    - 保持 pool3(5) 输出 5x5 特征图（不使用全局池化）
    - 保持相同的全连接层结构
    
    参数:
        attention_type: 注意力类型
            - 'se': SE 通道注意力 (全局平均池化)
            - 'cbam': CBAM (SE + 空间注意力，含 MaxPool，可能破坏相位)
            - 'far': FAR 局部注意力 (局部平均池化)
            - 'dual': 【创新】SE + FAR 串联 (相位保持双尺度注意力 PP-DSA)
        se_reduction: 注意力模块的通道压缩比，默认 4
        deep_only: 是否只在深层使用注意力 (Block2, Block3)，默认 False
        far_kernel: FAR 局部池化核大小，默认 3
    """
    def __init__(self, attention_type='se', se_reduction=4, deep_only=False, far_kernel=3,
                 use_cbam=False):  # use_cbam 保留用于向后兼容
        super().__init__()
        
        # 向后兼容：如果使用旧的 use_cbam 参数
        if use_cbam:
            attention_type = 'cbam'
        
        self.attention_type = attention_type
        self.se_reduction = se_reduction
        self.deep_only = deep_only
        
        # 定义注意力构建函数
        def build_attn(channels):
            if attention_type == 'cbam':
                return ComplexCBAM(channels, reduction=se_reduction)
            elif attention_type == 'far':
                return ComplexFARBlock(channels, kernel_size=far_kernel, reduction=se_reduction)
            elif attention_type == 'dual':
                return ComplexDualAttention(channels, reduction=se_reduction, far_kernel=far_kernel)
            else:  # 'se' 或默认
                return ComplexSEBlock(channels, reduction=se_reduction)
        
        # Block 1: 100 -> 50
        self.conv1 = ComplexConv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(32)
        self.act1 = ModReLU(32, bias_init=-0.5)
        # 浅层注意力可选
        if not deep_only:
            self.attn1 = build_attn(32)
        else:
            self.attn1 = None
        self.pool1 = ComplexAvgPool2d(2)
        
        # Block 2: 50 -> 25
        self.conv2 = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(64)
        self.act2 = ModReLU(64, bias_init=-0.5)
        self.attn2 = build_attn(64)
        self.pool2 = ComplexAvgPool2d(2)
        
        # Block 3: 25 -> 5
        self.conv3 = ComplexConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm2d(128)
        self.act3 = ModReLU(128, bias_init=-0.5)
        self.attn3 = build_attn(128)
        self.pool3 = ComplexAvgPool2d(5)  # 输出 5x5
        
        # 全连接层
        self.fc_in_dim = 128 * 5 * 5 * 2  # 6400
        
        self.fc1 = nn.Linear(self.fc_in_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        x: [B, 2, 100, 100]
        """
        x = x.unsqueeze(2)  # [B, 2, 1, 100, 100]
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.attn1 is not None:
            x = self.attn1(x)
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
        x = self.pool3(x)  # [B, 2, 128, 5, 5]
        
        # 展平
        b = x.shape[0]
        x = x.view(b, -1)  # [B, 6400]
        
        # 全连接回归
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FDA_CVNN_FAR(nn.Module):
    """
    保守版 FAR 注意力 FDA-CVNN
    
    FAR vs SE 的区别：
    - SE: 全局池化 → 通道级权重 [B, 1, C, 1, 1]（所有位置相同权重）
    - FAR: 局部池化 → 空间+通道级权重 [B, 1, C, H, W]（不同位置不同权重）
    
    设计原则：保持原始架构，只替换注意力模块
    """
    def __init__(self, far_kernel_size=3):
        super().__init__()
        
        # Block 1: 100 -> 50
        self.conv1 = ComplexConv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(32)
        self.act1 = ModReLU(32, bias_init=-0.5)
        self.attn1 = ComplexFARBlock(32, kernel_size=far_kernel_size)
        self.pool1 = ComplexAvgPool2d(2)
        
        # Block 2: 50 -> 25
        self.conv2 = ComplexConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(64)
        self.act2 = ModReLU(64, bias_init=-0.5)
        self.attn2 = ComplexFARBlock(64, kernel_size=far_kernel_size)
        self.pool2 = ComplexAvgPool2d(2)
        
        # Block 3: 25 -> 5
        self.conv3 = ComplexConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm2d(128)
        self.act3 = ModReLU(128, bias_init=-0.5)
        self.attn3 = ComplexFARBlock(128, kernel_size=far_kernel_size)
        self.pool3 = ComplexAvgPool2d(5)  # 输出 5x5
        
        # 全连接层 (与原始一致)
        self.fc_in_dim = 128 * 5 * 5 * 2  # 6400
        
        self.fc1 = nn.Linear(self.fc_in_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        x: [B, 2, 100, 100]
        """
        x = x.unsqueeze(2)  # [B, 2, 1, 100, 100]
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.attn1(x)  # FAR 注意力
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
        x = self.pool3(x)  # [B, 2, 128, 5, 5]
        
        # 展平
        b = x.shape[0]
        x = x.view(b, -1)  # [B, 6400]
        
        # 全连接回归
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
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
    
    # 测试 FAR 版本
    print("\n" + "=" * 60)
    print("测试 FDA_CVNN_FAR 模型 (FAR注意力) ⭐")
    print("=" * 60)
    
    model_far = FDA_CVNN_FAR(far_kernel_size=3)
    print(f"模型参数量: {model_far.count_parameters():,}")
    
    with torch.no_grad():
        y_far = model_far(x)
    print(f"输出形状: {y_far.shape}")
    print(f"输出范围: [{y_far.min().item():.4f}, {y_far.max().item():.4f}]")
    
    # 测试轻量级模型
    print("\n" + "=" * 60)
    print("测试 FDA_CVNN_Light 模型")
    print("=" * 60)
    model_light = FDA_CVNN_Light()
    print(f"轻量级模型参数量: {model_light.count_parameters():,}")
    
    with torch.no_grad():
        y_light = model_light(x)
    print(f"输出形状: {y_light.shape}")
    
    # 模型对比总结
    print("\n" + "=" * 60)
    print("📊 模型对比总结")
    print("=" * 60)
    print(f"{'模型':<25} {'参数量':>15} {'注意力类型':<20}")
    print("-" * 60)
    print(f"{'FDA_CVNN':<25} {model.count_parameters():>15,} {'无':<20}")
    print(f"{'FDA_CVNN_Attention (SE)':<25} {model_attn.count_parameters():>15,} {'通道级 (全局池化)':<20}")
    print(f"{'FDA_CVNN_Attention (CBAM)':<25} {model_cbam.count_parameters():>15,} {'通道+空间':<20}")
    print(f"{'FDA_CVNN_FAR ⭐':<25} {model_far.count_parameters():>15,} {'空间+通道 (局部池化)':<20}")
    print(f"{'FDA_CVNN_Light':<25} {model_light.count_parameters():>15,} {'无':<20}")
