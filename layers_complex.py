"""
复数神经网络层
包含：复数卷积、ModReLU激活、复数池化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    """
    真正的复数卷积：(A+iB)*(C+iD) = (AC-BD) + i(AD+BC)
    
    输入形状: [Batch, 2, C_in, H, W] 其中 dim=1 是实部/虚部
    输出形状: [Batch, 2, C_out, H, W]
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_rr = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_ri = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_ir = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_ii = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        """
        x: [B, 2, C, H, W] - dim1=0是实部, dim1=1是虚部
        """
        real = x[:, 0]  # [B, C, H, W]
        imag = x[:, 1]  # [B, C, H, W]
        
        # 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        out_real = self.conv_rr(real) - self.conv_ii(imag)
        out_imag = self.conv_ri(real) + self.conv_ir(imag)
        
        return torch.stack([out_real, out_imag], dim=1)


class ComplexBatchNorm2d(nn.Module):
    """
    复数批归一化
    分别对实部和虚部进行归一化
    """
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)
        
    def forward(self, x):
        real = self.bn_real(x[:, 0])
        imag = self.bn_imag(x[:, 1])
        return torch.stack([real, imag], dim=1)


class ModReLU(nn.Module):
    """
    ModReLU 激活函数
    z_out = ReLU(|z| + b) * (z / |z|)
    
    关键：b 必须初始化为负值，创造非线性死区！
    如果 b=0，则退化为恒等映射，网络失去非线性能力。
    """
    def __init__(self, num_features, bias_init=-0.5):
        super().__init__()
        # 每个通道一个可学习的偏置
        self.bias = nn.Parameter(torch.full((num_features,), bias_init))
        
    def forward(self, x):
        """
        x: [B, 2, C, H, W]
        """
        real = x[:, 0]  # [B, C, H, W]
        imag = x[:, 1]
        
        # 计算模值
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        
        # 激活幅度: ReLU(|z| + b)
        # bias 形状调整为 [1, C, 1, 1] 以便广播
        b = self.bias.view(1, -1, 1, 1)
        mag_activated = F.relu(mag + b)
        
        # 缩放因子
        scale = mag_activated / (mag + 1e-8)
        
        # 输出：保持相位，调整幅度
        out_real = real * scale
        out_imag = imag * scale
        
        return torch.stack([out_real, out_imag], dim=1)


class ComplexAvgPool2d(nn.Module):
    """
    复数平均池化 (线性操作，不破坏相位)
    
    警告：绝不能用 MaxPool！MaxPool 会分别取实部虚部的最大值，
    可能来自不同位置，破坏相位关系。
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride, padding)
        
    def forward(self, x):
        """
        x: [B, 2, C, H, W]
        """
        real = self.pool(x[:, 0])
        imag = self.pool(x[:, 1])
        return torch.stack([real, imag], dim=1)


class ComplexAdaptiveAvgPool2d(nn.Module):
    """
    复数自适应平均池化
    """
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)
        
    def forward(self, x):
        real = self.pool(x[:, 0])
        imag = self.pool(x[:, 1])
        return torch.stack([real, imag], dim=1)


class ComplexDropout(nn.Module):
    """
    复数Dropout
    对实部和虚部使用相同的mask
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
            
        # 生成mask [B, 1, C, H, W]
        mask_shape = (x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4])
        mask = torch.bernoulli(torch.full(mask_shape, 1 - self.p, device=x.device))
        mask = mask / (1 - self.p)  # 缩放
        
        return x * mask


class ComplexLinear(nn.Module):
    """
    复数全连接层
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_rr = nn.Linear(in_features, out_features)
        self.fc_ri = nn.Linear(in_features, out_features)
        self.fc_ir = nn.Linear(in_features, out_features)
        self.fc_ii = nn.Linear(in_features, out_features)
        
    def forward(self, x_real, x_imag):
        """
        x_real, x_imag: [B, in_features]
        """
        out_real = self.fc_rr(x_real) - self.fc_ii(x_imag)
        out_imag = self.fc_ri(x_real) + self.fc_ir(x_imag)
        return out_real, out_imag


if __name__ == "__main__":
    # 测试复数层
    print("测试复数网络层...")
    
    B, C, H, W = 4, 16, 50, 50
    x = torch.randn(B, 2, C, H, W)  # 模拟复数输入
    
    # 测试复数卷积
    conv = ComplexConv2d(C, 32, kernel_size=3, padding=1)
    out = conv(x)
    print(f"ComplexConv2d: {x.shape} -> {out.shape}")
    
    # 测试ModReLU
    act = ModReLU(32, bias_init=-0.5)
    out2 = act(out)
    print(f"ModReLU: {out.shape} -> {out2.shape}")
    print(f"  bias值: {act.bias[:4].detach().numpy()}")
    
    # 测试复数池化
    pool = ComplexAvgPool2d(2)
    out3 = pool(out2)
    print(f"ComplexAvgPool2d: {out2.shape} -> {out3.shape}")
    
    # 测试BatchNorm
    bn = ComplexBatchNorm2d(32)
    out4 = bn(out2)
    print(f"ComplexBatchNorm2d: {out2.shape} -> {out4.shape}")
    
    print("\n所有层测试通过！")
