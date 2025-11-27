# -*- coding: utf-8 -*-
"""
复数波束精修正网络 (Complex Beam Refinement Network)

基于粗精结合策略的核心创新模块
输入: 5x5 复数patch
输出: 残差修正 (delta_r, delta_theta)
"""

import torch
import torch.nn as nn
from complex_layers import ComplexLinear, ComplexFlatten, ModReLU


class ComplexBeamRefineNet(nn.Module):
    """
    复数波束精修正网络
    
    网络结构:
    - 输入: 5x5 复数patch (25个复数值)
    - 全连接层: 25 -> 64 -> 32
    - 输出: 2维实数 (delta_r, delta_theta)
    
    特点:
    1. 轻量级 - 仅约5K参数
    2. 端到端训练 - 直接回归残差
    3. 保留相位信息 - 充分利用复数特征
    """
    
    def __init__(self, 
                 patch_size: int = 5,
                 hidden_dim1: int = 64,
                 hidden_dim2: int = 32,
                 dropout_rate: float = 0.1):
        """
        Args:
            patch_size: Patch大小
            hidden_dim1: 第一隐层维度
            hidden_dim2: 第二隐层维度
            dropout_rate: Dropout概率
        """
        super().__init__()
        
        self.patch_size = patch_size
        input_dim = patch_size * patch_size  # 25
        
        # 展平层
        self.flatten = ComplexFlatten()
        
        # 复数全连接层1: 25 -> 64
        self.fc1 = ComplexLinear(input_dim, hidden_dim1)
        self.act1 = ModReLU(num_features=hidden_dim1, bias_init=-0.1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 复数全连接层2: 64 -> 32
        self.fc2 = ComplexLinear(hidden_dim1, hidden_dim2)
        self.act2 = ModReLU(num_features=hidden_dim2, bias_init=-0.1)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 输出层: 实数线性层
        # 输入维度: 32*2 (实部+虚部拼接)
        self.fc_out = nn.Linear(hidden_dim2 * 2, 2)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入patch, shape (batch, 2, patch_size, patch_size)
               2通道为[实部, 虚部]
        
        Returns:
            out: 残差预测, shape (batch, 2)
                 [delta_r_norm, delta_theta_norm]
        """
        # 转换为复数 (如果是2通道实数输入)
        if not x.is_complex() and x.shape[1] == 2:
            x = torch.complex(x[:, 0], x[:, 1])  # (batch, patch_size, patch_size)
        
        # 展平: (batch, patch_size, patch_size) -> (batch, patch_size*patch_size)
        x = self.flatten(x)
        
        # 第一层: 25 -> 64
        x = self.fc1(x)
        x = self.act1(x)
        # Dropout应用在实部和虚部
        x = torch.complex(
            self.dropout1(x.real),
            self.dropout1(x.imag)
        )
        
        # 第二层: 64 -> 32
        x = self.fc2(x)
        x = self.act2(x)
        x = torch.complex(
            self.dropout2(x.real),
            self.dropout2(x.imag)
        )
        
        # 转换为实数: 拼接实虚部
        x_real = torch.cat([x.real, x.imag], dim=1)  # (batch, 64)
        
        # 输出层: 64 -> 2
        out = self.fc_out(x_real)  # (batch, 2)
        
        return out


class ComplexBeamRefineNet_Deep(nn.Module):
    """
    深度复数波束精修正网络
    
    更多层次,更强表达能力
    """
    
    def __init__(self,
                 patch_size: int = 5,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.patch_size = patch_size
        input_dim = patch_size * patch_size
        
        self.flatten = ComplexFlatten()
        
        # 更深的网络: 25 -> 128 -> 64 -> 32
        self.fc1 = ComplexLinear(input_dim, 128)
        self.act1 = ModReLU(num_features=128, bias_init=-0.1)
        
        self.fc2 = ComplexLinear(128, 64)
        self.act2 = ModReLU(num_features=64, bias_init=-0.1)
        
        self.fc3 = ComplexLinear(64, 32)
        self.act3 = ModReLU(num_features=32, bias_init=-0.1)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 输出层
        self.fc_out = nn.Linear(32 * 2, 2)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex() and x.shape[1] == 2:
            x = torch.complex(x[:, 0], x[:, 1])
        
        x = self.flatten(x)
        
        # 层1
        x = self.act1(self.fc1(x))
        x = torch.complex(self.dropout(x.real), self.dropout(x.imag))
        
        # 层2
        x = self.act2(self.fc2(x))
        x = torch.complex(self.dropout(x.real), self.dropout(x.imag))
        
        # 层3
        x = self.act3(self.fc3(x))
        
        # 输出
        x_real = torch.cat([x.real, x.imag], dim=1)
        out = self.fc_out(x_real)
        
        return out


def count_parameters(model: nn.Module) -> int:
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("测试 ComplexBeamRefineNet")
    print("=" * 60)
    
    # 创建模型
    model = ComplexBeamRefineNet(patch_size=5)
    print(f"\n模型参数量: {count_parameters(model):,}")
    
    # 创建测试输入
    batch_size = 8
    patch_size = 5
    x = torch.randn(batch_size, 2, patch_size, patch_size)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    out = model(x)
    print(f"输出形状: {out.shape}")
    print(f"输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
    
    # 测试深度模型
    print("\n" + "=" * 60)
    print("测试 ComplexBeamRefineNet_Deep")
    print("=" * 60)
    
    model_deep = ComplexBeamRefineNet_Deep(patch_size=5)
    print(f"\n模型参数量: {count_parameters(model_deep):,}")
    
    out_deep = model_deep(x)
    print(f"输出形状: {out_deep.shape}")
    print(f"输出范围: [{out_deep.min().item():.4f}, {out_deep.max().item():.4f}]")
    
    print("\n✓ 测试通过!")
