# -*- coding: utf-8 -*-
"""
快速诊断脚本 - 检查训练问题
"""

import torch
import numpy as np
from model import CVNN_Estimator_Pro, CVNN_Estimator_Light
from dataset_cached import create_dataloaders_cached
from config import DEVICE

def diagnose():
    print("=" * 60)
    print("训练问题诊断")
    print("=" * 60)
    
    # 1. 创建小数据集
    print("\n[1] 创建测试数据...")
    train_loader, val_loader, _ = create_dataloaders_cached(
        train_size=100,
        val_size=50,
        test_size=50,
        batch_size=16,
        num_workers=0,
        verbose=False
    )
    
    # 2. 检查数据
    print("\n[2] 检查输入数据...")
    batch_R, batch_label, batch_raw = next(iter(train_loader))
    print(f"  输入 R shape: {batch_R.shape}, dtype: {batch_R.dtype}")
    print(f"  输入 R 范围: [{batch_R.min():.4f}, {batch_R.max():.4f}]")
    print(f"  标签 (归一化) shape: {batch_label.shape}")
    print(f"  标签 (归一化) 范围: r=[{batch_label[:,0].min():.3f}, {batch_label[:,0].max():.3f}], θ=[{batch_label[:,1].min():.3f}, {batch_label[:,1].max():.3f}]")
    print(f"  原始标签 范围: r=[{batch_raw[:,0].min():.1f}m, {batch_raw[:,0].max():.1f}m], θ=[{batch_raw[:,1].min():.1f}°, {batch_raw[:,1].max():.1f}°]")
    
    # 3. 创建模型并测试前向传播
    print("\n[3] 测试模型前向传播...")
    model = CVNN_Estimator_Pro(use_batchnorm=True, dropout_rate=0.2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    batch_R = batch_R.to(device)
    
    with torch.no_grad():
        outputs = model(batch_R)
    
    print(f"  输出 shape: {outputs.shape}, dtype: {outputs.dtype}")
    print(f"  输出范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print(f"  输出均值: {outputs.mean():.4f}, 标准差: {outputs.std():.4f}")
    print(f"  前5个样本输出:")
    for i in range(min(5, outputs.shape[0])):
        r_pred, theta_pred = outputs[i].cpu().numpy()
        r_true, theta_true = batch_label[i].numpy()
        print(f"    样本{i+1}: 预测 r_norm={r_pred:.4f}, θ_norm={theta_pred:.4f} | 真实 r_norm={r_true:.4f}, θ_norm={theta_true:.4f}")
    
    # 4. 检查输出是否在合理范围
    print("\n[4] 输出范围检查...")
    r_out = outputs[:, 0].cpu()
    theta_out = outputs[:, 1].cpu()
    
    # 理想情况: 输出在[0,1]范围（归一化标签）
    r_in_range = ((r_out >= 0) & (r_out <= 1)).float().mean() * 100
    theta_in_range = ((theta_out >= 0) & (theta_out <= 1)).float().mean() * 100
    
    print(f"  r 在[0,1]范围内: {r_in_range:.1f}%")
    print(f"  θ 在[0,1]范围内: {theta_in_range:.1f}%")
    
    if r_in_range < 50 or theta_in_range < 50:
        print("\n  ⚠️ 警告: 大量输出超出[0,1]范围!")
        print("  原因: ComplexToReal使用abs模式，输出>=0但可能>1")
        print("  建议: 在输出层添加Sigmoid激活或修改归一化方式")
    
    # 5. 计算RMSE
    print("\n[5] RMSE计算测试...")
    # 反归一化
    pred_r = r_out.numpy() * 2000  # r_max - r_min = 2000
    pred_theta = theta_out.numpy() * 120 - 60  # theta_range = [-60, 60]
    
    true_r = batch_raw[:, 0].numpy()
    true_theta = batch_raw[:, 1].numpy()
    
    rmse_r = np.sqrt(np.mean((pred_r - true_r) ** 2))
    rmse_theta = np.sqrt(np.mean((pred_theta - true_theta) ** 2))
    
    print(f"  RMSE_r: {rmse_r:.2f}m")
    print(f"  RMSE_θ: {rmse_theta:.2f}°")
    
    if rmse_r > 1000 or rmse_theta > 60:
        print("\n  ⚠️ RMSE异常高! 可能原因:")
        print("    1. 模型输出未经过Sigmoid约束到[0,1]")
        print("    2. 初始化权重导致输出远离目标范围")
        print("    3. 学习率过大/过小")
    
    # 6. 建议
    print("\n" + "=" * 60)
    print("诊断建议")
    print("=" * 60)
    
    if outputs.min() < 0 or outputs.max() > 1:
        print("\n✗ 问题: 模型输出超出[0,1]范围")
        print("  解决: 在输出层添加Sigmoid")
        print("""
  修改 model.py 中的输出:
  
  # 当前代码:
  x = self.fc_out(x)
  out = self.to_real(x)
  
  # 修改为:
  x = self.fc_out(x)
  out = torch.sigmoid(self.to_real(x))  # 添加Sigmoid
        """)
    else:
        print("\n✓ 模型输出在正常范围内")


if __name__ == "__main__":
    diagnose()
