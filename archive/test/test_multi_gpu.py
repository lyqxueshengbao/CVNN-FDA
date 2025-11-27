# -*- coding: utf-8 -*-
"""
多GPU DataParallel 兼容性测试脚本
"""

import torch
import torch.nn as nn
from model import CVNN_Estimator_Pro
from dataset import create_dataloaders

def test_multi_gpu_forward():
    """测试多GPU前向传播"""
    print("=" * 60)
    print("测试 DataParallel 兼容性")
    print("=" * 60)
    
    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"\n可用GPU数量: {num_gpus}")
    
    if num_gpus < 2:
        print("警告: 需要至少2个GPU来测试DataParallel")
        print("将只测试单GPU情况...")
    
    # 创建模型
    print("\n创建 Pro 模型...")
    model = CVNN_Estimator_Pro(use_batchnorm=True, dropout_rate=0.2)
    device = torch.device('cuda:0')
    model = model.to(device)
    
    # 包装为 DataParallel
    if num_gpus > 1:
        print(f"包装为 DataParallel (使用 {num_gpus} 个GPU)...")
        model = nn.DataParallel(model)
    
    # 创建测试数据
    print("\n创建测试数据...")
    train_loader, val_loader, _ = create_dataloaders(
        train_size=64,
        val_size=16,
        test_size=16,
        batch_size=32,
        num_workers=0
    )
    
    # 测试前向传播
    print("\n测试前向传播...")
    model.eval()
    
    with torch.no_grad():
        for i, (batch_R, batch_label, _) in enumerate(train_loader):
            print(f"\nBatch {i+1}:")
            print(f"  输入形状: {batch_R.shape}, dtype: {batch_R.dtype}")
            
            # 移动到GPU
            batch_R = batch_R.to(device)
            batch_label = batch_label.to(device)
            
            # 前向传播
            try:
                outputs = model(batch_R)
                print(f"  输出形状: {outputs.shape}, dtype: {outputs.dtype}")
                print(f"  输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print("  ✓ 前向传播成功!")
            except Exception as e:
                print(f"  ✗ 前向传播失败: {e}")
                raise
            
            # 只测试2个batch
            if i >= 1:
                break
    
    print("\n" + "=" * 60)
    print("测试完成! DataParallel 兼容性正常")
    print("=" * 60)


if __name__ == "__main__":
    test_multi_gpu_forward()
