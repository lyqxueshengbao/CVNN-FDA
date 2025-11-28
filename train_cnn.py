"""
Real-CNN 训练脚本 (消融实验)
用于训练实数卷积神经网络，作为 CVNN 的对比基线
"""
import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import config as cfg
from models_baseline import RealCNN
from dataset import create_dataloaders
from utils_physics import denormalize_labels
from train import RangeAngleLoss, compute_metrics  # 复用 CVNN 的损失函数和评估指标

def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # 前向传播
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            
            total_loss += loss.item()
            all_preds.append(preds)
            all_labels.append(batch_y)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics

def train_real_cnn(epochs=None, lr=None, batch_size=None, train_samples=None):
    """
    Real-CNN 训练主函数
    """
    # 参数设置 (复用 config.py，保持公平对比)
    epochs = epochs or cfg.epochs
    lr = lr or cfg.lr
    batch_size = batch_size or cfg.batch_size
    train_samples = train_samples or cfg.train_samples
    device = cfg.device
    
    print("=" * 60)
    print("Real-CNN 基线模型训练")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"训练样本: {train_samples}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {lr}")
    print(f"训练轮数: {epochs}")
    print("=" * 60)
    
    # 创建模型
    model = RealCNN().to(device)
    print(f"模型参数量: {model.count_parameters():,}")
    
    # 创建数据加载器
    train_loader, val_loader, _ = create_dataloaders(
        train_samples=train_samples,
        batch_size=batch_size,
        online_train=True
    )
    
    # 优化器和损失函数 (与 CVNN 保持一致)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = RangeAngleLoss(lambda_angle=1.0, range_weight=2.0)
    
    # 训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'val_rmse_r': [], 'val_rmse_theta': []
    }
    
    best_val_rmse_r = float('inf')
    
    # 保存路径
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(cfg.checkpoint_dir, "real_cnn_best.pth")
    
    # 训练循环
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 学习率调整
        scheduler.step()
        
        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_rmse_r'].append(val_metrics['rmse_r'])
        history['val_rmse_theta'].append(val_metrics['rmse_theta'])
        
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val   Loss: {val_metrics['loss']:.6f}, "
              f"RMSE_r: {val_metrics['rmse_r']:.2f}m, "
              f"RMSE_θ: {val_metrics['rmse_theta']:.2f}°")
        
        # 保存最佳模型
        if val_metrics['rmse_r'] < best_val_rmse_r:
            best_val_rmse_r = val_metrics['rmse_r']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_rmse_r': val_metrics['rmse_r'],
            }, save_path)
            print(f"  ★ 保存最佳模型 (RMSE_r: {best_val_rmse_r:.2f}m)")
            
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Real-CNN 训练完成！")
    print(f"总用时: {total_time/60:.1f} 分钟")
    print(f"最佳 RMSE_r: {best_val_rmse_r:.2f}m")
    print(f"模型已保存至: {save_path}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--samples', type=int, default=None)
    args = parser.parse_args()
    
    train_real_cnn(epochs=args.epochs, train_samples=args.samples)
