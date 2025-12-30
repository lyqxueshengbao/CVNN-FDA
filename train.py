"""
FDA-CVNN 训练脚本
"""
import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import config as cfg
from model import FDA_CVNN, FDA_CVNN_Light, FDA_CVNN_Attention, FDA_CVNN_FAR
from dataset import create_dataloaders, FastDataLoader
from utils_physics import denormalize_labels


class RangeAngleLoss(nn.Module):
    """
    距离-角度联合损失函数
    
    支持多种损失计算方式:
    - 'l1': L1Loss 分别计算距离和角度误差后加权求和 (默认)
    - 'l2': L2 欧氏距离，把 (r, θ) 看作二维点
    - 'complex': 复数欧氏距离，把 (r, θ) 转为极坐标复数 z = r * exp(jθ)
    """
    def __init__(self, lambda_angle: float = 1.0, range_weight: float = 2.0, 
                 loss_type: str = 'l1'):
        super().__init__()
        self.lambda_angle = lambda_angle
        self.range_weight = range_weight
        self.loss_type = loss_type
        
        # L1 损失基准
        self.criterion = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r_pred, theta_pred = pred[:, 0], pred[:, 1]
        r_target, theta_target = target[:, 0], target[:, 1]
        
        if self.loss_type == 'l2':
            # L2 欧氏距离: sqrt((r_pred - r_true)^2 + (theta_pred - theta_true)^2)
            # 加权版本
            r_diff = self.range_weight * (r_pred - r_target)
            theta_diff = self.lambda_angle * (theta_pred - theta_target)
            loss = torch.sqrt(r_diff**2 + theta_diff**2 + 1e-8).mean()
            
        elif self.loss_type == 'complex':
            # 复数欧氏距离: |z_pred - z_true| where z = r * exp(j * theta)
            # 注意: 输入是归一化值 [0,1]，需要转换到合理的角度范围
            # theta 归一化到 [0,1] 对应实际角度范围，这里转为弧度 [-π/2, π/2] 近似
            theta_pred_rad = (theta_pred - 0.5) * np.pi  # 映射到 [-π/2, π/2]
            theta_target_rad = (theta_target - 0.5) * np.pi
            
            # 复数表示 (使用加权的 r)
            r_pred_w = self.range_weight * r_pred
            r_target_w = self.range_weight * r_target
            
            # z = r * exp(j*theta) = r*cos(theta) + j*r*sin(theta)
            real_pred = r_pred_w * torch.cos(theta_pred_rad)
            imag_pred = r_pred_w * torch.sin(theta_pred_rad)
            real_target = r_target_w * torch.cos(theta_target_rad)
            imag_target = r_target_w * torch.sin(theta_target_rad)
            
            # 复数欧氏距离: |z_pred - z_target|
            loss = torch.sqrt((real_pred - real_target)**2 + 
                              (imag_pred - imag_target)**2 + 1e-8).mean()
        else:
            # 默认 L1: 分别计算后加权求和
            loss_r = self.criterion(r_pred, r_target)
            loss_theta = self.criterion(theta_pred, theta_target)
            loss = self.range_weight * loss_r + self.lambda_angle * loss_theta
        
        return loss


def compute_metrics(preds, labels):
    """
    计算评估指标
    
    参数:
        preds: 预测值 [B, 2] (归一化)
        labels: 真实值 [B, 2] (归一化)
    
    返回:
        dict: 包含各项指标
    """
    # 反归一化
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    preds_physical = denormalize_labels(preds_np)
    labels_physical = denormalize_labels(labels_np)
    
    # 距离误差
    r_pred = preds_physical[:, 0]
    r_true = labels_physical[:, 0]
    r_error = r_pred - r_true
    rmse_r = np.sqrt(np.mean(r_error ** 2))
    mae_r = np.mean(np.abs(r_error))
    
    # 角度误差
    theta_pred = preds_physical[:, 1]
    theta_true = labels_physical[:, 1]
    theta_error = theta_pred - theta_true
    rmse_theta = np.sqrt(np.mean(theta_error ** 2))
    mae_theta = np.mean(np.abs(theta_error))
    
    return {
        'rmse_r': rmse_r,
        'mae_r': mae_r,
        'rmse_theta': rmse_theta,
        'mae_theta': mae_theta
    }


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
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
        all_preds.append(preds.detach())
        all_labels.append(batch_y.detach())
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    # 计算指标
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics


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


def train(model_type='standard', epochs=None, lr=None, batch_size=None,
          train_samples=None, snr_train_range=None, save_best=True,
          se_reduction=4, deep_only=False, snapshots=None, random_snapshots=False,
          loss_type='l1'):
    """
    主训练函数
    
    参数:
        model_type: 模型类型 ('standard', 'light', 'attention', 'cbam', 'far', 'dual')
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批次大小
        loss_type: 损失函数类型 ('l1', 'l2', 'complex')
        train_samples: 训练样本数
        snr_train_range: 训练SNR范围
        save_best: 是否保存最佳模型
        se_reduction: SE 模块的通道压缩比 (4, 8, 16)
        deep_only: 是否只在深层使用注意力 (跳过 Block1)
        snapshots: 快拍数 L (None 则使用 config.py 中的默认值)
        random_snapshots: 是否随机化快拍数训练 (提高对不同快拍数的鲁棒性)
    """
    # 参数设置
    epochs = epochs or cfg.epochs
    lr = lr or cfg.lr
    batch_size = batch_size or cfg.batch_size
    train_samples = train_samples or cfg.train_samples
    snr_train_range = snr_train_range or (cfg.snr_train_min, cfg.snr_train_max)
    device = cfg.device
    
    # 设置快拍数 (动态修改 config)
    if snapshots is not None:
        cfg.L_snapshots = snapshots
    L = cfg.L_snapshots
    
    # 快拍数随机化范围
    L_range = None
    if random_snapshots:
        L_range = (1, 36)  # 训练时快拍数从 1 到 100 随机
        L_str = f"random({L_range[0]}-{L_range[1]})"
        save_suffix = "Lrandom"
    else:
        L_str = str(L)
        save_suffix = f"L{L}"

    # 添加 SNR 后缀
    snr_min, snr_max = snr_train_range
    if snr_min == snr_max:
        # 单个 SNR 值
        snr_str = f"SNR{snr_min}" if snr_min >= 0 else f"SNRm{abs(snr_min)}"
        save_suffix += f"_{snr_str}"
    else:
        # SNR 范围（负数用 m 表示）
        snr_min_str = f"{snr_min}" if snr_min >= 0 else f"m{abs(snr_min)}"
        snr_max_str = f"{snr_max}" if snr_max >= 0 else f"m{abs(snr_max)}"
        save_suffix += f"_SNR{snr_min_str}to{snr_max_str}"

    # 动态生成保存路径：包含模型类型、快拍数和 SNR
    # 格式: checkpoints/fda_cvnn_{model_type}_L{snapshots}_SNR{snr}_best.pth
    if model_type == 'standard':
        save_name = f"fda_cvnn_{save_suffix}_best.pth"
    else:
        save_name = f"fda_cvnn_{model_type}_{save_suffix}_best.pth"
    save_path = os.path.join(cfg.checkpoint_dir, save_name)
    
    print("=" * 60)
    print("FDA-CVNN 训练")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"模型: {model_type}")
    print(f"快拍数: L = {L_str}" + (" ⭐ 随机化训练" if random_snapshots else ""))
    print(f"训练样本: {train_samples}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {lr}")
    print(f"训练轮数: {epochs}")
    print(f"SNR范围: {snr_train_range} dB")
    print(f"保存路径: {save_path}")
    print("=" * 60)
    
    # 创建模型
    if model_type == 'light':
        model = FDA_CVNN_Light().to(device)
    elif model_type == 'attention' or model_type == 'se':
        model = FDA_CVNN_Attention(attention_type='se', se_reduction=se_reduction, deep_only=deep_only).to(device)
        print(f"使用 SE 通道注意力机制 (reduction={se_reduction}, deep_only={deep_only})")
    elif model_type == 'cbam':
        model = FDA_CVNN_Attention(attention_type='cbam', se_reduction=se_reduction, deep_only=deep_only).to(device)
        print(f"使用 CBAM (通道+空间) 注意力机制 (reduction={se_reduction}, deep_only={deep_only})")
    elif model_type == 'far':
        model = FDA_CVNN_Attention(attention_type='far', se_reduction=se_reduction, deep_only=deep_only).to(device)
        print(f"使用 FAR 局部注意力机制 (reduction={se_reduction}, deep_only={deep_only})")
    elif model_type == 'dual':
        model = FDA_CVNN_Attention(attention_type='dual', se_reduction=se_reduction, deep_only=deep_only).to(device)
        print(f"使用 PP-DSA 双尺度注意力 (SE+FAR) ⭐ (reduction={se_reduction}, deep_only={deep_only})")
    else:
        model = FDA_CVNN().to(device)
    
    print(f"模型参数量: {model.count_parameters():,}")
    
    # 创建数据加载器
    # 1. 获取验证集和测试集 (保持离线固定模式，确保评估一致性)
    _, val_loader, _ = create_dataloaders(
        train_samples=100, # 占位，不使用
        batch_size=batch_size,
        snr_train_range=snr_train_range,
        online_train=True
    )
    
    # 2. 使用 FastDataLoader 加速训练 (GPU直接生成)
    print("启用 GPU 加速数据生成...")
    train_loader = FastDataLoader(
        batch_size=batch_size,
        num_samples=train_samples,
        snr_range=snr_train_range,
        device=device,
        L_range=L_range  # 快拍数随机化
    )
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 使用余弦退火调度器，学习率平滑下降，有助于收敛到更深的山谷
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    # 损失函数：支持 l1, l2, complex 三种类型
    criterion = RangeAngleLoss(lambda_angle=1.0, range_weight=2.0, loss_type=loss_type)
    print(f"损失函数类型: {loss_type}")
    
    # 训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_rmse_r': [], 'val_rmse_r': [],
        'train_rmse_theta': [], 'val_rmse_theta': []
    }
    
    best_val_rmse_r = float('inf')
    best_epoch = 0
    
    # 创建检查点目录
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    # 训练循环
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 学习率调整 (CosineAnnealingLR 不需要传参数)
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_rmse_r'].append(train_metrics['rmse_r'])
        history['val_rmse_r'].append(val_metrics['rmse_r'])
        history['train_rmse_theta'].append(train_metrics['rmse_theta'])
        history['val_rmse_theta'].append(val_metrics['rmse_theta'])
        
        # 打印指标
        print(f"  Train - Loss: {train_metrics['loss']:.6f}, "
              f"RMSE_r: {train_metrics['rmse_r']:.2f}m, "
              f"RMSE_θ: {train_metrics['rmse_theta']:.2f}°")
        print(f"  Val   - Loss: {val_metrics['loss']:.6f}, "
              f"RMSE_r: {val_metrics['rmse_r']:.2f}m, "
              f"RMSE_θ: {val_metrics['rmse_theta']:.2f}°")
        
        # 保存最佳模型
        if save_best and val_metrics['rmse_r'] < best_val_rmse_r:
            best_val_rmse_r = val_metrics['rmse_r']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_type': model_type,  # 保存模型类型，便于 benchmark 加载
                'se_reduction': se_reduction,  # 保存 SE 压缩比
                'deep_only': deep_only,  # 保存是否只在深层使用注意力
                'snapshots': L,  # 保存快拍数
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse_r': val_metrics['rmse_r'],
                'val_rmse_theta': val_metrics['rmse_theta'],
            }, save_path)  # 使用动态路径
            print(f"  ★ 保存最佳模型 (RMSE_r: {best_val_rmse_r:.2f}m) -> {save_name}")
    
    # 训练完成
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"总用时: {total_time/60:.1f} 分钟")
    print(f"最佳模型: Epoch {best_epoch}, RMSE_r = {best_val_rmse_r:.2f}m")
    print(f"保存位置: {save_path}")
    print("=" * 60)
    
    # 保存训练历史 (包含模型类型、快拍数和 SNR)
    history_filename = f"training_history_{model_type}_{save_suffix}.json"
    history_path = os.path.join(cfg.checkpoint_dir, history_filename)
    # 转换为Python原生类型
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    print(f"训练历史已保存到: {history_path}")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FDA-CVNN Training')
    parser.add_argument('--model', type=str, default='standard', 
                        choices=['standard', 'light', 'attention', 'cbam', 'far'],
                        help='Model type: standard, light, attention (SE), cbam (SE+spatial), far (FAR attention)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--train_samples', type=int, default=None,
                        help='Number of training samples')
    parser.add_argument('--snr_min', type=float, default=None,
                        help='Minimum training SNR')
    parser.add_argument('--snr_max', type=float, default=None,
                        help='Maximum training SNR')
    parser.add_argument('--loss', type=str, default='l1',
                        choices=['l1', 'l2', 'complex'],
                        help='Loss type: l1 (MAE), l2 (Euclidean), complex (polar complex distance)')
    
    args = parser.parse_args()
    
    snr_range = None
    if args.snr_min is not None and args.snr_max is not None:
        snr_range = (args.snr_min, args.snr_max)
    
    train(
        model_type=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        snr_train_range=snr_range,
        loss_type=args.loss
    )
