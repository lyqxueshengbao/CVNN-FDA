# -*- coding: utf-8 -*-
"""
FDA-MIMO 雷达参数估计 - 训练模块
Training Module for FDA-MIMO Radar Range-Angle Estimation using CVNN

包含:
1. Trainer 类: 训练器
2. 训练辅助函数
3. 学习率调度器
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    MODEL_SAVE_PATH, RESULTS_PATH,
    r_min, r_max, theta_min, theta_max
)
from utils import denormalize_labels


class RangeAngleLoss(nn.Module):
    """
    距离-角度联合损失函数
    
    Loss = MSE(r_pred, r_gt) + λ * MSE(θ_pred, θ_gt)
    
    由于距离和角度的数值范围不同,使用归一化后的标签计算损失
    """
    
    def __init__(self, lambda_angle: float = 1.0):
        """
        Args:
            lambda_angle: 角度损失权重 (默认1.0,即等权重)
        """
        super(RangeAngleLoss, self).__init__()
        self.lambda_angle = lambda_angle
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        
        Args:
            pred: 预测值, shape (batch, 2), [r_norm, θ_norm]
            target: 目标值, shape (batch, 2), [r_norm, θ_norm]
        
        Returns:
            loss: 总损失
        """
        # 分离距离和角度
        r_pred, theta_pred = pred[:, 0], pred[:, 1]
        r_target, theta_target = target[:, 0], target[:, 1]
        
        # 计算各自的MSE损失
        loss_r = self.mse(r_pred, r_target)
        loss_theta = self.mse(theta_pred, theta_target)
        
        # 加权总损失
        total_loss = loss_r + self.lambda_angle * loss_theta
        
        return total_loss


class EarlyStopping:
    """
    早停机制
    
    当验证损失不再改善时停止训练
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = True):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善量
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """
    CVNN 训练器
    
    负责模型训练、验证、保存和加载
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 learning_rate: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY,
                 lambda_angle: float = 1.0,
                 device: torch.device = DEVICE,
                 save_path: str = MODEL_SAVE_PATH):
        """
        初始化训练器
        
        Args:
            model: CVNN 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            learning_rate: 学习率
            weight_decay: 权重衰减
            lambda_angle: 角度损失权重
            device: 训练设备
            save_path: 模型保存路径
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_path = save_path
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(RESULTS_PATH, exist_ok=True)
        
        # 损失函数
        self.criterion = RangeAngleLoss(lambda_angle=lambda_angle)
        
        # 优化器 (Adam 支持复数参数的梯度)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse_r': [],
            'train_rmse_theta': [],
            'val_rmse_r': [],
            'val_rmse_theta': [],
            'lr': []
        }
        
        # 最佳模型
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self) -> Tuple[float, float, float]:
        """
        训练一个epoch
        
        Returns:
            avg_loss: 平均损失
            rmse_r: 距离RMSE (真实单位)
            rmse_theta: 角度RMSE (真实单位)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_R, batch_label, batch_raw in pbar:
            # 移动数据到设备
            batch_R = batch_R.to(self.device)
            batch_label = batch_label.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_R)
            
            # 计算损失
            loss = self.criterion(outputs, batch_label)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item() * batch_R.size(0)
            all_preds.append(outputs.detach().cpu())
            all_targets.append(batch_raw)  # 使用原始标签计算RMSE
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader.dataset)
        
        # 计算RMSE (真实单位)
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # 反归一化预测值
        pred_r = all_preds[:, 0] * (r_max - r_min) + r_min
        pred_theta = all_preds[:, 1] * (theta_max - theta_min) + theta_min
        
        # 真实值
        true_r = all_targets[:, 0]
        true_theta = all_targets[:, 1]
        
        # 计算RMSE
        rmse_r = np.sqrt(np.mean((pred_r - true_r) ** 2))
        rmse_theta = np.sqrt(np.mean((pred_theta - true_theta) ** 2))
        
        return avg_loss, rmse_r, rmse_theta
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """
        验证模型
        
        Returns:
            avg_loss: 平均损失
            rmse_r: 距离RMSE (真实单位)
            rmse_theta: 角度RMSE (真实单位)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_R, batch_label, batch_raw in self.val_loader:
            batch_R = batch_R.to(self.device)
            batch_label = batch_label.to(self.device)
            
            outputs = self.model(batch_R)
            loss = self.criterion(outputs, batch_label)
            
            total_loss += loss.item() * batch_R.size(0)
            all_preds.append(outputs.cpu())
            all_targets.append(batch_raw)
        
        avg_loss = total_loss / len(self.val_loader.dataset)
        
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        pred_r = all_preds[:, 0] * (r_max - r_min) + r_min
        pred_theta = all_preds[:, 1] * (theta_max - theta_min) + theta_min
        
        true_r = all_targets[:, 0]
        true_theta = all_targets[:, 1]
        
        rmse_r = np.sqrt(np.mean((pred_r - true_r) ** 2))
        rmse_theta = np.sqrt(np.mean((pred_theta - true_theta) ** 2))
        
        return avg_loss, rmse_r, rmse_theta
    
    def train(self,
              num_epochs: int = NUM_EPOCHS,
              early_stopping_patience: int = 15,
              verbose: bool = True) -> Dict:
        """
        完整训练流程
        
        Args:
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            verbose: 是否打印详细信息
        
        Returns:
            history: 训练历史记录
        """
        print("=" * 60)
        print("开始训练 CVNN 模型")
        print(f"设备: {self.device}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print("=" * 60)
        
        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=verbose)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_rmse_r, train_rmse_theta = self.train_epoch()
            
            # 验证
            val_loss, val_rmse_r, val_rmse_theta = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse_r'].append(train_rmse_r)
            self.history['train_rmse_theta'].append(train_rmse_theta)
            self.history['val_rmse_r'].append(val_rmse_r)
            self.history['val_rmse_theta'].append(val_rmse_theta)
            self.history['lr'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # 打印信息
            if verbose:
                print(f"\nEpoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_loss:.6f} | RMSE_r: {train_rmse_r:.2f}m | RMSE_θ: {train_rmse_theta:.2f}°")
                print(f"  Val Loss:   {val_loss:.6f} | RMSE_r: {val_rmse_r:.2f}m | RMSE_θ: {val_rmse_theta:.2f}°")
                print(f"  LR: {current_lr:.2e}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.save_model('best_model.pth')
                if verbose:
                    print(f"  ✓ 保存最佳模型 (epoch {self.best_epoch})")
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_model(f'model_epoch_{epoch+1}.pth')
            
            # 早停检查
            if early_stopping(val_loss):
                print(f"\n早停触发! 训练在 epoch {epoch+1} 停止")
                break
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"训练完成! 总用时: {total_time/60:.1f} 分钟")
        print(f"最佳模型: Epoch {self.best_epoch}, Val Loss: {self.best_val_loss:.6f}")
        print("=" * 60)
        
        # 保存最终模型
        self.save_model('final_model.pth')
        
        # 保存训练历史
        self.save_history()
        
        return self.history
    
    def save_model(self, filename: str):
        """保存模型"""
        filepath = os.path.join(self.save_path, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history
        }, filepath)
    
    def load_model(self, filename: str):
        """加载模型"""
        filepath = os.path.join(self.save_path, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint['history']
        print(f"加载模型: {filename}")
    
    def save_history(self):
        """保存训练历史"""
        import json
        filepath = os.path.join(RESULTS_PATH, 'training_history.json')
        
        # 转换 numpy 类型为 Python 原生类型
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                history_serializable[key] = [float(v) if hasattr(v, 'item') else v for v in value]
            else:
                history_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        print(f"训练历史已保存到: {filepath}")


def compute_rmse(predictions: np.ndarray, 
                 targets: np.ndarray,
                 denormalize: bool = True) -> Tuple[float, float]:
    """
    计算 RMSE
    
    论文公式 (5-15), (5-16):
    RMSE_r = sqrt(1/N * Σ(r_pred - r_true)^2)
    RMSE_θ = sqrt(1/N * Σ(θ_pred - θ_true)^2)
    
    Args:
        predictions: 预测值, shape (N, 2)
        targets: 真实值, shape (N, 2)
        denormalize: 是否反归一化
    
    Returns:
        rmse_r: 距离 RMSE [m]
        rmse_theta: 角度 RMSE [degrees]
    """
    if denormalize:
        pred_r = predictions[:, 0] * (r_max - r_min) + r_min
        pred_theta = predictions[:, 1] * (theta_max - theta_min) + theta_min
    else:
        pred_r = predictions[:, 0]
        pred_theta = predictions[:, 1]
    
    true_r = targets[:, 0]
    true_theta = targets[:, 1]
    
    rmse_r = np.sqrt(np.mean((pred_r - true_r) ** 2))
    rmse_theta = np.sqrt(np.mean((pred_theta - true_theta) ** 2))
    
    return rmse_r, rmse_theta


if __name__ == "__main__":
    # 简单测试
    print("=" * 60)
    print("训练模块测试")
    print("=" * 60)
    
    # 测试损失函数
    print("\n1. 测试 RangeAngleLoss:")
    criterion = RangeAngleLoss(lambda_angle=1.0)
    pred = torch.rand(32, 2)
    target = torch.rand(32, 2)
    loss = criterion(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    
    # 测试早停
    print("\n2. 测试 EarlyStopping:")
    es = EarlyStopping(patience=3, verbose=False)
    losses = [1.0, 0.9, 0.85, 0.85, 0.86, 0.87, 0.88]
    for i, l in enumerate(losses):
        stop = es(l)
        print(f"   Epoch {i+1}: loss={l}, stop={stop}")
        if stop:
            break
    
    # 测试 RMSE 计算
    print("\n3. 测试 RMSE 计算:")
    pred = np.array([[0.5, 0.5], [0.6, 0.4]])
    target = np.array([[1000, 0], [1200, -12]])  # 真实值
    rmse_r, rmse_theta = compute_rmse(pred, target, denormalize=True)
    print(f"   RMSE_r: {rmse_r:.2f} m")
    print(f"   RMSE_θ: {rmse_theta:.2f}°")
    
    print("\n" + "=" * 60)
    print("训练模块测试完成!")
    print("=" * 60)
