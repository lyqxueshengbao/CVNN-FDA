# -*- coding: utf-8 -*-
"""
FDA-MIMO 雷达参数估计 - 评估和可视化模块
Evaluation and Visualization Module for FDA-MIMO Radar Range-Angle Estimation

包含:
1. 模型评估函数
2. RMSE 计算 (论文公式 5-15, 5-16)
3. RMSE vs SNR 曲线绘制
4. 结果可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from config import (
    DEVICE, RESULTS_PATH, TEST_SIZE, BATCH_SIZE,
    SNR_TEST_RANGE, r_min, r_max, theta_min, theta_max
)
from dataset import create_test_loader_with_snr
from train import compute_rmse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


@torch.no_grad()
def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: torch.device = DEVICE) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备
    
    Returns:
        predictions: 预测值数组, shape (N, 2)
        targets: 真实值数组, shape (N, 2)
        rmse_r: 距离 RMSE [m]
        rmse_theta: 角度 RMSE [degrees]
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for batch_R, batch_label, batch_raw in test_loader:
        batch_R = batch_R.to(device)
        outputs = model(batch_R)
        
        all_preds.append(outputs.cpu())
        all_targets.append(batch_raw)
    
    predictions = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    # 反归一化预测值
    pred_r = predictions[:, 0] * (r_max - r_min) + r_min
    pred_theta = predictions[:, 1] * (theta_max - theta_min) + theta_min
    
    # 组合为完整预测
    predictions_denorm = np.column_stack([pred_r, pred_theta])
    
    # 计算 RMSE
    rmse_r, rmse_theta = compute_rmse(predictions, targets, denormalize=True)
    
    return predictions_denorm, targets, rmse_r, rmse_theta


def evaluate_model_vs_snr(model: nn.Module,
                          snr_range: List[float],
                          test_size: int = TEST_SIZE,
                          batch_size: int = BATCH_SIZE,
                          device: torch.device = DEVICE,
                          verbose: bool = True) -> Dict[str, List[float]]:
    """
    评估模型在不同 SNR 下的性能
    
    论文对应: 第5章实验部分,绘制 RMSE vs SNR 曲线
    
    Args:
        model: 训练好的模型
        snr_range: SNR 范围列表 [dB]
        test_size: 每个 SNR 的测试样本数
        batch_size: 批大小
        device: 设备
        verbose: 是否打印详细信息
    
    Returns:
        results: 包含 SNR、RMSE_r、RMSE_theta 的字典
    """
    model.eval()
    
    results = {
        'snr': [],
        'rmse_r': [],
        'rmse_theta': []
    }
    
    if verbose:
        print("=" * 60)
        print("评估模型在不同 SNR 下的性能")
        print("=" * 60)
    
    for snr in tqdm(snr_range, desc="Testing SNR"):
        # 创建该 SNR 下的测试数据
        test_loader = create_test_loader_with_snr(
            test_size=test_size,
            snr=snr,
            batch_size=batch_size
        )
        
        # 评估
        _, _, rmse_r, rmse_theta = evaluate_model(model, test_loader, device)
        
        # 记录结果
        results['snr'].append(snr)
        results['rmse_r'].append(rmse_r)
        results['rmse_theta'].append(rmse_theta)
        
        if verbose:
            print(f"  SNR = {snr:>3} dB | RMSE_r: {rmse_r:>6.2f} m | RMSE_θ: {rmse_theta:>5.2f}°")
    
    if verbose:
        print("=" * 60)
    
    return results


def plot_rmse_vs_snr(results: Dict[str, List[float]],
                     save_path: Optional[str] = None,
                     show: bool = True):
    """
    绘制 RMSE vs SNR 曲线
    
    论文对应: 图 5.3 等性能曲线
    
    Args:
        results: evaluate_model_vs_snr 的返回结果
        save_path: 保存路径
        show: 是否显示图像
    """
    snr = results['snr']
    rmse_r = results['rmse_r']
    rmse_theta = results['rmse_theta']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 距离 RMSE
    ax1.plot(snr, rmse_r, 'o-', linewidth=2, markersize=8, color='#2E86C1', label='CVNN')
    ax1.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('距离 RMSE (m)', fontsize=12, fontweight='bold')
    ax1.set_title('距离估计性能 vs SNR', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    
    # 角度 RMSE
    ax2.plot(snr, rmse_theta, 's-', linewidth=2, markersize=8, color='#E74C3C', label='CVNN')
    ax2.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('角度 RMSE (°)', fontsize=12, fontweight='bold')
    ax2.set_title('角度估计性能 vs SNR', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_scatter_comparison(predictions: np.ndarray,
                           targets: np.ndarray,
                           save_path: Optional[str] = None,
                           show: bool = True):
    """
    绘制预测值 vs 真实值散点图
    
    Args:
        predictions: 预测值, shape (N, 2)
        targets: 真实值, shape (N, 2)
        save_path: 保存路径
        show: 是否显示图像
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 距离散点图
    ax1.scatter(targets[:, 0], predictions[:, 0], alpha=0.5, s=20, c='#2E86C1')
    ax1.plot([r_min, r_max], [r_min, r_max], 'r--', linewidth=2, label='理想预测')
    ax1.set_xlabel('真实距离 (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('预测距离 (m)', fontsize=12, fontweight='bold')
    ax1.set_title('距离预测对比', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 角度散点图
    ax2.scatter(targets[:, 1], predictions[:, 1], alpha=0.5, s=20, c='#E74C3C')
    ax2.plot([theta_min, theta_max], [theta_min, theta_max], 'r--', linewidth=2, label='理想预测')
    ax2.set_xlabel('真实角度 (°)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('预测角度 (°)', fontsize=12, fontweight='bold')
    ax2.set_title('角度预测对比', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_error_distribution(predictions: np.ndarray,
                           targets: np.ndarray,
                           save_path: Optional[str] = None,
                           show: bool = True):
    """
    绘制误差分布直方图
    
    Args:
        predictions: 预测值, shape (N, 2)
        targets: 真实值, shape (N, 2)
        save_path: 保存路径
        show: 是否显示图像
    """
    errors_r = predictions[:, 0] - targets[:, 0]
    errors_theta = predictions[:, 1] - targets[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 距离误差分布
    ax1.hist(errors_r, bins=50, alpha=0.7, color='#2E86C1', edgecolor='black')
    ax1.axvline(0, color='r', linestyle='--', linewidth=2, label='零误差')
    ax1.set_xlabel('距离误差 (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('频数', fontsize=12, fontweight='bold')
    ax1.set_title(f'距离误差分布 (均值: {np.mean(errors_r):.2f} m, 标准差: {np.std(errors_r):.2f} m)',
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 角度误差分布
    ax2.hist(errors_theta, bins=50, alpha=0.7, color='#E74C3C', edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='零误差')
    ax2.set_xlabel('角度误差 (°)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('频数', fontsize=12, fontweight='bold')
    ax2.set_title(f'角度误差分布 (均值: {np.mean(errors_theta):.2f}°, 标准差: {np.std(errors_theta):.2f}°)',
                 fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_training_history(history: Dict[str, List[float]],
                         save_path: Optional[str] = None,
                         show: bool = True):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
        show: 是否显示图像
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('训练和验证损失', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 距离 RMSE 曲线
    axes[0, 1].plot(epochs, history['train_rmse_r'], 'b-', label='训练 RMSE_r', linewidth=2)
    axes[0, 1].plot(epochs, history['val_rmse_r'], 'r-', label='验证 RMSE_r', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('RMSE (m)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('距离 RMSE', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 角度 RMSE 曲线
    axes[1, 0].plot(epochs, history['train_rmse_theta'], 'b-', label='训练 RMSE_θ', linewidth=2)
    axes[1, 0].plot(epochs, history['val_rmse_theta'], 'r-', label='验证 RMSE_θ', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('RMSE (°)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('角度 RMSE', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 学习率曲线
    axes[1, 1].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('学习率变化', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def save_results_to_file(results: Dict,
                        predictions: np.ndarray,
                        targets: np.ndarray,
                        filepath: str):
    """
    保存评估结果到文件
    
    Args:
        results: SNR 评估结果
        predictions: 预测值
        targets: 真实值
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("FDA-MIMO 雷达参数估计 - CVNN 评估结果\n")
        f.write("=" * 60 + "\n\n")
        
        # SNR 性能
        if results:
            f.write("1. 不同 SNR 下的性能:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'SNR (dB)':<10} {'RMSE_r (m)':<15} {'RMSE_θ (°)':<15}\n")
            f.write("-" * 60 + "\n")
            for i in range(len(results['snr'])):
                f.write(f"{results['snr'][i]:<10} {results['rmse_r'][i]:<15.2f} {results['rmse_theta'][i]:<15.2f}\n")
            f.write("\n")
        
        # 整体统计
        f.write("2. 整体预测统计:\n")
        f.write("-" * 60 + "\n")
        f.write(f"样本数: {len(predictions)}\n")
        f.write(f"距离 RMSE: {np.sqrt(np.mean((predictions[:, 0] - targets[:, 0])**2)):.2f} m\n")
        f.write(f"角度 RMSE: {np.sqrt(np.mean((predictions[:, 1] - targets[:, 1])**2)):.2f}°\n")
        f.write(f"距离误差均值: {np.mean(predictions[:, 0] - targets[:, 0]):.2f} m\n")
        f.write(f"距离误差标准差: {np.std(predictions[:, 0] - targets[:, 0]):.2f} m\n")
        f.write(f"角度误差均值: {np.mean(predictions[:, 1] - targets[:, 1]):.2f}°\n")
        f.write(f"角度误差标准差: {np.std(predictions[:, 1] - targets[:, 1]):.2f}°\n")
        f.write("\n")
        
        f.write("=" * 60 + "\n")
    
    print(f"结果已保存到: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("评估和可视化模块测试")
    print("=" * 60)
    
    # 生成模拟数据
    N = 1000
    predictions = np.column_stack([
        np.random.uniform(r_min, r_max, N),
        np.random.uniform(theta_min, theta_max, N)
    ])
    targets = predictions + np.random.randn(N, 2) * [50, 2]  # 添加噪声
    
    # 测试 RMSE 计算
    rmse_r = np.sqrt(np.mean((predictions[:, 0] - targets[:, 0])**2))
    rmse_theta = np.sqrt(np.mean((predictions[:, 1] - targets[:, 1])**2))
    print(f"\nRMSE_r: {rmse_r:.2f} m")
    print(f"RMSE_θ: {rmse_theta:.2f}°")
    
    # 测试绘图功能
    print("\n测试绘图功能...")
    
    # 散点图
    plot_scatter_comparison(predictions, targets, show=False)
    print("  ✓ 散点图测试完成")
    
    # 误差分布
    plot_error_distribution(predictions, targets, show=False)
    print("  ✓ 误差分布图测试完成")
    
    # RMSE vs SNR 曲线
    mock_results = {
        'snr': list(range(-10, 25, 5)),
        'rmse_r': [200, 150, 100, 70, 50, 40, 35],
        'rmse_theta': [8, 6, 4, 3, 2.5, 2, 1.8]
    }
    plot_rmse_vs_snr(mock_results, show=False)
    print("  ✓ RMSE vs SNR 曲线测试完成")
    
    print("\n" + "=" * 60)
    print("评估模块测试完成!")
    print("=" * 60)
