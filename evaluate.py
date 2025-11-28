"""
FDA-CVNN 多信噪比测试脚本
在不同SNR下评估模型性能
"""
import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm

import config as cfg
from model import FDA_CVNN, FDA_CVNN_Light
from dataset import FDADataset
from torch.utils.data import DataLoader
from utils_physics import denormalize_labels


def evaluate_at_snr(model, snr_db, num_samples=1000, batch_size=64, device='cuda'):
    """
    在指定SNR下评估模型
    
    参数:
        model: 训练好的模型
        snr_db: 信噪比 (dB)
        num_samples: 测试样本数
        batch_size: 批次大小
        device: 设备
    
    返回:
        dict: 评估指标
    """
    # 创建测试数据集 (固定SNR)
    # 确保种子为正整数
    seed = abs(int(cfg.seed + snr_db * 100)) % (2**31)
    dataset = FDADataset(
        num_samples,
        snr_db=snr_db,
        online=False,
        seed=seed
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            all_preds.append(preds.cpu())
            all_labels.append(batch_y)
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # 反归一化
    preds_physical = denormalize_labels(all_preds)
    labels_physical = denormalize_labels(all_labels)
    
    # 计算误差
    r_pred = preds_physical[:, 0]
    r_true = labels_physical[:, 0]
    theta_pred = preds_physical[:, 1]
    theta_true = labels_physical[:, 1]
    
    r_error = r_pred - r_true
    theta_error = theta_pred - theta_true
    
    results = {
        'snr_db': snr_db,
        'num_samples': num_samples,
        # 距离指标
        'rmse_r': float(np.sqrt(np.mean(r_error ** 2))),
        'mae_r': float(np.mean(np.abs(r_error))),
        'std_r': float(np.std(r_error)),
        'max_r': float(np.max(np.abs(r_error))),
        # 角度指标
        'rmse_theta': float(np.sqrt(np.mean(theta_error ** 2))),
        'mae_theta': float(np.mean(np.abs(theta_error))),
        'std_theta': float(np.std(theta_error)),
        'max_theta': float(np.max(np.abs(theta_error))),
    }
    
    return results


def test_multiple_snr(model_path=None, snr_list=None, num_samples=1000, 
                      model_type='standard', save_results=True):
    """
    在多个SNR下测试模型
    
    参数:
        model_path: 模型权重路径
        snr_list: SNR列表 (dB)
        num_samples: 每个SNR的测试样本数
        model_type: 模型类型
        save_results: 是否保存结果
    """
    device = cfg.device
    
    # 默认SNR列表
    if snr_list is None:
        snr_list = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
    
    # 默认模型路径
    if model_path is None:
        model_path = cfg.model_save_path
    
    print("=" * 70)
    print("FDA-CVNN 多信噪比测试")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"模型: {model_path}")
    print(f"SNR范围: {snr_list} dB")
    print(f"每个SNR样本数: {num_samples}")
    print("=" * 70)
    
    # 加载模型
    if model_type == 'light':
        model = FDA_CVNN_Light().to(device)
    else:
        model = FDA_CVNN().to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载模型权重 (Epoch {checkpoint.get('epoch', '?')})")
        else:
            model.load_state_dict(checkpoint)
            print("加载模型权重")
    else:
        print(f"警告: 模型文件不存在 {model_path}，使用随机初始化权重")
    
    # 测试每个SNR
    all_results = []
    
    print("\n" + "-" * 70)
    print(f"{'SNR(dB)':>8} | {'RMSE_r(m)':>12} | {'MAE_r(m)':>12} | {'RMSE_θ(°)':>12} | {'MAE_θ(°)':>12}")
    print("-" * 70)
    
    for snr in snr_list:
        results = evaluate_at_snr(model, snr, num_samples, device=device)
        all_results.append(results)
        
        print(f"{snr:>8.0f} | {results['rmse_r']:>12.2f} | {results['mae_r']:>12.2f} | "
              f"{results['rmse_theta']:>12.2f} | {results['mae_theta']:>12.2f}")
    
    print("-" * 70)
    
    # 计算平均值
    avg_rmse_r = np.mean([r['rmse_r'] for r in all_results])
    avg_rmse_theta = np.mean([r['rmse_theta'] for r in all_results])
    print(f"{'平均':>8} | {avg_rmse_r:>12.2f} | {'-':>12} | {avg_rmse_theta:>12.2f} | {'-':>12}")
    print("=" * 70)
    
    # 保存结果
    if save_results:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "snr_test_results.json")
        
        output = {
            'config': {
                'model_path': model_path,
                'model_type': model_type,
                'num_samples': num_samples,
                'snr_list': snr_list,
            },
            'results': all_results,
            'summary': {
                'avg_rmse_r': avg_rmse_r,
                'avg_rmse_theta': avg_rmse_theta,
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n结果已保存到: {results_path}")
    
    return all_results


def plot_results(results_path="results/snr_test_results.json"):
    """
    绘制SNR-RMSE曲线 (可选，需要matplotlib)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
        return
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    snr_list = [r['snr_db'] for r in results]
    rmse_r = [r['rmse_r'] for r in results]
    rmse_theta = [r['rmse_theta'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 距离RMSE
    ax1.plot(snr_list, rmse_r, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('RMSE (m)', fontsize=12)
    ax1.set_title('Distance Estimation RMSE', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(snr_list)
    
    # 角度RMSE
    ax2.plot(snr_list, rmse_theta, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('RMSE (°)', fontsize=12)
    ax2.set_title('Angle Estimation RMSE', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(snr_list)
    
    plt.tight_layout()
    
    plot_path = "results/snr_performance.png"
    plt.savefig(plot_path, dpi=150)
    print(f"图表已保存到: {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FDA-CVNN 多SNR测试')
    parser.add_argument('--model', type=str, default=None,
                        help='模型权重路径')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'light'],
                        help='模型类型')
    parser.add_argument('--samples', type=int, default=1000,
                        help='每个SNR的测试样本数')
    parser.add_argument('--snr_min', type=float, default=-10,
                        help='最小SNR (dB)')
    parser.add_argument('--snr_max', type=float, default=30,
                        help='最大SNR (dB)')
    parser.add_argument('--snr_step', type=float, default=5,
                        help='SNR步长 (dB)')
    parser.add_argument('--plot', action='store_true',
                        help='绘制结果图')
    
    args = parser.parse_args()
    
    # 生成SNR列表
    snr_list = list(np.arange(args.snr_min, args.snr_max + 0.1, args.snr_step))
    
    # 运行测试
    results = test_multiple_snr(
        model_path=args.model,
        snr_list=snr_list,
        num_samples=args.samples,
        model_type=args.model_type
    )
    
    # 绘图
    if args.plot:
        plot_results()
