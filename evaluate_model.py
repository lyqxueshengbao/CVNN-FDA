# -*- coding: utf-8 -*-
"""
FDA-MIMO CVNN 模型评测脚本
支持指定SNR进行测试
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm

from config import r_min, r_max, theta_min, theta_max, MN
from model import CVNN_Improved
from utils import generate_echo_signal, compute_sample_covariance_matrix, complex_normalize


def generate_test_data(num_samples, snr, L=500):
    """生成测试数据"""
    print(f"生成测试数据: {num_samples}样本, SNR={snr}dB")
    
    data_list = []
    labels_list = []
    
    for _ in tqdm(range(num_samples), desc="生成中"):
        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(theta_min, theta_max)
        
        Y = generate_echo_signal([(r, theta)], snr, L)
        R = compute_sample_covariance_matrix(Y)
        R_norm = complex_normalize(R)
        
        # 转换为2通道实数
        R_real = np.real(R_norm).astype(np.float32)
        R_imag = np.imag(R_norm).astype(np.float32)
        data_list.append(np.stack([R_real, R_imag], axis=0))
        labels_list.append([r, theta])
    
    data = torch.from_numpy(np.array(data_list))
    labels = torch.from_numpy(np.array(labels_list, dtype=np.float32))
    
    return data, labels


def evaluate_model(model, data, labels, device, batch_size=64):
    """评测模型"""
    model.eval()
    
    r_errors = []
    theta_errors = []
    
    num_samples = data.shape[0]
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_data = data[i:i+batch_size].to(device)
            batch_labels = labels[i:i+batch_size]
            
            out = model(batch_data)
            
            # 反归一化
            r_pred = out[:, 0].cpu() * (r_max - r_min) + r_min
            theta_pred = out[:, 1].cpu() * (theta_max - theta_min) + theta_min
            
            r_errors.extend((r_pred - batch_labels[:, 0]).abs().numpy())
            theta_errors.extend((theta_pred - batch_labels[:, 1]).abs().numpy())
    
    r_errors = np.array(r_errors)
    theta_errors = np.array(theta_errors)
    
    rmse_r = np.sqrt(np.mean(r_errors**2))
    rmse_theta = np.sqrt(np.mean(theta_errors**2))
    mae_r = np.mean(r_errors)
    mae_theta = np.mean(theta_errors)
    
    # 百分位数
    r_50 = np.percentile(r_errors, 50)
    r_90 = np.percentile(r_errors, 90)
    r_95 = np.percentile(r_errors, 95)
    
    return {
        'rmse_r': rmse_r,
        'rmse_theta': rmse_theta,
        'mae_r': mae_r,
        'mae_theta': mae_theta,
        'r_50': r_50,
        'r_90': r_90,
        'r_95': r_95,
        'r_errors': r_errors,
        'theta_errors': theta_errors
    }


def main():
    parser = argparse.ArgumentParser(description='评测FDA-MIMO CVNN模型')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='模型检查点路径')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='测试样本数')
    parser.add_argument('--snr', type=float, default=10.0,
                        help='测试SNR (dB)')
    parser.add_argument('--snr_range', type=str, default=None,
                        help='SNR范围，如 "-10,20"，优先于--snr')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    model = CVNN_Improved(input_channels=2, output_size=2)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # 处理DataParallel前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 确定SNR
    if args.snr_range:
        snr_min, snr_max = map(float, args.snr_range.split(','))
        snr_list = [snr_min + (snr_max - snr_min) * i / 6 for i in range(7)]
        print(f"\n测试多个SNR: {[f'{s:.1f}dB' for s in snr_list]}")
    else:
        snr_list = [args.snr]
        print(f"\n测试SNR: {args.snr}dB")
    
    print("=" * 70)
    
    results = {}
    for snr in snr_list:
        data, labels = generate_test_data(args.num_samples, snr)
        metrics = evaluate_model(model, data, labels, device, args.batch_size)
        results[snr] = metrics
        
        print(f"\nSNR = {snr:5.1f} dB:")
        print(f"  RMSE_r  = {metrics['rmse_r']:6.2f} m")
        print(f"  RMSE_θ  = {metrics['rmse_theta']:5.2f}°")
        print(f"  MAE_r   = {metrics['mae_r']:6.2f} m")
        print(f"  MAE_θ   = {metrics['mae_theta']:5.2f}°")
        print(f"  r误差分位: 50%={metrics['r_50']:.1f}m, 90%={metrics['r_90']:.1f}m, 95%={metrics['r_95']:.1f}m")
    
    print("\n" + "=" * 70)
    
    # 汇总表
    if len(snr_list) > 1:
        print("\n汇总表:")
        print("-" * 50)
        print(f"{'SNR (dB)':>10} | {'RMSE_r (m)':>12} | {'RMSE_θ (°)':>12}")
        print("-" * 50)
        for snr in snr_list:
            m = results[snr]
            print(f"{snr:>10.1f} | {m['rmse_r']:>12.2f} | {m['rmse_theta']:>12.2f}")
        print("-" * 50)


if __name__ == '__main__':
    main()
