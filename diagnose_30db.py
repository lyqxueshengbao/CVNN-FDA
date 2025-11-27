# -*- coding: utf-8 -*-
"""
30dB 过拟合测试 - 验证模型结构上限
如果30dB下RMSE还是>10m，说明代码有Bug
如果30dB下RMSE<5m，说明模型结构OK，问题在于混合SNR训练策略
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time

from model import CVNN_Improved, CVNN_Pro, get_model
from config import r_min, r_max, theta_min, theta_max
from utils import generate_echo_signal, compute_sample_covariance_matrix, complex_normalize


def generate_data(num_samples, snr, L=500):
    """生成指定SNR的数据"""
    data_list, labels_list = [], []
    for _ in tqdm(range(num_samples), desc=f'生成SNR={snr}dB数据'):
        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(theta_min, theta_max)
        Y = generate_echo_signal([(r, theta)], snr, L)
        R = compute_sample_covariance_matrix(Y)
        R_norm = complex_normalize(R)
        data_list.append(np.stack([np.real(R_norm), np.imag(R_norm)], axis=0).astype(np.float32))
        labels_list.append([r, theta])
    return torch.from_numpy(np.array(data_list)), torch.from_numpy(np.array(labels_list, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description='30dB过拟合测试')
    parser.add_argument('--snr', type=float, default=30, help='测试SNR (dB)')
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--val_size', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='medium', 
                        choices=['cvnn', 'medium', 'pro', 'real'],
                        help='模型类型: cvnn(~300K), medium(~1.5M), pro(~19M), real')
    parser.add_argument('--multi_gpu', action='store_true', help='使用多GPU')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 检测GPU数量
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = args.multi_gpu and num_gpus > 1
    
    print('='*70)
    print('30dB 过拟合测试 - 验证模型结构上限')
    print('='*70)
    print(f'设备: {device}')
    if torch.cuda.is_available():
        for i in range(num_gpus):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        if use_multi_gpu:
            print(f'使用 DataParallel ({num_gpus} GPUs)')
    print(f'模型: {args.model}')
    print(f'SNR: {args.snr}dB (高SNR，几乎无噪声)')
    print(f'训练集: {args.train_size}, 验证集: {args.val_size}')
    print(f'Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}')
    print()

    # 生成数据
    print('生成数据...')
    train_data, train_labels = generate_data(args.train_size, args.snr)
    val_data, val_labels = generate_data(args.val_size, args.snr)

    # 归一化标签
    train_labels_norm = train_labels.clone()
    train_labels_norm[:, 0] = (train_labels[:, 0] - r_min) / (r_max - r_min)
    train_labels_norm[:, 1] = (train_labels[:, 1] - theta_min) / (theta_max - theta_min)

    train_loader = DataLoader(
        TensorDataset(train_data, train_labels_norm, train_labels), 
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        TensorDataset(val_data, val_labels), 
        batch_size=args.batch_size, num_workers=4
    )

    # 模型
    model = get_model(args.model).to(device)
    
    # 多GPU支持
    if use_multi_gpu:
        model = nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    num_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数: {num_params:,}')
    print()
    print('开始训练...')
    print('-'*70)

    best_rmse_r = float('inf')
    best_rmse_theta = float('inf')

    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练
        model.train()
        train_loss = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        model.eval()
        r_errs, t_errs = [], []
        with torch.no_grad():
            for x, raw in val_loader:
                x = x.to(device)
                out = model(x)
                r_pred = out[:, 0].cpu() * (r_max - r_min) + r_min
                t_pred = out[:, 1].cpu() * (theta_max - theta_min) + theta_min
                r_errs.extend((r_pred - raw[:, 0]).abs().numpy())
                t_errs.extend((t_pred - raw[:, 1]).abs().numpy())

        rmse_r = np.sqrt(np.mean(np.array(r_errs)**2))
        rmse_t = np.sqrt(np.mean(np.array(t_errs)**2))
        scheduler.step()
        
        elapsed = time.time() - start_time

        marker = ''
        if rmse_r < best_rmse_r:
            best_rmse_r = rmse_r
            best_rmse_theta = rmse_t
            marker = ' *'

        print(f'Epoch {epoch+1:2d}/{args.epochs}: Loss={train_loss:.6f}, '
              f'RMSE_r={rmse_r:6.2f}m, RMSE_θ={rmse_t:5.3f}°, '
              f'Time={elapsed:.1f}s{marker}')

    print('-'*70)
    print()
    print('='*70)
    print('诊断结果:')
    print('='*70)
    print(f'SNR = {args.snr}dB')
    print(f'最佳 RMSE_r: {best_rmse_r:.2f}m')
    print(f'最佳 RMSE_θ: {best_rmse_theta:.3f}°')
    print()
    
    if best_rmse_r < 1:
        print('✅ 完美! 模型结构完全OK，能做到 <1m')
        print('   问题100%在于混合SNR训练策略')
        print('   建议: 采用课程学习 (先高SNR预训练，再低SNR微调)')
    elif best_rmse_r < 5:
        print('✅ 模型结构OK! 能做到 <5m')
        print('   问题在于混合SNR训练策略，建议采用课程学习')
    elif best_rmse_r < 10:
        print('⚠️ 模型基本OK，但有优化空间')
        print('   可能需要更深的网络或更好的输入表示')
    else:
        print('❌ 模型结构有问题或代码有Bug!')
        print('   需要检查: 导向矢量公式、归一化、标签生成')
    print('='*70)


if __name__ == '__main__':
    main()
