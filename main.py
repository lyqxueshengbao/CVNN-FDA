# -*- coding: utf-8 -*-
"""
FDA-MIMO 雷达参数估计 - 主程序入口
Main Program for FDA-MIMO Radar Range-Angle Estimation using CVNN

完整流程:
1. 数据集创建
2. 模型构建
3. 模型训练
4. 模型评估
5. 结果可视化

使用方法:
    训练模型: python main.py --mode train
    评估模型: python main.py --mode evaluate --model_path ./checkpoints/best_model.pth
    完整流程: python main.py --mode all
"""

import argparse
import os
import sys
import torch
import numpy as np

from config import (
    DEVICE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
    SNR_TRAIN_MIN, SNR_TRAIN_MAX, SNR_TEST_RANGE,
    MODEL_SAVE_PATH, RESULTS_PATH
)
from dataset import create_dataloaders
from model import get_model, count_parameters
from train import Trainer
from evaluate import (
    evaluate_model_vs_snr,
    evaluate_model,
    plot_rmse_vs_snr,
    plot_scatter_comparison,
    plot_error_distribution,
    plot_training_history,
    save_results_to_file,
    create_test_loader_with_snr
)


def train_model(args):
    """训练模型"""
    print("\n" + "=" * 70)
    print(" " * 20 + "开始训练 CVNN 模型")
    print("=" * 70)
    
    # 创建数据加载器
    print("\n[1/5] 创建数据集...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        snr_range=(SNR_TRAIN_MIN, SNR_TRAIN_MAX)
    )
    print(f"  ✓ 训练集: {args.train_size} 样本")
    print(f"  ✓ 验证集: {args.val_size} 样本")
    print(f"  ✓ 测试集: {args.test_size} 样本")
    
    # 创建模型
    print("\n[2/5] 创建模型...")
    model = get_model(
        model_name=args.model,
        dropout_rate=args.dropout,
        use_batchnorm=args.use_bn
    )
    print(f"  ✓ 模型类型: {args.model}")
    print(f"  ✓ 参数数量: {count_parameters(model):,}")
    print(f"  ✓ 设备: {DEVICE}")
    
    # 创建训练器
    print("\n[3/5] 初始化训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lambda_angle=args.lambda_angle,
        device=DEVICE,
        save_path=MODEL_SAVE_PATH,
        use_multi_gpu=args.multi_gpu
    )
    print("  ✓ 优化器: Adam")
    print(f"  ✓ 学习率: {args.lr}")
    print(f"  ✓ 权重衰减: {args.weight_decay}")
    
    # 训练
    print("\n[4/5] 开始训练...")
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        verbose=True
    )
    
    # 绘制训练曲线
    print("\n[5/5] 保存训练曲线...")
    plot_training_history(
        history,
        save_path=os.path.join(RESULTS_PATH, 'training_history.png'),
        show=False
    )
    
    print("\n" + "=" * 70)
    print(" " * 25 + "训练完成!")
    print("=" * 70)
    
    return trainer


def evaluate_model_main(args):
    """评估模型"""
    print("\n" + "=" * 70)
    print(" " * 20 + "评估 CVNN 模型")
    print("=" * 70)
    
    # 加载模型
    print("\n[1/4] 加载模型...")
    model = get_model(
        model_name=args.model,
        dropout_rate=0.0,  # 评估时不使用 dropout
        use_batchnorm=args.use_bn
    )
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        sys.exit(1)
    
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    print(f"  ✓ 模型已加载: {args.model_path}")
    print(f"  ✓ 最佳 Epoch: {checkpoint.get('best_epoch', 'N/A')}")
    print(f"  ✓ 最佳验证损失: {checkpoint.get('best_val_loss', 'N/A'):.6f}")
    
    # 评估不同 SNR 下的性能
    print("\n[2/4] 评估不同 SNR 下的性能...")
    results = evaluate_model_vs_snr(
        model=model,
        snr_range=list(SNR_TEST_RANGE),
        test_size=args.test_size,
        batch_size=args.batch_size,
        device=DEVICE,
        verbose=True
    )
    
    # 绘制 RMSE vs SNR 曲线
    print("\n[3/4] 绘制性能曲线...")
    plot_rmse_vs_snr(
        results,
        save_path=os.path.join(RESULTS_PATH, 'rmse_vs_snr.png'),
        show=False
    )
    
    # 在固定 SNR 下评估详细性能
    print(f"\n[4/4] 在 SNR = {args.eval_snr} dB 下评估详细性能...")
    test_loader = create_test_loader_with_snr(
        test_size=args.test_size,
        snr=args.eval_snr,
        batch_size=args.batch_size
    )
    predictions, targets, rmse_r, rmse_theta = evaluate_model(model, test_loader, DEVICE)
    
    print(f"  ✓ RMSE_r: {rmse_r:.2f} m")
    print(f"  ✓ RMSE_θ: {rmse_theta:.2f}°")
    
    # 绘制散点图和误差分布
    plot_scatter_comparison(
        predictions, targets,
        save_path=os.path.join(RESULTS_PATH, 'scatter_comparison.png'),
        show=False
    )
    
    plot_error_distribution(
        predictions, targets,
        save_path=os.path.join(RESULTS_PATH, 'error_distribution.png'),
        show=False
    )
    
    # 保存结果到文件
    save_results_to_file(
        results, predictions, targets,
        filepath=os.path.join(RESULTS_PATH, 'evaluation_results.txt')
    )
    
    print("\n" + "=" * 70)
    print(" " * 25 + "评估完成!")
    print("=" * 70)


def test_modules():
    """测试各模块"""
    print("\n" + "=" * 70)
    print(" " * 20 + "测试各模块功能")
    print("=" * 70)
    
    print("\n[1/6] 测试 utils 模块...")
    os.system(f'{sys.executable} utils.py')
    
    print("\n[2/6] 测试 complex_layers 模块...")
    os.system(f'{sys.executable} complex_layers.py')
    
    print("\n[3/6] 测试 dataset 模块...")
    os.system(f'{sys.executable} dataset.py')
    
    print("\n[4/6] 测试 model 模块...")
    os.system(f'{sys.executable} model.py')
    
    print("\n[5/6] 测试 train 模块...")
    os.system(f'{sys.executable} train.py')
    
    print("\n[6/6] 测试 evaluate 模块...")
    os.system(f'{sys.executable} evaluate.py')
    
    print("\n" + "=" * 70)
    print(" " * 25 + "所有模块测试完成!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='FDA-MIMO 雷达参数估计 - CVNN 实现',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'evaluate', 'all', 'test'],
                       help='运行模式')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='standard',
                       choices=['standard', 'light', 'deep', 'pro'],
                       help='模型类型')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                       help='模型路径 (用于评估)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout 概率')
    parser.add_argument('--lambda_angle', type=float, default=1.0,
                       help='角度损失权重')
    parser.add_argument('--patience', type=int, default=15,
                       help='早停耐心值')
    parser.add_argument('--use_bn', action='store_true', default=True,
                       help='使用批归一化')
    parser.add_argument('--multi_gpu', action='store_true', default=True,
                       help='使用多GPU训练 (DataParallel)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    
    # 数据集参数
    parser.add_argument('--train_size', type=int, default=TRAIN_SIZE,
                       help='训练集大小')
    parser.add_argument('--val_size', type=int, default=VAL_SIZE,
                       help='验证集大小')
    parser.add_argument('--test_size', type=int, default=TEST_SIZE,
                       help='测试集大小')
    
    # 评估参数
    parser.add_argument('--eval_snr', type=float, default=10.0,
                       help='评估用的固定 SNR [dB]')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 执行对应模式
    if args.mode == 'train':
        trainer = train_model(args)
        
    elif args.mode == 'evaluate':
        evaluate_model_main(args)
        
    elif args.mode == 'all':
        # 完整流程: 训练 + 评估
        trainer = train_model(args)
        args.model_path = os.path.join(MODEL_SAVE_PATH, 'best_model.pth')
        evaluate_model_main(args)
        
    elif args.mode == 'test':
        test_modules()


if __name__ == "__main__":
    main()
