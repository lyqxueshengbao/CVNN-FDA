"""
FDA-CVNN 一键运行入口
"""
import argparse
import torch
import numpy as np

import config as cfg


def test_config():
    """测试配置"""
    print("\n" + "=" * 60)
    print("1. 测试配置")
    print("=" * 60)
    
    from config import delta_f, c, r_max, R_max, M, N
    
    print(f"载频 f0 = {cfg.f0/1e9:.1f} GHz")
    print(f"频率增量 delta_f = {delta_f/1e3:.0f} kHz")
    print(f"阵元数 M×N = {M}×{N}")
    print(f"距离范围 [0, {r_max}] m")
    print(f"最大不模糊距离 R_max = {R_max:.0f} m")
    
    if R_max >= r_max:
        print("✓ 物理参数正确，无模糊")
    else:
        print("✗ 警告：存在物理模糊！")
    
    return True


def test_signal():
    """测试信号生成"""
    print("\n" + "=" * 60)
    print("2. 测试信号生成")
    print("=" * 60)
    
    from utils_physics import generate_covariance_matrix, get_steering_vector
    
    # 测试导向矢量
    u = get_steering_vector(1000, 30)
    print(f"导向矢量形状: {u.shape}")
    
    # 测试协方差矩阵
    R = generate_covariance_matrix(1000, 30, snr_db=20)
    print(f"协方差矩阵形状: {R.shape}")
    print(f"实部范围: [{R[0].min():.4f}, {R[0].max():.4f}]")
    
    # 测试不同距离的信号区分度
    print("\n信号区分度测试:")
    r1, r2 = 1000, 1010  # 10米差距
    R1 = generate_covariance_matrix(r1, 30, snr_db=50)  # 高SNR
    R2 = generate_covariance_matrix(r2, 30, snr_db=50)
    
    # 计算相关性
    R1_flat = R1.flatten()
    R2_flat = R2.flatten()
    corr = np.corrcoef(R1_flat, R2_flat)[0, 1]
    print(f"  r={r1}m vs r={r2}m (差10m): 相关系数={corr:.4f}")
    
    return True


def test_layers():
    """测试复数层"""
    print("\n" + "=" * 60)
    print("3. 测试复数网络层")
    print("=" * 60)
    
    from layers_complex import ComplexConv2d, ModReLU, ComplexAvgPool2d
    
    x = torch.randn(2, 2, 8, 50, 50)  # [B, 2, C, H, W]
    
    # 测试卷积
    conv = ComplexConv2d(8, 16, kernel_size=3, padding=1)
    out = conv(x)
    print(f"ComplexConv2d: {x.shape} -> {out.shape}")
    
    # 测试激活
    act = ModReLU(16, bias_init=-0.5)
    out = act(out)
    print(f"ModReLU (bias={act.bias[0].item():.2f}): 输出形状 {out.shape}")
    
    # 测试池化
    pool = ComplexAvgPool2d(2)
    out = pool(out)
    print(f"ComplexAvgPool2d: -> {out.shape}")
    
    return True


def test_model():
    """测试模型"""
    print("\n" + "=" * 60)
    print("4. 测试模型")
    print("=" * 60)
    
    from model import FDA_CVNN
    
    model = FDA_CVNN()
    print(f"模型参数量: {model.count_parameters():,}")
    
    # 测试前向传播
    x = torch.randn(4, 2, 100, 100)
    with torch.no_grad():
        y = model(x)
    
    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")
    print(f"输出范围: [{y.min().item():.4f}, {y.max().item():.4f}]")
    
    return True


def test_dataset():
    """测试数据集"""
    print("\n" + "=" * 60)
    print("5. 测试数据集")
    print("=" * 60)
    
    from dataset import FDADataset, create_dataloaders
    
    # 测试数据集
    dataset = FDADataset(100, snr_db=20, online=False, seed=42)
    x, y = dataset[0]
    print(f"样本形状: x={x.shape}, y={y.shape}")
    
    # 测试DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples=100, val_samples=50, test_samples=50, batch_size=16
    )
    
    for batch_x, batch_y in train_loader:
        print(f"批次形状: x={batch_x.shape}, y={batch_y.shape}")
        break
    
    return True


def quick_train():
    """快速训练测试"""
    print("\n" + "=" * 60)
    print("6. 快速训练测试 (5 epochs)")
    print("=" * 60)
    
    from train import train
    
    model, history = train(
        model_type='light',  # 用轻量级模型
        epochs=5,
        train_samples=500,
        batch_size=32
    )
    
    print(f"\n最终 RMSE_r: {history['val_rmse_r'][-1]:.2f}m")
    print(f"最终 RMSE_θ: {history['val_rmse_theta'][-1]:.2f}°")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='FDA-CVNN 项目入口')
    parser.add_argument('--test', action='store_true', help='运行所有测试')
    parser.add_argument('--train', action='store_true', help='开始训练')
    parser.add_argument('--benchmark', action='store_true', help='运行对比实验')
    parser.add_argument('--quick', action='store_true', help='快速训练测试')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--samples', type=int, default=50000, help='训练样本数')
    parser.add_argument('--batch', type=int, default=512, help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--model', type=str, default='standard', 
                        choices=['standard', 'light', 'attention', 'se', 'cbam', 'far', 'dual'],
                        help='模型类型: standard(原始), se/attention(SE), cbam, far, dual(SE+FAR创新)')
    parser.add_argument('--se_reduction', type=int, default=4, choices=[4, 8, 16],
                        help='注意力模块通道压缩比')
    parser.add_argument('--deep_only', action='store_true',
                        help='只在深层使用注意力，跳过Block1')
    parser.add_argument('--snapshots', type=int, default=None,
                        help='快拍数 L (默认使用 config.py 中的值，如 1, 5, 10, 50)')
    parser.add_argument('--snr', type=int, default=0,
                        help='固定信噪比 (用于 --snapshots-benchmark，默认 0 dB)')
    parser.add_argument('--snapshots-benchmark', action='store_true',
                        help='运行快拍数对比实验 (固定 SNR，对比不同快拍数)')
    parser.add_argument('--random-snapshots', action='store_true',
                        help='训练时随机化快拍数 (L=1~100)，提高对不同快拍数的鲁棒性')
    parser.add_argument('--use-random-model', action='store_true',
                        help='测试时使用 Lrandom 通用模型 (一个模型测所有快拍数)')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='评测时每个条件下的样本数 (默认 500)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FDA-MIMO CVNN 项目")
    print("=" * 60)
    print(f"设备: {cfg.device}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    if args.test:
        # 运行所有测试
        test_config()
        test_signal()
        test_layers()
        test_model()
        test_dataset()
        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        
    elif args.quick:
        # 快速训练测试
        test_config()
        quick_train()
        
    elif args.train:
        # 正式训练
        from train import train
        train(
            model_type=args.model,
            epochs=args.epochs,
            train_samples=args.samples,
            batch_size=args.batch,
            lr=args.lr,
            se_reduction=args.se_reduction,
            deep_only=args.deep_only,
            snapshots=args.snapshots,
            random_snapshots=args.random_snapshots
        )
    
    elif args.benchmark:
        # 运行对比实验 (自动匹配快拍数对应的模型)
        from benchmark import run_benchmark, plot_results
        snr_list, results, L = run_benchmark(L_snapshots=args.snapshots, num_samples=args.num_samples)
        plot_results(snr_list, results, L_snapshots=L)
    
    elif args.snapshots_benchmark:
        # 运行快拍数对比实验 (固定 SNR)
        from benchmark import run_snapshots_benchmark, plot_snapshots_results
        if args.use_random_model:
            # 使用通用模型，可以测更大范围的快拍数
            L_list = [1, 5, 10, 25, 50, 100]
        else:
            # 使用各自训练的模型
            L_list = [1, 5, 10, 15, 20, 25]  # 你训练过的快拍数列表
        L_list, results, snr = run_snapshots_benchmark(
            snr_db=args.snr, 
            L_list=L_list,
            num_samples=args.num_samples,
            use_random_model=args.use_random_model
        )
        plot_snapshots_results(L_list, results, snr)
        
    else:
        # 默认运行测试
        print("\n使用方法:")
        print("  python main.py --test    # 运行所有测试")
        print("  python main.py --quick   # 快速训练测试")
        print("  python main.py --train   # 正式训练 (原始模型)")
        print("  python main.py --train --model dual --snapshots 1   # 单快拍 Dual 模型")
        print("  python main.py --train --model dual --snapshots 50  # 50快拍 Dual 模型")
        print("  python main.py --train --model se --snapshots 10    # 10快拍 SE 模型")
        print("  python main.py --train --model dual --se_reduction 8  # reduction=8")
        print("  python main.py --train --epochs 300 --samples 50000 --batch 64 --lr 1e-4")
        print("")
        print("  python main.py --benchmark                 # 对比实验 (默认快拍数)")
        print("  python main.py --benchmark --snapshots 1   # 单快拍对比实验")
        print("  python main.py --benchmark --snapshots 50  # 50快拍对比实验")
        print("")
        print("  python main.py --snapshots-benchmark           # 快拍数对比 (SNR=0dB)")
        print("  python main.py --snapshots-benchmark --snr -5  # 快拍数对比 (SNR=-5dB)")
        print("  python main.py --snapshots-benchmark --snr 10  # 快拍数对比 (SNR=10dB)")
        print("")
        print("  # 🌟 随机快拍数训练 (提高鲁棒性，一个模型适应所有快拍数)")
        print("  python main.py --train --model dual --random-snapshots --epochs 300")


if __name__ == "__main__":
    main()
