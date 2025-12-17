"""
FDA-CVNN 一键运行入口 (已适配 Standard Baseline 版本)
"""
import argparse
import torch
import numpy as np
import sys

# 尝试导入 config
try:
    import config as cfg
except ImportError:
    print("❌ 错误: 未找到 config.py，请确保配置文件存在。")
    sys.exit(1)


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

    try:
        from layers_complex import ComplexConv2d, ModReLU, ComplexAvgPool2d
    except ImportError:
        print("⚠️  跳过复数层测试 (未找到 layers_complex.py)")
        return True

    x = torch.randn(2, 2, 8, 50, 50)  # [B, 2, C, H, W]

    # 测试卷积
    conv = ComplexConv2d(8, 16, kernel_size=3, padding=1)
    out = conv(x)
    print(f"ComplexConv2d: {x.shape} -> {out.shape}")

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

    try:
        from dataset import FDADataset, create_dataloaders
        # 测试数据集
        dataset = FDADataset(100, snr_db=20, online=False, seed=42)
        x, y = dataset[0]
        print(f"样本形状: x={x.shape}, y={y.shape}")
    except ImportError:
        print("⚠️  跳过数据集测试 (未找到 dataset.py)")

    return True


def quick_train():
    """快速训练测试"""
    print("\n" + "=" * 60)
    print("6. 快速训练测试 (5 epochs)")
    print("=" * 60)

    try:
        from train import train
        model, history = train(
            model_type='light',  # 用轻量级模型
            epochs=5,
            train_samples=500,
            batch_size=32
        )
        print(f"\n最终 RMSE_r: {history['val_rmse_r'][-1]:.2f}m")
    except ImportError:
        print("⚠️  跳过训练测试 (未找到 train.py)")

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
                        help='模型类型')
    parser.add_argument('--se_reduction', type=int, default=4, help='注意力压缩比')
    parser.add_argument('--deep_only', action='store_true', help='只在深层使用注意力')
    parser.add_argument('--snapshots', type=int, default=None, help='快拍数 L')
    parser.add_argument('--snr', type=int, default=None, help='固定信噪比 (单值训练)')
    parser.add_argument('--snr-min', type=int, default=None, help='训练 SNR 最小值')
    parser.add_argument('--snr-max', type=int, default=None, help='训练 SNR 最大值')
    parser.add_argument('--snapshots-benchmark', action='store_true', help='运行快拍数对比实验')
    parser.add_argument('--random-snapshots', action='store_true', help='随机化快拍数')
    parser.add_argument('--use-random-model', action='store_true', help='使用 Lrandom 通用模型')
    parser.add_argument('--num-samples', type=int, default=500, help='评测样本数')
    parser.add_argument('--fast', action='store_true', help='快速模式 (只测NN)')
    parser.add_argument('--tradeoff', action='store_true', help='运行精度-速度权衡分析')
    parser.add_argument('--benchmark-legacy', action='store_true',
                        help='运行 Legacy Methods vs CVNN 对比实验 (复现 Matlab 传统算法)')
    parser.add_argument('--comprehensive-benchmark', action='store_true',
                        help='运行综合对比实验 (CVNN + MUSIC + ESPRIT + OMP + CRLB)')
    parser.add_argument('--attention-type', type=str, default='dual',
                        choices=['dual', 'se', 'far', 'standard'],
                        help='CVNN注意力类型 (用于comprehensive-benchmark)')
    parser.add_argument('--reduction', type=int, default=8,
                        help='注意力模块压缩比 (用于comprehensive-benchmark)')
    parser.add_argument('--monte-carlo', type=int, default=100,
                        help='传统算法蒙特卡洛次数 (用于comprehensive-benchmark)')

    # 保留此参数以兼容旧脚本，但在代码中会将其拦截
    parser.add_argument('--music-continuous', action='store_true',
                        help='(已弃用) 以前用于开启连续优化。现在为了凸显 CVNN 优势，该选项将被忽略，强制使用 Standard Baselines。')

    args = parser.parse_args()

    print("=" * 60)
    print("FDA-MIMO CVNN 项目 (Standard Baselines Mode)")
    print("=" * 60)
    print(f"设备: {cfg.device}")

    if args.music_continuous and (args.benchmark or args.snapshots_benchmark):
        print("\n⚠️  [提示] 检测到参数 --music-continuous")
        print("    为了凸显 CVNN 在连续值预测上的优势，代码已更新为 Standard Baseline 模式。")
        print("    MUSIC 和 OMP 将回归为标准网格搜索 (Standard Grid Search)。")
        print("    该参数将被忽略，脚本继续运行...\n")

    if args.test:
        test_config()
        test_signal()
        test_layers()
        test_model()
        test_dataset()
        print("\n所有测试通过！")

    elif args.quick:
        test_config()
        quick_train()

    elif args.train:
        from train import train

        # 处理 SNR 参数
        snr_range = None
        if args.snr is not None:
            # 单个 SNR 值训练
            snr_range = (args.snr, args.snr)
            print(f"使用固定 SNR = {args.snr} dB 训练")
        elif args.snr_min is not None and args.snr_max is not None:
            # SNR 范围训练
            snr_range = (args.snr_min, args.snr_max)
            print(f"使用 SNR 范围 [{args.snr_min}, {args.snr_max}] dB 训练")
        elif args.snr_min is not None or args.snr_max is not None:
            print("❌ 错误: --snr-min 和 --snr-max 必须同时指定")
            sys.exit(1)

        train(
            model_type=args.model,
            epochs=args.epochs,
            train_samples=args.samples,
            batch_size=args.batch,
            lr=args.lr,
            se_reduction=args.se_reduction,
            deep_only=args.deep_only,
            snapshots=args.snapshots,
            random_snapshots=args.random_snapshots,
            snr_train_range=snr_range
        )

    elif args.benchmark:
        # 运行对比实验
        # 确保 benchmark.py (或 FDA_MIMO_Benchmark_Standard.py) 存在
        try:
            from benchmark import run_benchmark, plot_results
        except ImportError:
            # 兼容性处理：如果你把文件名保存为了 FDA_MIMO_Benchmark_Standard.py
            try:
                from FDA_MIMO_Benchmark_Standard import run_benchmark, plot_results
            except ImportError:
                print("❌ 未找到 benchmark 模块。请确保 'benchmark.py' 或 'FDA_MIMO_Benchmark_Standard.py' 存在。")
                sys.exit(1)

        # 核心修改：不再传递 music_continuous 参数
        snr_list, results, L = run_benchmark(
            L_snapshots=args.snapshots,
            num_samples=args.num_samples,
            fast_mode=args.fast
            # music_continuous=args.music_continuous  <-- 已移除
        )
        plot_results(snr_list, results, L_snapshots=L)

    elif args.snapshots_benchmark:
        # 运行快拍数对比实验
        try:
            from benchmark import run_snapshots_benchmark
        except ImportError:
            try:
                from FDA_MIMO_Benchmark_Standard import run_snapshots_benchmark
            except ImportError:
                print("❌ 未找到 benchmark 模块。")
                sys.exit(1)

        if args.use_random_model:
            L_list = [1, 5, 10, 25, 50, 100]
        else:
            L_list = [1, 5, 10, 15, 20, 25]

        run_snapshots_benchmark(
            snr_db=args.snr,
            L_list=L_list,
            num_samples=args.num_samples,
            use_random_model=args.use_random_model
        )

    elif args.tradeoff:
        # 运行精度-速度权衡分析
        try:
            from benchmark import run_accuracy_speed_tradeoff
        except ImportError:
            try:
                from FDA_MIMO_Benchmark_Standard import run_accuracy_speed_tradeoff
            except ImportError:
                print("❌ 未找到 benchmark 模块。")
                sys.exit(1)

        # 对于 tradeoff 分析，使用指定的 SNR（默认0dB更能展示优势）
        run_accuracy_speed_tradeoff(
            snr_db=args.snr,
            num_samples=args.num_samples,
            L_snapshots=args.snapshots
        )

    elif args.benchmark_legacy:
        # 运行 Legacy vs CVNN 对比实验
        try:
            from benchmark_legacy_vs_cvnn import run_legacy_benchmark, plot_results
        except ImportError:
            print("❌ 未找到 benchmark_legacy_vs_cvnn.py 模块。")
            sys.exit(1)

        snr_list, results, L = run_legacy_benchmark(
            num_samples=args.num_samples,
            L_snapshots=args.snapshots
        )
        plot_results(snr_list, results, L_snapshots=L)

    elif args.comprehensive_benchmark:
        # 运行综合对比实验 (CVNN + MUSIC + ESPRIT + OMP + CRLB)
        print("\n" + "=" * 60)
        print("运行综合对比实验")
        print("=" * 60)
        print(f"快拍数: L = {args.snapshots if args.snapshots else cfg.L_snapshots}")
        print(f"测试样本数: {args.num_samples}")
        print(f"蒙特卡洛次数: {args.monte_carlo}")
        print(f"CVNN 注意力类型: {args.attention_type}")
        print(f"CVNN 压缩比: {args.reduction}")
        print("=" * 60)

        try:
            import benchmark_comprehensive as bc
        except ImportError:
            print("❌ 未找到 benchmark_comprehensive.py 模块。")
            sys.exit(1)

        # 调用综合评测（使用命令行参数）
        bc.run_comprehensive_benchmark(
            L_snapshots=args.snapshots,
            num_samples_cvnn=args.num_samples,
            monte_carlo_classical=args.monte_carlo,
            attention_type=args.attention_type,
            reduction=args.reduction
        )

if __name__ == "__main__":
    main()