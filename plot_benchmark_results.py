"""
从保存的JSON结果中绘制评测对比图
支持自定义SNR范围
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
import os

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def load_results(json_path):
    """加载JSON结果文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def filter_by_snr(data, snr_min, snr_max):
    """根据SNR范围过滤数据"""
    snr_list = np.array(data['snr_db_list'])
    mask = (snr_list >= snr_min) & (snr_list <= snr_max)
    indices = np.where(mask)[0]

    filtered_data = {
        'snr_db_list': snr_list[mask].tolist()
    }

    # 过滤所有指标
    for key in data.keys():
        if key == 'snr_db_list':
            continue
        if isinstance(data[key], list) and len(data[key]) == len(snr_list):
            filtered_data[key] = [data[key][i] for i in indices]

    return filtered_data


def plot_comparison(all_results, crlb_results, save_path, snr_range=None,
                   show_plot=True, figsize=(10, 6), dpi=300):
    """
    绘制所有算法的对比图

    参数:
        all_results: 所有算法的结果列表
        crlb_results: CRLB结果
        save_path: 保存路径
        snr_range: SNR范围 (min, max)，None表示使用全部
        show_plot: 是否显示图形
        figsize: 图形大小
        dpi: 图形分辨率
    """
    os.makedirs(save_path, exist_ok=True)

    # 应用SNR范围过滤
    if snr_range is not None:
        snr_min, snr_max = snr_range
        print(f"\n过滤SNR范围: [{snr_min}, {snr_max}] dB")
        all_results = [filter_by_snr(r, snr_min, snr_max) for r in all_results]
        crlb_results = filter_by_snr(crlb_results, snr_min, snr_max)
        suffix = f"_snr_{snr_min}_{snr_max}"
    else:
        suffix = ""

    # 颜色和标记
    colors = ['b', 'g', 'r', 'm', 'c', 'y']
    markers = ['o', 's', '^', 'd', 'v', '<']
    linestyles = ['-', '-', '-', '-', '-', '--']

    # ========== 角度 RMSE ==========
    plt.figure(figsize=figsize)
    for idx, result in enumerate(all_results):
        snr_list = result['snr_db_list']
        rmse_theta = result['rmse_theta']
        label = result['algorithm']
        plt.semilogy(snr_list, rmse_theta,
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2, markersize=8, label=label)

    # 绘制 CRLB
    plt.semilogy(crlb_results['snr_db_list'], crlb_results['crlb_theta'],
                'k--', linewidth=2, marker='*', markersize=10, label='CRLB')

    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Angle RMSE (deg)', fontsize=14, fontweight='bold')
    plt.title('Angle Estimation Performance Comparison', fontsize=16, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.tight_layout()

    # 保存图片
    plt.savefig(f'{save_path}/angle_comparison{suffix}.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(f'{save_path}/angle_comparison{suffix}.pdf', bbox_inches='tight')
    print(f"✓ 已保存角度对比图: {save_path}/angle_comparison{suffix}.png")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # ========== 距离 RMSE ==========
    plt.figure(figsize=figsize)
    for idx, result in enumerate(all_results):
        snr_list = result['snr_db_list']
        rmse_r = result['rmse_r']
        label = result['algorithm']
        plt.semilogy(snr_list, rmse_r,
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2, markersize=8, label=label)

    # 绘制 CRLB
    plt.semilogy(crlb_results['snr_db_list'], crlb_results['crlb_r'],
                'k--', linewidth=2, marker='*', markersize=10, label='CRLB')

    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Range RMSE (m)', fontsize=14, fontweight='bold')
    plt.title('Range Estimation Performance Comparison', fontsize=16, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.tight_layout()

    # 保存图片
    plt.savefig(f'{save_path}/range_comparison{suffix}.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(f'{save_path}/range_comparison{suffix}.pdf', bbox_inches='tight')
    print(f"✓ 已保存距离对比图: {save_path}/range_comparison{suffix}.png")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # ========== 计算时间对比 ==========
    plt.figure(figsize=figsize)
    algorithms = [r['algorithm'] for r in all_results]
    avg_times = [np.mean(r['avg_time']) * 1000 for r in all_results]  # 转为毫秒

    bars = plt.bar(algorithms, avg_times, color=colors[:len(algorithms)], alpha=0.7, edgecolor='black')
    plt.ylabel('Average Time (ms)', fontsize=14, fontweight='bold')
    plt.title('Computational Time Comparison', fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # 保存图片
    plt.savefig(f'{save_path}/time_comparison{suffix}.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(f'{save_path}/time_comparison{suffix}.pdf', bbox_inches='tight')
    print(f"✓ 已保存时间对比图: {save_path}/time_comparison{suffix}.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def print_summary_table(all_results, crlb_results, snr_value=None):
    """
    打印性能总结表格

    参数:
        all_results: 所有算法的结果列表
        crlb_results: CRLB结果
        snr_value: 指定的SNR值，None表示打印所有
    """
    print("\n" + "=" * 80)
    print("性能总结")
    print("=" * 80)

    snr_list = np.array(crlb_results['snr_db_list'])

    if snr_value is not None:
        # 查找最接近的SNR值
        idx = np.argmin(np.abs(snr_list - snr_value))
        actual_snr = snr_list[idx]

        print(f"\nSNR = {actual_snr} dB")
        print(f"{'Algorithm':<15} {'Angle RMSE (°)':<18} {'Range RMSE (m)':<18} {'Time (ms)':<15}")
        print("-" * 80)

        for result in all_results:
            algo = result['algorithm']
            angle_rmse = result['rmse_theta'][idx]
            range_rmse = result['rmse_r'][idx]
            avg_time = result['avg_time'][idx] * 1000
            print(f"{algo:<15} {angle_rmse:<18.6f} {range_rmse:<18.4f} {avg_time:<15.4f}")

        # CRLB
        crlb_angle = crlb_results['crlb_theta'][idx]
        crlb_range = crlb_results['crlb_r'][idx]
        print("-" * 80)
        print(f"{'CRLB':<15} {crlb_angle:<18.6f} {crlb_range:<18.4f} {'-':<15}")

    else:
        # 打印所有SNR的结果
        print(f"\n{'SNR (dB)':<10} {'Algorithm':<15} {'Angle RMSE (°)':<18} {'Range RMSE (m)':<18}")
        print("-" * 80)

        for snr_idx, snr in enumerate(snr_list):
            print(f"\n{snr:<10.0f}")
            for result in all_results:
                algo = result['algorithm']
                angle_rmse = result['rmse_theta'][snr_idx]
                range_rmse = result['rmse_r'][snr_idx]
                print(f"{'':10} {algo:<15} {angle_rmse:<18.6f} {range_rmse:<18.4f}")

            # CRLB
            crlb_angle = crlb_results['crlb_theta'][snr_idx]
            crlb_range = crlb_results['crlb_r'][snr_idx]
            print(f"{'':10} {'CRLB':<15} {crlb_angle:<18.6f} {crlb_range:<18.4f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='从保存的JSON结果绘制评测对比图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 绘制全部SNR范围
  python plot_benchmark_results.py results/comprehensive_benchmark/benchmark_results.json

  # 只绘制 0-20dB 范围
  python plot_benchmark_results.py results/comprehensive_benchmark/benchmark_results.json --snr-min 0 --snr-max 20

  # 查看特定SNR点的性能
  python plot_benchmark_results.py results/comprehensive_benchmark/benchmark_results.json --summary 20

  # 不显示图形（服务器模式）
  python plot_benchmark_results.py results/comprehensive_benchmark/benchmark_results.json --no-show
        """
    )

    parser.add_argument('json_path', type=str, help='JSON结果文件路径')
    parser.add_argument('--snr-min', type=float, default=None, help='最小SNR (dB)')
    parser.add_argument('--snr-max', type=float, default=None, help='最大SNR (dB)')
    parser.add_argument('--output', type=str, default=None, help='输出目录（默认与JSON同目录）')
    parser.add_argument('--summary', type=float, default=None, help='打印指定SNR的性能总结')
    parser.add_argument('--summary-all', action='store_true', help='打印所有SNR的性能总结')
    parser.add_argument('--no-show', action='store_true', help='不显示图形（适用于服务器）')
    parser.add_argument('--dpi', type=int, default=300, help='图片分辨率')
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 6], help='图形大小 (宽 高)')

    args = parser.parse_args()

    # 检查JSON文件是否存在
    if not os.path.exists(args.json_path):
        print(f"❌ 错误: 文件不存在 {args.json_path}")
        return 1

    print("\n" + "=" * 80)
    print("FDA-MIMO 评测结果绘图工具")
    print("=" * 80)
    print(f"读取数据: {args.json_path}")

    # 加载结果
    results_dict = load_results(args.json_path)
    crlb_results = results_dict['crlb']
    all_results = results_dict['algorithms']

    # 确定输出目录
    if args.output is None:
        output_dir = os.path.dirname(args.json_path)
    else:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)

    # 确定SNR范围
    snr_range = None
    if args.snr_min is not None or args.snr_max is not None:
        full_snr = np.array(crlb_results['snr_db_list'])
        snr_min = args.snr_min if args.snr_min is not None else full_snr.min()
        snr_max = args.snr_max if args.snr_max is not None else full_snr.max()
        snr_range = (snr_min, snr_max)
        print(f"SNR范围: [{snr_min}, {snr_max}] dB")
    else:
        full_snr = np.array(crlb_results['snr_db_list'])
        print(f"SNR范围: [{full_snr.min()}, {full_snr.max()}] dB (全部)")

    # 绘图
    print("\n绘制对比图...")
    plot_comparison(
        all_results, crlb_results, output_dir,
        snr_range=snr_range,
        show_plot=not args.no_show,
        figsize=tuple(args.figsize),
        dpi=args.dpi
    )

    # 打印总结表格
    if args.summary is not None:
        print_summary_table(all_results, crlb_results, snr_value=args.summary)
    elif args.summary_all:
        print_summary_table(all_results, crlb_results, snr_value=None)

    print("\n" + "=" * 80)
    print("✓ 完成！")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
