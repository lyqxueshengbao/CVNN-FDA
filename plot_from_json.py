"""
从JSON文件灵活绘制评测结果

功能：
- 自定义SNR区间
- 自定义角度/距离范围
- 选择要对比的算法
- 多种可视化方式
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import List, Dict, Optional


def load_json_data(json_path: str) -> Dict:
    """加载JSON数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_samples(samples: List[Dict],
                   r_range: Optional[tuple] = None,
                   theta_range: Optional[tuple] = None) -> List[Dict]:
    """
    过滤样本数据

    Args:
        samples: 样本列表
        r_range: 距离范围 (r_min, r_max)
        theta_range: 角度范围 (theta_min, theta_max)
    """
    filtered = samples

    if r_range is not None:
        r_min, r_max = r_range
        filtered = [s for s in filtered if r_min <= s['r_true'] <= r_max]

    if theta_range is not None:
        theta_min, theta_max = theta_range
        filtered = [s for s in filtered if theta_min <= s['theta_true'] <= theta_max]

    return filtered


def compute_rmse(samples: List[Dict], method: str) -> tuple:
    """计算指定方法的RMSE"""
    if not samples:
        return np.nan, np.nan

    r_errors = [(s[method][0] - s['r_true'])**2 for s in samples if method in s]
    theta_errors = [(s[method][1] - s['theta_true'])**2 for s in samples if method in s]

    if not r_errors:
        return np.nan, np.nan

    rmse_r = np.sqrt(np.mean(r_errors))
    rmse_theta = np.sqrt(np.mean(theta_errors))

    return rmse_r, rmse_theta


def plot_snr_curves(data: Dict,
                    snr_range: Optional[tuple] = None,
                    methods: Optional[List[str]] = None,
                    r_range: Optional[tuple] = None,
                    theta_range: Optional[tuple] = None,
                    output_path: str = 'results/custom_plot.png'):
    """
    绘制SNR-RMSE曲线

    Args:
        data: JSON数据
        snr_range: SNR范围 (snr_min, snr_max)
        methods: 要绘制的方法列表
        r_range: 距离过滤范围
        theta_range: 角度过滤范围
        output_path: 输出路径
    """
    # 获取所有SNR
    all_snrs = []
    for key in data['detailed_samples'].keys():
        if key.startswith('SNR_'):
            snr = float(key.replace('SNR_', '').replace('dB', ''))
            all_snrs.append(snr)
    all_snrs = sorted(all_snrs)

    # 过滤SNR范围
    if snr_range is not None:
        snr_min, snr_max = snr_range
        all_snrs = [s for s in all_snrs if snr_min <= s <= snr_max]

    if not all_snrs:
        print("❌ 没有符合条件的SNR数据")
        return

    # 确定要绘制的方法
    if methods is None:
        # 从第一个样本中获取所有方法
        first_snr_key = f"SNR_{int(all_snrs[0])}dB"
        first_sample = data['detailed_samples'][first_snr_key][0]
        methods = [k for k in first_sample.keys() if k not in ['r_true', 'theta_true']]

    # 收集数据
    results = {m: {'rmse_r': [], 'rmse_theta': []} for m in methods}

    for snr in all_snrs:
        snr_key = f"SNR_{int(snr)}dB"
        if snr_key not in data['detailed_samples']:
            continue

        samples = data['detailed_samples'][snr_key]
        samples = filter_samples(samples, r_range, theta_range)

        for method in methods:
            rmse_r, rmse_theta = compute_rmse(samples, method)
            results[method]['rmse_r'].append(rmse_r)
            results[method]['rmse_theta'].append(rmse_theta)

    # 绘图 - 自动适配所有方法
    colors = {
        'CVNN': '#1f77b4',
        'Real-CNN': '#2ca02c',
        'MUSIC': '#d62728',
        'Capon': '#9467bd',
        'Beamforming': '#8c564b',
        'ESPRIT': '#ff7f0e',
        'CRB': '#000000'
    }
    markers = {
        'CVNN': 'o',
        'Real-CNN': '^',
        'MUSIC': 's',
        'Capon': 'D',
        'Beamforming': 'v',
        'ESPRIT': 'd',
        'CRB': 'x'
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 距离RMSE
    for method in methods:
        if method == 'CRB':
            continue
        ax1.plot(all_snrs, results[method]['rmse_r'],
                color=colors.get(method, 'gray'),
                marker=markers.get(method, 'o'),
                linewidth=2, markersize=8, label=method)

    # 如果有CRB数据，单独绘制
    if 'CRB' in results and any(not np.isnan(x) for x in results['CRB']['rmse_r']):
        ax1.plot(all_snrs, results['CRB']['rmse_r'],
                'k--', linewidth=2, label='CRB', alpha=0.6)

    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('RMSE Range (m)', fontsize=12)
    ax1.set_title('Range Estimation Accuracy', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 角度RMSE
    for method in methods:
        if method == 'CRB':
            continue
        ax2.plot(all_snrs, results[method]['rmse_theta'],
                color=colors.get(method, 'gray'),
                marker=markers.get(method, 'o'),
                linewidth=2, markersize=8, label=method)

    if 'CRB' in results and any(not np.isnan(x) for x in results['CRB']['rmse_theta']):
        ax2.plot(all_snrs, results['CRB']['rmse_theta'],
                'k--', linewidth=2, label='CRB', alpha=0.6)

    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('RMSE Angle (°)', fontsize=12)
    ax2.set_title('Angle Estimation Accuracy', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # 添加标题说明过滤条件
    title_parts = []
    if snr_range:
        title_parts.append(f"SNR: [{snr_range[0]}, {snr_range[1]}] dB")
    if r_range:
        title_parts.append(f"Range: [{r_range[0]}, {r_range[1]}] m")
    if theta_range:
        title_parts.append(f"Angle: [{theta_range[0]}, {theta_range[1]}]°")

    if title_parts:
        fig.suptitle(' | '.join(title_parts), fontsize=11, y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_path}")
    plt.close()


def plot_scatter(data: Dict,
                 snr: float,
                 method: str = 'CVNN',
                 r_range: Optional[tuple] = None,
                 theta_range: Optional[tuple] = None,
                 output_path: str = 'results/scatter_plot.png'):
    """
    绘制预测值vs真实值散点图

    Args:
        data: JSON数据
        snr: 指定SNR
        method: 要绘制的方法
        r_range: 距离过滤范围
        theta_range: 角度过滤范围
        output_path: 输出路径
    """
    snr_key = f"SNR_{int(snr)}dB"
    if snr_key not in data['detailed_samples']:
        print(f"❌ 没有SNR={snr}dB的数据")
        return

    samples = data['detailed_samples'][snr_key]
    samples = filter_samples(samples, r_range, theta_range)

    if not samples:
        print("❌ 没有符合条件的样本")
        return

    # 提取数据
    r_true = [s['r_true'] for s in samples]
    theta_true = [s['theta_true'] for s in samples]
    r_pred = [s[method][0] for s in samples if method in s]
    theta_pred = [s[method][1] for s in samples if method in s]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 距离散点图
    ax1.scatter(r_true, r_pred, alpha=0.5, s=20)
    r_min, r_max = min(r_true), max(r_true)
    ax1.plot([r_min, r_max], [r_min, r_max], 'r--', linewidth=2, label='Perfect')
    ax1.set_xlabel('True Range (m)', fontsize=12)
    ax1.set_ylabel('Predicted Range (m)', fontsize=12)
    ax1.set_title(f'{method} Range Estimation (SNR={snr}dB)', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 角度散点图
    ax2.scatter(theta_true, theta_pred, alpha=0.5, s=20)
    theta_min, theta_max = min(theta_true), max(theta_true)
    ax2.plot([theta_min, theta_max], [theta_min, theta_max], 'r--', linewidth=2, label='Perfect')
    ax2.set_xlabel('True Angle (°)', fontsize=12)
    ax2.set_ylabel('Predicted Angle (°)', fontsize=12)
    ax2.set_title(f'{method} Angle Estimation (SNR={snr}dB)', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 散点图已保存: {output_path}")
    plt.close()


def plot_error_distribution(data: Dict,
                            snr: float,
                            methods: Optional[List[str]] = None,
                            r_range: Optional[tuple] = None,
                            theta_range: Optional[tuple] = None,
                            output_path: str = 'results/error_dist.png'):
    """
    绘制误差分布直方图

    Args:
        data: JSON数据
        snr: 指定SNR
        methods: 要绘制的方法列表
        r_range: 距离过滤范围
        theta_range: 角度过滤范围
        output_path: 输出路径
    """
    snr_key = f"SNR_{int(snr)}dB"
    if snr_key not in data['detailed_samples']:
        print(f"❌ 没有SNR={snr}dB的数据")
        return

    samples = data['detailed_samples'][snr_key]
    samples = filter_samples(samples, r_range, theta_range)

    if not samples:
        print("❌ 没有符合条件的样本")
        return

    if methods is None:
        methods = [k for k in samples[0].keys() if k not in ['r_true', 'theta_true']]

    fig, axes = plt.subplots(2, len(methods), figsize=(5*len(methods), 8))
    if len(methods) == 1:
        axes = axes.reshape(-1, 1)

    colors = {
        'CVNN': '#1f77b4',
        'Real-CNN': '#2ca02c',
        'MUSIC': '#d62728',
        'Capon': '#9467bd',
        'Beamforming': '#8c564b',
        'ESPRIT': '#ff7f0e'
    }

    for i, method in enumerate(methods):
        # 计算误差
        r_errors = [s[method][0] - s['r_true'] for s in samples if method in s]
        theta_errors = [s[method][1] - s['theta_true'] for s in samples if method in s]

        # 距离误差分布
        ax1 = axes[0, i]
        ax1.hist(r_errors, bins=30, alpha=0.7, color=colors.get(method, 'gray'), edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Range Error (m)', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title(f'{method}\nRMSE={np.sqrt(np.mean(np.array(r_errors)**2)):.2f}m', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 角度误差分布
        ax2 = axes[1, i]
        ax2.hist(theta_errors, bins=30, alpha=0.7, color=colors.get(method, 'gray'), edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Angle Error (°)', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title(f'{method}\nRMSE={np.sqrt(np.mean(np.array(theta_errors)**2)):.2f}°', fontsize=12)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Error Distribution (SNR={snr}dB)', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 误差分布图已保存: {output_path}")
    plt.close()


def plot_comparison_table(data: Dict,
                          snr_range: Optional[tuple] = None,
                          methods: Optional[List[str]] = None,
                          r_range: Optional[tuple] = None,
                          theta_range: Optional[tuple] = None):
    """打印性能对比表格"""
    # 获取所有SNR
    all_snrs = []
    for key in data['detailed_samples'].keys():
        if key.startswith('SNR_'):
            snr = float(key.replace('SNR_', '').replace('dB', ''))
            all_snrs.append(snr)
    all_snrs = sorted(all_snrs)

    # 过滤SNR范围
    if snr_range is not None:
        snr_min, snr_max = snr_range
        all_snrs = [s for s in all_snrs if snr_min <= s <= snr_max]

    if methods is None:
        first_snr_key = f"SNR_{int(all_snrs[0])}dB"
        first_sample = data['detailed_samples'][first_snr_key][0]
        methods = [k for k in first_sample.keys() if k not in ['r_true', 'theta_true']]

    print("\n" + "="*80)
    print("性能对比表格")
    print("="*80)
    print(f"{'SNR(dB)':<10} | ", end='')
    for method in methods:
        print(f"{method:<20} | ", end='')
    print()
    print("-"*80)

    for snr in all_snrs:
        snr_key = f"SNR_{int(snr)}dB"
        if snr_key not in data['detailed_samples']:
            continue

        samples = data['detailed_samples'][snr_key]
        samples = filter_samples(samples, r_range, theta_range)

        print(f"{snr:<10.0f} | ", end='')
        for method in methods:
            rmse_r, rmse_theta = compute_rmse(samples, method)
            print(f"R:{rmse_r:>6.2f} θ:{rmse_theta:>5.2f} | ", end='')
        print()

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='从JSON文件灵活绘制评测结果')

    # 输入输出
    parser.add_argument('--json', type=str, required=True,
                        help='JSON数据文件路径')
    parser.add_argument('--output', type=str, default='results/custom_plot.png',
                        help='输出图片路径')

    # 绘图类型
    parser.add_argument('--plot-type', type=str, default='curves',
                        choices=['curves', 'scatter', 'error_dist', 'table'],
                        help='绘图类型: curves(SNR曲线), scatter(散点图), error_dist(误差分布), table(表格)')

    # SNR范围
    parser.add_argument('--snr-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='SNR范围，例如: --snr-range -5 10')
    parser.add_argument('--snr', type=float,
                        help='指定单个SNR (用于scatter和error_dist)')

    # 距离和角度范围
    parser.add_argument('--r-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='距离范围 (m)，例如: --r-range 0 500')
    parser.add_argument('--theta-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='角度范围 (°)，例如: --theta-range -30 30')

    # 方法选择
    parser.add_argument('--methods', type=str, nargs='+',
                        help='要绘制的方法，例如: --methods CVNN MUSIC')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.json):
        print(f"❌ 文件不存在: {args.json}")
        return

    # 加载数据
    print(f"📂 加载数据: {args.json}")
    data = load_json_data(args.json)

    # 转换范围参数
    snr_range = tuple(args.snr_range) if args.snr_range else None
    r_range = tuple(args.r_range) if args.r_range else None
    theta_range = tuple(args.theta_range) if args.theta_range else None

    # 根据类型绘图
    if args.plot_type == 'curves':
        plot_snr_curves(data, snr_range, args.methods, r_range, theta_range, args.output)

    elif args.plot_type == 'scatter':
        if args.snr is None:
            print("❌ scatter模式需要指定--snr参数")
            return
        method = args.methods[0] if args.methods else 'CVNN'
        plot_scatter(data, args.snr, method, r_range, theta_range, args.output)

    elif args.plot_type == 'error_dist':
        if args.snr is None:
            print("❌ error_dist模式需要指定--snr参数")
            return
        plot_error_distribution(data, args.snr, args.methods, r_range, theta_range, args.output)

    elif args.plot_type == 'table':
        plot_comparison_table(data, snr_range, args.methods, r_range, theta_range)


if __name__ == "__main__":
    main()
