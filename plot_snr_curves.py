"""
绘制SNR-RMSE折线图
从合并的JSON文件或单独的JSON文件生成论文质量的图表
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 颜色和标记配置
COLORS = {
    'CVNN': '#1f77b4',
    'Real-CNN': '#2ca02c',
    'MUSIC': '#d62728',
    'MUSIC_standard': '#d62728',
    'MUSIC_coarse': '#ff9896',
    'MUSIC_dense': '#8b0000',
    'Capon': '#9467bd',
    'Capon_standard': '#9467bd',
    'Beamforming': '#8c564b',
    'ESPRIT': '#ff7f0e',
    'CRB': '#000000'
}

MARKERS = {
    'CVNN': 'o',
    'Real-CNN': '^',
    'MUSIC': 's',
    'MUSIC_standard': 's',
    'MUSIC_coarse': 's',
    'MUSIC_dense': 's',
    'Capon': 'D',
    'Capon_standard': 'D',
    'Beamforming': 'v',
    'ESPRIT': 'd',
    'CRB': 'x'
}

LABELS = {
    'CVNN': 'CVNN',
    'Real-CNN': 'Real-CNN',
    'MUSIC': 'MUSIC',
    'MUSIC_standard': 'MUSIC',
    'MUSIC_coarse': 'MUSIC (粗网格)',
    'MUSIC_dense': 'MUSIC (密集网格)',
    'Capon': 'Capon',
    'Capon_standard': 'Capon',
    'Beamforming': 'Beamforming',
    'ESPRIT': 'ESPRIT',
    'CRB': 'CRB'
}

def load_data(json_path):
    """加载JSON数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_snr_curves(json_path, methods=None, snr_range=None,
                    show_crb=True, log_scale=True, output_path=None,
                    figsize=(14, 5), dpi=300):
    """
    绘制SNR-RMSE折线图

    Args:
        json_path: JSON文件路径
        methods: 要绘制的方法列表（None表示全部）
        snr_range: SNR范围 (min, max)
        show_crb: 是否显示CRB
        log_scale: 是否使用对数坐标
        output_path: 输出路径（None表示自动生成）
        figsize: 图片大小
        dpi: 分辨率
    """
    # 加载数据
    data = load_data(json_path)

    # 获取SNR列表
    snr_list = data['config']['snr_list']

    # 过滤SNR范围
    if snr_range is not None:
        snr_min, snr_max = snr_range
        indices = [i for i, snr in enumerate(snr_list) if snr_min <= snr <= snr_max]
        snr_list = [snr_list[i] for i in indices]
    else:
        indices = list(range(len(snr_list)))

    # 获取所有方法
    all_methods = list(data['summary'].keys())
    if 'CRB' in all_methods and not show_crb:
        all_methods.remove('CRB')

    # 过滤方法
    if methods is not None:
        all_methods = [m for m in all_methods if m in methods or m.startswith(tuple(methods))]

    print(f"绘制方法: {all_methods}")
    print(f"SNR范围: {snr_list}")

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 绘制距离RMSE
    for method in all_methods:
        if method == 'CRB':
            continue

        rmse_r = data['summary'][method]['rmse_r']
        rmse_r_filtered = [rmse_r[i] for i in indices]

        label = LABELS.get(method, method)
        color = COLORS.get(method, 'gray')
        marker = MARKERS.get(method, 'o')

        ax1.plot(snr_list, rmse_r_filtered,
                color=color, marker=marker, linewidth=2, markersize=8,
                label=label, alpha=0.8)

    # 绘制CRB
    if show_crb and 'CRB' in data['summary']:
        crb_r = data['summary']['CRB']['rmse_r']
        crb_r_filtered = [crb_r[i] for i in indices]
        ax1.plot(snr_list, crb_r_filtered,
                'k--', linewidth=2, label='CRB', alpha=0.6)

    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('RMSE Range (m)', fontsize=12)
    ax1.set_title('Distance Estimation Performance', fontsize=13)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3, which='both', linestyle='--')
    if log_scale:
        ax1.set_yscale('log')

    # 绘制角度RMSE
    for method in all_methods:
        if method == 'CRB':
            continue

        rmse_theta = data['summary'][method]['rmse_theta']
        rmse_theta_filtered = [rmse_theta[i] for i in indices]

        label = LABELS.get(method, method)
        color = COLORS.get(method, 'gray')
        marker = MARKERS.get(method, 'o')

        ax2.plot(snr_list, rmse_theta_filtered,
                color=color, marker=marker, linewidth=2, markersize=8,
                label=label, alpha=0.8)

    # 绘制CRB
    if show_crb and 'CRB' in data['summary']:
        crb_theta = data['summary']['CRB']['rmse_theta']
        crb_theta_filtered = [crb_theta[i] for i in indices]
        ax2.plot(snr_list, crb_theta_filtered,
                'k--', linewidth=2, label='CRB', alpha=0.6)

    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('RMSE Angle (°)', fontsize=12)
    ax2.set_title('Angle Estimation Performance', fontsize=13)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, which='both', linestyle='--')
    if log_scale:
        ax2.set_yscale('log')

    # 添加配置信息
    L = data['config'].get('L_snapshots', 'N/A')
    fig.suptitle(f'Performance Comparison (L={L} snapshots)', fontsize=14, y=0.98)

    plt.tight_layout()

    # 保存图片
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = f'results/{base_name}_curves.png'

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\n✅ 图表已保存: {output_path}")

    plt.show()

def plot_comparison_table(json_path, methods=None, output_path=None):
    """
    绘制性能对比表格图

    Args:
        json_path: JSON文件路径
        methods: 要显示的方法列表
        output_path: 输出路径
    """
    data = load_data(json_path)
    snr_list = data['config']['snr_list']

    # 获取方法
    all_methods = list(data['summary'].keys())
    if methods is not None:
        all_methods = [m for m in all_methods if m in methods]

    # 准备表格数据
    table_data = []
    headers = ['Method'] + [f'{snr}dB' for snr in snr_list] + ['Avg']

    for method in all_methods:
        if method == 'CRB':
            continue
        rmse_r = data['summary'][method]['rmse_r']
        row = [LABELS.get(method, method)]
        row.extend([f'{r:.1f}' for r in rmse_r])
        row.append(f'{np.mean(rmse_r):.1f}')
        table_data.append(row)

    # 添加CRB
    if 'CRB' in data['summary']:
        crb_r = data['summary']['CRB']['rmse_r']
        row = ['CRB']
        row.extend([f'{r:.1f}' for r in crb_r])
        row.append(f'{np.mean(crb_r):.1f}')
        table_data.append(row)

    # 创建表格图
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15] + [0.08] * len(snr_list) + [0.08])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置CRB行样式
    if 'CRB' in data['summary']:
        for i in range(len(headers)):
            table[(len(table_data), i)].set_facecolor('#FFF9C4')

    L = data['config'].get('L_snapshots', 'N/A')
    plt.title(f'RMSE Range (m) - L={L} snapshots', fontsize=14, pad=20)

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = f'results/{base_name}_table.png'

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 表格已保存: {output_path}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制SNR-RMSE折线图')
    parser.add_argument('--json', type=str, required=True,
                        help='JSON数据文件路径')
    parser.add_argument('--methods', type=str, nargs='+',
                        help='要绘制的方法（不指定则绘制全部）')
    parser.add_argument('--snr-range', type=float, nargs=2,
                        metavar=('MIN', 'MAX'),
                        help='SNR范围')
    parser.add_argument('--no-crb', action='store_true',
                        help='不显示CRB')
    parser.add_argument('--linear', action='store_true',
                        help='使用线性坐标（默认对数）')
    parser.add_argument('--output', type=str,
                        help='输出路径')
    parser.add_argument('--table', action='store_true',
                        help='绘制表格而不是曲线')
    parser.add_argument('--dpi', type=int, default=300,
                        help='图片分辨率')

    args = parser.parse_args()

    if args.table:
        plot_comparison_table(
            json_path=args.json,
            methods=args.methods,
            output_path=args.output
        )
    else:
        plot_snr_curves(
            json_path=args.json,
            methods=args.methods,
            snr_range=tuple(args.snr_range) if args.snr_range else None,
            show_crb=not args.no_crb,
            log_scale=not args.linear,
            output_path=args.output,
            dpi=args.dpi
        )
