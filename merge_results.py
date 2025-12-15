"""
合并各算法的独立评测结果
"""
import os
import json
import argparse
import glob

def merge_results(result_dir='results', L_snapshots=10, output_name=None):
    """
    合并各算法的评测结果

    Args:
        result_dir: 结果目录
        L_snapshots: 快拍数
        output_name: 输出文件名（可选）
    """
    print(f"{'='*70}")
    print(f"合并评测结果 (L={L_snapshots})")
    print(f"{'='*70}")

    # 查找所有相关的结果文件
    patterns = [
        f"{result_dir}/cvnn_L{L_snapshots}.json",
        f"{result_dir}/music_L{L_snapshots}_*.json",
        f"{result_dir}/capon_L{L_snapshots}_*.json",
        f"{result_dir}/beamforming_L{L_snapshots}_*.json",
    ]

    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))

    if not all_files:
        print(f"❌ 未找到任何结果文件")
        print(f"   搜索模式: {patterns}")
        return None

    print(f"\n找到 {len(all_files)} 个结果文件:")
    for f in all_files:
        print(f"  - {os.path.basename(f)}")

    # 加载所有结果
    all_results = {}
    snr_list = None

    for file_path in all_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        method = data['config']['method']
        grid_size = data['config'].get('grid_size', '')

        # 构造方法名
        if grid_size:
            method_name = f"{method}_{grid_size}"
        else:
            method_name = method

        all_results[method_name] = data

        # 获取SNR列表
        if snr_list is None:
            snr_list = data['config']['snr_list']

    # 构造合并结果
    merged = {
        'config': {
            'L_snapshots': L_snapshots,
            'num_samples': all_results[list(all_results.keys())[0]]['config']['num_samples'],
            'snr_list': snr_list,
            'methods': list(all_results.keys())
        },
        'summary': {},
        'detailed_samples': {}
    }

    # 合并summary
    for method_name, data in all_results.items():
        merged['summary'][method_name] = data['summary']

    # 合并detailed_samples
    for snr in snr_list:
        snr_key = f'SNR_{int(snr)}dB'
        merged['detailed_samples'][snr_key] = {}

        for method_name, data in all_results.items():
            if snr_key in data['detailed_samples']:
                samples = data['detailed_samples'][snr_key]
                # 转换格式：从列表转为字典
                for i, sample in enumerate(samples):
                    if i not in merged['detailed_samples'][snr_key]:
                        merged['detailed_samples'][snr_key][i] = {
                            'r_true': sample['r_true'],
                            'theta_true': sample['theta_true']
                        }
                    merged['detailed_samples'][snr_key][i][method_name] = [
                        sample['r_est'],
                        sample['theta_est']
                    ]

    # 转换回列表格式
    for snr_key in merged['detailed_samples']:
        sample_dict = merged['detailed_samples'][snr_key]
        sample_list = []
        for i in sorted(sample_dict.keys()):
            sample_list.append(sample_dict[i])
        merged['detailed_samples'][snr_key] = sample_list

    # 保存合并结果
    if output_name is None:
        output_name = f'merged_L{L_snapshots}.json'

    output_path = os.path.join(result_dir, output_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)

    print(f"\n✅ 合并结果已保存: {output_path}")

    # 打印性能对比表
    print(f"\n{'='*70}")
    print("性能对比表")
    print(f"{'='*70}")
    print(f"{'方法':<20} {'RMSE_r (m)':<40} {'Time (ms)':<20}")
    print("-"*70)

    for method_name in merged['summary']:
        rmse_r_list = merged['summary'][method_name]['rmse_r']
        time_list = merged['summary'][method_name]['time_ms']

        rmse_str = ', '.join([f"{r:.1f}" for r in rmse_r_list])
        time_str = f"{np.mean(time_list):.1f}"

        print(f"{method_name:<20} {rmse_str:<40} {time_str:<20}")

    print(f"{'='*70}")

    return merged

if __name__ == "__main__":
    import numpy as np

    parser = argparse.ArgumentParser(description='合并评测结果')
    parser.add_argument('--L', type=int, default=10,
                        help='快拍数')
    parser.add_argument('--dir', type=str, default='results',
                        help='结果目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件名')

    args = parser.parse_args()

    merge_results(
        result_dir=args.dir,
        L_snapshots=args.L,
        output_name=args.output
    )
