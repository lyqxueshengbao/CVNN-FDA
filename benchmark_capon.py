"""
Capon算法独立评测脚本
"""
import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import config as cfg
from benchmark import capon_2d
from utils_physics import generate_covariance_matrix
import time

def benchmark_capon(snr_list, L_snapshots, num_samples, grid_size='standard', output_dir='results'):
    """
    Capon评测

    Args:
        snr_list: SNR列表
        L_snapshots: 快拍数
        num_samples: 每个SNR的样本数
        grid_size: 网格大小 ('coarse', 'standard', 'dense')
        output_dir: 输出目录
    """
    cfg.L_snapshots = L_snapshots

    # 网格配置
    grid_configs = {
        'coarse': (50, 30),
        'standard': (80, 50),
        'dense': (150, 100)
    }
    num_r, num_theta = grid_configs.get(grid_size, (80, 50))

    print(f"📊 快拍数: L={L_snapshots}")
    print(f"📊 样本数: {num_samples}")
    print(f"📊 SNR范围: {snr_list}")
    print(f"📊 网格大小: {num_r}×{num_theta} ({grid_size})")

    # 生成网格
    r_grid = np.linspace(0, cfg.r_max, num_r)
    theta_grid = np.linspace(cfg.theta_min, cfg.theta_max, num_theta)

    r_step = r_grid[1] - r_grid[0]
    theta_step = theta_grid[1] - theta_grid[0]
    print(f"📐 网格步长: Δr={r_step:.2f}m, Δθ={theta_step:.3f}°")

    results = {
        'config': {
            'method': 'Capon',
            'L_snapshots': L_snapshots,
            'num_samples': num_samples,
            'snr_list': snr_list,
            'grid_size': grid_size,
            'num_r_points': num_r,
            'num_theta_points': num_theta
        },
        'summary': {
            'rmse_r': [],
            'rmse_theta': [],
            'time_ms': []
        },
        'detailed_samples': {}
    }

    print(f"\n{'='*70}")
    print("开始Capon评测")
    print(f"{'='*70}")

    for snr in snr_list:
        print(f"\n📡 SNR = {snr:+3d} dB")

        errors_r = []
        errors_theta = []
        times = []
        samples = []

        for _ in tqdm(range(num_samples), desc=f"SNR={snr}dB"):
            # 生成目标
            r_true = np.random.uniform(0, cfg.r_max)
            theta_true = np.random.uniform(cfg.theta_min, cfg.theta_max)

            # 生成协方差矩阵
            R = generate_covariance_matrix(r_true, theta_true, snr)
            R_complex = R[0] + 1j * R[1]

            # 运行Capon
            t0 = time.time()
            r_est, theta_est = capon_2d(R_complex, r_grid, theta_grid)
            elapsed = time.time() - t0

            # 记录误差
            errors_r.append((r_est - r_true)**2)
            errors_theta.append((theta_est - theta_true)**2)
            times.append(elapsed)

            # 保存详细结果
            samples.append({
                'r_true': float(r_true),
                'theta_true': float(theta_true),
                'r_est': float(r_est),
                'theta_est': float(theta_est)
            })

        # 统计
        rmse_r = np.sqrt(np.mean(errors_r))
        rmse_theta = np.sqrt(np.mean(errors_theta))
        avg_time = np.mean(times) * 1000

        results['summary']['rmse_r'].append(float(rmse_r))
        results['summary']['rmse_theta'].append(float(rmse_theta))
        results['summary']['time_ms'].append(float(avg_time))
        results['detailed_samples'][f'SNR_{snr}dB'] = samples

        print(f"  RMSE_r: {rmse_r:.2f}m, RMSE_θ: {rmse_theta:.3f}°, Time: {avg_time:.1f}ms")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'capon_L{L_snapshots}_{grid_size}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 结果已保存: {output_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capon独立评测')
    parser.add_argument('--snr-list', type=float, nargs='+',
                        default=[-5, 0, 5, 10, 15, 20],
                        help='SNR列表')
    parser.add_argument('--L', type=int, default=10,
                        help='快拍数')
    parser.add_argument('--samples', type=int, default=500,
                        help='每个SNR的样本数')
    parser.add_argument('--grid', type=str, default='standard',
                        choices=['coarse', 'standard', 'dense'],
                        help='网格大小')
    parser.add_argument('--output', type=str, default='results',
                        help='输出目录')

    args = parser.parse_args()

    benchmark_capon(
        snr_list=args.snr_list,
        L_snapshots=args.L,
        num_samples=args.samples,
        grid_size=args.grid,
        output_dir=args.output
    )
