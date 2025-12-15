"""
CVNN算法独立评测脚本
"""
import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
import config as cfg
from model import FDA_CVNN, FDA_CVNN_Attention
from utils_physics import generate_covariance_matrix

def load_cvnn_model(device, model_path=None, L_snapshots=None):
    """加载CVNN模型"""
    if model_path is None:
        checkpoint_dir = cfg.checkpoint_dir
        candidates = [
            f"{checkpoint_dir}/fda_cvnn_L{L_snapshots}_best.pth",
            f"{checkpoint_dir}/fda_cvnn_best.pth"
        ]
        for path in candidates:
            if os.path.exists(path):
                model_path = path
                break

    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        return FDA_CVNN().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # 检测模型类型
    keys = list(state_dict.keys())
    has_attention = any('attn' in k for k in keys)

    if has_attention:
        model = FDA_CVNN_Attention().to(device)
    else:
        model = FDA_CVNN().to(device)

    # 移除module.前缀
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    print(f"✅ 加载模型: {model_path}")
    return model

def benchmark_cvnn(snr_list, L_snapshots, num_samples, output_dir='results'):
    """
    CVNN评测

    Args:
        snr_list: SNR列表
        L_snapshots: 快拍数
        num_samples: 每个SNR的样本数
        output_dir: 输出目录
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 设备: {device}")
    print(f"📊 快拍数: L={L_snapshots}")
    print(f"📊 样本数: {num_samples}")
    print(f"📊 SNR范围: {snr_list}")

    # 加载模型
    cfg.L_snapshots = L_snapshots
    model = load_cvnn_model(device, L_snapshots=L_snapshots)
    model.eval()

    # Warm-up
    dummy = torch.randn(1, 2, cfg.M * cfg.N, cfg.M * cfg.N).to(device)
    for _ in range(3):
        with torch.no_grad():
            model(dummy)

    results = {
        'config': {
            'method': 'CVNN',
            'L_snapshots': L_snapshots,
            'num_samples': num_samples,
            'snr_list': snr_list
        },
        'summary': {
            'rmse_r': [],
            'rmse_theta': [],
            'time_ms': []
        },
        'detailed_samples': {}
    }

    print(f"\n{'='*70}")
    print("开始CVNN评测")
    print(f"{'='*70}")

    for snr in snr_list:
        print(f"\n📡 SNR = {int(snr):+3d} dB")

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
            R_tensor = torch.FloatTensor(R).unsqueeze(0).to(device)

            # 推理
            import time
            t0 = time.time()
            with torch.no_grad():
                pred = model(R_tensor).cpu().numpy()[0]
            elapsed = time.time() - t0

            # 反归一化
            r_est = pred[0] * cfg.r_max
            theta_est = pred[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min

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
    output_path = os.path.join(output_dir, f'cvnn_L{L_snapshots}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 结果已保存: {output_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CVNN独立评测')
    parser.add_argument('--snr-list', type=float, nargs='+',
                        default=[-5, 0, 5, 10, 15, 20],
                        help='SNR列表')
    parser.add_argument('--L', type=int, default=10,
                        help='快拍数')
    parser.add_argument('--samples', type=int, default=500,
                        help='每个SNR的样本数')
    parser.add_argument('--output', type=str, default='results',
                        help='输出目录')

    args = parser.parse_args()

    benchmark_cvnn(
        snr_list=args.snr_list,
        L_snapshots=args.L,
        num_samples=args.samples,
        output_dir=args.output
    )
