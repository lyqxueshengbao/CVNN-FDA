import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import config as cfg
from model import FDA_CVNN, FDA_CVNN_Attention
from utils_physics import generate_covariance_matrix


def get_model_by_type(model_type: str, device):
    """
    根据模型类型实例化正确结构的模型
    
    Args:
        model_type: 模型类型
            - 'full' / 'dual': 完整模型 (带 Dual 注意力 = SE + FAR)
            - 'se': 只用 SE 注意力
            - 'far': 只用 FAR 注意力
            - 'no-attn' / 'standard': 无注意力的基线模型
    """
    model_type = model_type.lower()
    
    if model_type in ['full', 'dual']:
        model = FDA_CVNN_Attention(attention_type='dual')
    elif model_type == 'se':
        model = FDA_CVNN_Attention(attention_type='se')
    elif model_type == 'far':
        model = FDA_CVNN_Attention(attention_type='far')
    elif model_type == 'cbam':
        model = FDA_CVNN_Attention(attention_type='cbam')
    elif model_type in ['no-attn', 'standard', 'plain']:
        model = FDA_CVNN()  # 无注意力的基线
    else:
        print(f"⚠️ 未知模型类型 '{model_type}'，使用无注意力基线")
        model = FDA_CVNN()
    
    return model.to(device)


def load_model_with_structure(model_type: str, weight_path: str, device):
    """
    正确加载模型：先实例化正确结构，再加载权重
    """
    # 1. 实例化正确结构的模型
    model = get_model_by_type(model_type, device)
    
    # 2. 加载权重
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    
    ckpt = torch.load(weight_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    
    # 处理 DataParallel 的 module. 前缀
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    
    # 尝试 strict=True，如果失败则用 strict=False 并警告
    try:
        model.load_state_dict(sd, strict=True)
        print(f"  ✅ 权重加载成功 (strict=True)")
    except RuntimeError as e:
        print(f"  ⚠️ strict=True 失败，尝试 strict=False: {str(e)[:80]}...")
        model.load_state_dict(sd, strict=False)
    
    return model


def eval_model_rmse(model, device, snr_db: float, L_snapshots: int, num_samples: int):
    model.eval()
    errors_r = []
    errors_theta = []
    times = []

    for _ in range(num_samples):
        r_true = np.random.uniform(0, cfg.r_max)
        theta_true = np.random.uniform(cfg.theta_min, cfg.theta_max)
        R = generate_covariance_matrix(r_true, theta_true, snr_db)
        R_tensor = torch.FloatTensor(R).unsqueeze(0).to(device)

        t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

        if device.type == "cuda":
            t0.record()
        with torch.no_grad():
            pred = model(R_tensor).detach().cpu().numpy()[0]
        if device.type == "cuda":
            t1.record()
            torch.cuda.synchronize()
            ms = t0.elapsed_time(t1)
            times.append(ms / 1000.0)

        r_pred = pred[0] * cfg.r_max
        theta_pred = pred[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min

        errors_r.append((r_pred - r_true) ** 2)
        errors_theta.append((theta_pred - theta_true) ** 2)

    rmse_r = float(np.sqrt(np.mean(errors_r)))
    rmse_theta = float(np.sqrt(np.mean(errors_theta)))
    avg_time = float(np.mean(times)) if times else float("nan")
    return rmse_r, rmse_theta, avg_time


def main():
    parser = argparse.ArgumentParser(description="实验D：消融实验 (加载不同结构+权重对比)")
    parser.add_argument("--snr", type=float, default=-5, help="SNR (dB)")
    parser.add_argument("--snapshots", type=int, default=10, help="快拍数 L")
    parser.add_argument("--num-samples", type=int, default=300, help="每个模型评测样本数")

    # 每个模型需要指定：类型 + 权重路径
    parser.add_argument("--full", type=str, required=True, help="完整模型权重路径 (dual 注意力)")
    parser.add_argument("--full-type", type=str, default="dual", help="完整模型类型 (dual/se/far)")
    parser.add_argument("--no-attn", type=str, required=True, help="无注意力模型权重路径")
    parser.add_argument("--se-only", type=str, default=None, help="只有SE注意力的模型权重路径 (可选)")
    parser.add_argument("--far-only", type=str, default=None, help="只有FAR注意力的模型权重路径 (可选)")

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=" * 60)
    print(f"实验D：消融实验")
    print(f"=" * 60)
    print(f"Device: {device} | SNR={args.snr} dB | L={args.snapshots} | samples={args.num_samples}")

    # 设置快拍数用于数据生成
    cfg.L_snapshots = args.snapshots

    # 构建实验列表: (显示名称, 模型类型, 权重路径)
    experiments = [
        ("Full (Dual)", args.full_type, args.full),
        ("No-Attn", "no-attn", args.no_attn),
    ]
    
    if args.se_only:
        experiments.append(("SE-Only", "se", args.se_only))
    if args.far_only:
        experiments.append(("FAR-Only", "far", args.far_only))

    results = []
    for name, model_type, weight_path in experiments:
        print(f"\n📊 评估: {name} (type={model_type})")
        print(f"   权重: {weight_path}")
        
        # 正确加载：先实例化正确结构，再加载权重
        model = load_model_with_structure(model_type, weight_path, device)
        model.eval()

        rmse_r, rmse_theta, avg_t = eval_model_rmse(model, device, args.snr, args.snapshots, args.num_samples)
        results.append((name, rmse_r, rmse_theta, avg_t))
        print(f"   结果: RMSE_r={rmse_r:.3f}m | RMSE_θ={rmse_theta:.3f}° | time={avg_t*1000:.3f}ms")

    # 绘制柱状图
    labels = [r[0] for r in results]
    rmse_r_vals = [r[1] for r in results]
    rmse_t_vals = [r[2] for r in results]

    x = np.arange(len(labels))
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6'][:len(labels)]

    # 距离 RMSE 柱状图
    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, rmse_r_vals, color=colors)
    plt.xticks(x, labels, fontsize=12)
    plt.ylabel("RMSE Range (m)", fontsize=12)
    plt.title(f"Ablation Study (Range) | SNR={args.snr} dB, L={args.snapshots}", fontsize=14)
    # 在柱子上方显示数值
    for bar, val in zip(bars, rmse_r_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    out1 = os.path.join("results", f"ablation_range_SNR{args.snr}dB_L{args.snapshots}.png")
    plt.savefig(out1, dpi=200)
    plt.close()

    # 角度 RMSE 柱状图
    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, rmse_t_vals, color=colors)
    plt.xticks(x, labels, fontsize=12)
    plt.ylabel("RMSE Angle (deg)", fontsize=12)
    plt.title(f"Ablation Study (Angle) | SNR={args.snr} dB, L={args.snapshots}", fontsize=14)
    for bar, val in zip(bars, rmse_t_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    out2 = os.path.join("results", f"ablation_angle_SNR{args.snr}dB_L{args.snapshots}.png")
    plt.savefig(out2, dpi=200)
    plt.close()

    print(f"\n✅ 图片已保存:")
    print(f"   {out1}")
    print(f"   {out2}")
    
    # 打印汇总表格
    print(f"\n{'='*60}")
    print(f"{'Model':<15} {'RMSE_r (m)':<12} {'RMSE_θ (°)':<12} {'Time (ms)':<10}")
    print(f"{'-'*60}")
    for name, rmse_r, rmse_theta, avg_t in results:
        print(f"{name:<15} {rmse_r:<12.3f} {rmse_theta:<12.3f} {avg_t*1000:<10.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
