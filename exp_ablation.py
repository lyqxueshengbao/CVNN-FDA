import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import config as cfg
from benchmark import load_cvnn_model
from utils_physics import generate_covariance_matrix


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
    parser = argparse.ArgumentParser(description="实验D：消融实验 (加载不同权重对比)")
    parser.add_argument("--snr", type=float, default=-5, help="SNR (dB)")
    parser.add_argument("--snapshots", type=int, default=10, help="快拍数 L")
    parser.add_argument("--num-samples", type=int, default=300, help="每个模型评测样本数")

    parser.add_argument("--full", type=str, required=True, help="完整模型权重路径")
    parser.add_argument("--no-attn", type=str, required=True, help="去注意力模型权重路径")
    parser.add_argument("--no-cst", type=str, required=False, default=None, help="去 CST 模块模型权重路径 (可选)")

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | SNR={args.snr} dB | L={args.snapshots} | samples={args.num_samples}")

    # Temporarily set snapshots for data generation
    cfg.L_snapshots = args.snapshots

    paths = [("Full", args.full), ("No-Attn", args.no_attn)]
    if args.no_cst:
        paths.append(("No-CST", args.no_cst))

    results = []
    for name, path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        model = load_cvnn_model(device, L_snapshots=args.snapshots)
        ckpt = torch.load(path, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)

        rmse_r, rmse_theta, avg_t = eval_model_rmse(model, device, args.snr, args.snapshots, args.num_samples)
        results.append((name, rmse_r, rmse_theta, avg_t))
        print(f"{name:8s} | RMSE_r={rmse_r:.3f} m | RMSE_theta={rmse_theta:.3f} deg | time={avg_t:.6f} s")

    # Plot bar chart (range + angle)
    labels = [r[0] for r in results]
    rmse_r_vals = [r[1] for r in results]
    rmse_t_vals = [r[2] for r in results]

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 4))
    plt.bar(x, rmse_r_vals)
    plt.xticks(x, labels)
    plt.ylabel("RMSE Range (m)")
    plt.title(f"Ablation (Range) | SNR={args.snr} dB, L={args.snapshots}")
    plt.tight_layout()
    out1 = os.path.join("results", f"ablation_range_SNR{args.snr}dB_L{args.snapshots}.png")
    plt.savefig(out1, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(x, rmse_t_vals)
    plt.xticks(x, labels)
    plt.ylabel("RMSE Angle (deg)")
    plt.title(f"Ablation (Angle) | SNR={args.snr} dB, L={args.snapshots}")
    plt.tight_layout()
    out2 = os.path.join("results", f"ablation_angle_SNR{args.snr}dB_L{args.snapshots}.png")
    plt.savefig(out2, dpi=200)
    plt.close()

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
