import os
import time
import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from utils_physics import get_steering_vector


def generate_two_targets_covariance(
    r1: float,
    theta1: float,
    r2: float,
    theta2: float,
    snr_db: float,
    L: int = 1,
    correlated: bool = False,
):
    """Generate sample covariance for two targets in FDA-MIMO.

    Uses steering vectors from existing project code to stay consistent.

    Args:
        correlated: if True, uses identical source signals (fully coherent).
                    if False, uses independent complex Gaussian sources.

    Returns:
        R: complex covariance matrix, shape (MN, MN)
    """
    M, N = cfg.M, cfg.N
    MN = M * N

    a1 = get_steering_vector(r1, theta1).reshape(MN, 1)
    a2 = get_steering_vector(r2, theta2).reshape(MN, 1)
    A = np.concatenate([a1, a2], axis=1)  # (MN, 2)

    if correlated:
        s = (np.random.randn(1, L) + 1j * np.random.randn(1, L)) / np.sqrt(2)
        S = np.concatenate([s, s], axis=0)  # (2, L)
    else:
        S = (np.random.randn(2, L) + 1j * np.random.randn(2, L)) / np.sqrt(2)

    Xs = A @ S  # (MN, L)

    # Scale noise to achieve target SNR (per-snapshot average power)
    signal_power = np.mean(np.abs(Xs) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-12)

    Noise = (
        np.random.randn(MN, L) + 1j * np.random.randn(MN, L)
    ) / np.sqrt(2) * np.sqrt(noise_power)

    X = Xs + Noise
    R = (X @ X.conj().T) / L
    return R


def music_spectrum_2d(R: np.ndarray, r_grid: np.ndarray, theta_grid: np.ndarray, K_targets: int = 2):
    """Compute 2D MUSIC spectrum on a grid.

    Note: When L < K_targets, the sample covariance is rank-deficient.
          This is exactly the point of the experiment.
    """
    w, v = np.linalg.eigh(R)

    # Noise subspace: smallest MN-K eigenvectors
    # If K_targets is too large for the effective rank, this becomes ill-posed.
    Un = v[:, :-K_targets]

    # Vectorized dictionary
    M, N = cfg.M, cfg.N
    Rg, Tg = np.meshgrid(r_grid, theta_grid, indexing="ij")
    R_flat = Rg.reshape(-1)
    T_flat = Tg.reshape(-1)

    m_idx = np.arange(M).reshape(-1, 1)
    n_idx = np.arange(N).reshape(-1, 1)
    T_rad = np.deg2rad(T_flat)

    phi_tx = (
        -4 * np.pi * cfg.delta_f * m_idx * R_flat / cfg.c
        + 2 * np.pi * cfg.d * m_idx * np.sin(T_rad) / cfg.wavelength
    )
    a_tx = np.exp(1j * phi_tx)

    phi_rx = 2 * np.pi * cfg.d * n_idx * np.sin(T_rad) / cfg.wavelength
    a_rx = np.exp(1j * phi_rx)

    A = (a_tx[:, np.newaxis, :] * a_rx[np.newaxis, :, :]).reshape(M * N, -1)

    proj = Un.conj().T @ A
    denom = np.sum(np.abs(proj) ** 2, axis=0)
    P = 1.0 / (denom + 1e-12)
    return P.reshape(len(r_grid), len(theta_grid))


def build_dynamic_grid():
    """Use the same physics-based grid idea as benchmark, with safe caps."""
    res_r = cfg.c / (2 * cfg.M * cfg.delta_f)
    res_theta = np.rad2deg(cfg.wavelength / (cfg.N * cfg.d))

    step_r = res_r / 2
    step_theta = res_theta / 2

    num_r = max(int(cfg.r_max / step_r) + 1, 50)
    num_theta = max(int((cfg.theta_max - cfg.theta_min) / step_theta) + 1, 30)

    # Cap to keep memory/time reasonable for spectrum plotting
    num_r = min(num_r, 220)
    num_theta = min(num_theta, 140)

    r_grid = np.linspace(0, cfg.r_max, num_r)
    theta_grid = np.linspace(cfg.theta_min, cfg.theta_max, num_theta)
    return r_grid, theta_grid, res_r, res_theta


def run_resolution_experiment(
    snr_db: float = 20,
    L_list=(1, 20),
    theta_sep_deg: float = 3.0,
    correlated: bool = False,
):
    os.makedirs("results", exist_ok=True)

    # Pick two targets within configured bounds
    r_center = 0.45 * cfg.r_max
    dr = 0.05 * cfg.r_max
    r1 = float(np.clip(r_center - dr, 0, cfg.r_max))
    r2 = float(np.clip(r_center + dr, 0, cfg.r_max))

    t_center = (cfg.theta_min + cfg.theta_max) / 2
    theta1 = float(np.clip(t_center - theta_sep_deg / 2, cfg.theta_min, cfg.theta_max))
    theta2 = float(np.clip(t_center + theta_sep_deg / 2, cfg.theta_min, cfg.theta_max))

    r_grid, theta_grid, res_r, res_theta = build_dynamic_grid()

    print("=" * 70)
    print("实验C：双目标分辨 (MUSIC 失效展示)")
    print("=" * 70)
    print(f"Targets: (r1={r1:.1f}m, θ1={theta1:.2f}°), (r2={r2:.1f}m, θ2={theta2:.2f}°)")
    print(f"SNR={snr_db} dB | θ separation={theta_sep_deg:.2f}° | correlated={correlated}")
    print(f"Grid: {len(r_grid)}×{len(theta_grid)} | Res: Range={res_r:.2f}m Angle={res_theta:.2f}°")

    for L in L_list:
        t0 = time.time()
        R = generate_two_targets_covariance(r1, theta1, r2, theta2, snr_db=snr_db, L=L, correlated=correlated)

        # Intentionally set K_targets=2
        P = music_spectrum_2d(R, r_grid, theta_grid, K_targets=2)
        P_db = 10 * np.log10(P / (np.max(P) + 1e-12) + 1e-12)

        plt.figure(figsize=(10, 6))
        extent = [theta_grid[0], theta_grid[-1], r_grid[0], r_grid[-1]]
        plt.imshow(P_db, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        plt.colorbar(label="MUSIC Spectrum (dB, normalized)")
        plt.scatter([theta1, theta2], [r1, r2], c="r", s=60, marker="x", label="True Targets")
        plt.xlabel("Angle (deg)")
        plt.ylabel("Range (m)")
        plt.title(f"2-Target MUSIC Spectrum | L={L}, SNR={snr_db} dB")
        plt.legend(loc="upper right")
        plt.tight_layout()

        out_path = os.path.join("results", f"resolution_music_L{L}_SNR{snr_db}dB.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved: {out_path} | time={time.time()-t0:.2f}s")


if __name__ == "__main__":
    run_resolution_experiment(snr_db=20, L_list=(1, 20), theta_sep_deg=3.0, correlated=False)
