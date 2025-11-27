"""
FDA-MIMO 物理信号生成与协方差矩阵计算
"""
import numpy as np
import config as cfg


def get_steering_vector(r, theta):
    """
    生成 FDA-MIMO 联合导向矢量 (MN x 1)
    
    参数:
        r: 目标距离 (m)
        theta: 目标角度 (度)
    
    返回:
        u: 联合导向矢量，形状 (MN,)
    """
    # 角度转弧度
    theta_rad = np.deg2rad(theta)
    
    # 1. 发射导向矢量 (包含 FDA 距离相位项)
    m = np.arange(cfg.M)
    # FDA距离相位: -4π * m * Δf * r / c (双程)
    phi_range = -4 * np.pi * cfg.delta_f * m * r / cfg.c
    # 角度相位: 2π * d * m * sin(θ) / λ
    phi_angle_tx = 2 * np.pi * cfg.d * m * np.sin(theta_rad) / cfg.wavelength
    a_tx = np.exp(1j * (phi_range + phi_angle_tx))
    
    # 2. 接收导向矢量 (仅角度相关)
    n = np.arange(cfg.N)
    phi_angle_rx = 2 * np.pi * cfg.d * n * np.sin(theta_rad) / cfg.wavelength
    a_rx = np.exp(1j * phi_angle_rx)
    
    # 3. 联合导向矢量 (Kronecker积)
    u = np.kron(a_tx, a_rx)
    return u


def generate_covariance_matrix(r, theta, snr_db):
    """
    生成单样本的协方差矩阵
    
    参数:
        r: 目标距离 (m)
        theta: 目标角度 (度)
        snr_db: 信噪比 (dB)
    
    返回:
        R_tensor: 协方差矩阵，形状 [2, MN, MN] (实部, 虚部)
    """
    # 导向矢量
    u = get_steering_vector(r, theta).reshape(-1, 1)  # (MN, 1)
    
    # 信号源 (L个快拍，复高斯随机)
    s = (np.random.randn(1, cfg.L_snapshots) + 
         1j * np.random.randn(1, cfg.L_snapshots)) / np.sqrt(2)
    
    # 纯信号
    X_clean = u @ s  # (MN, L)
    
    # 加噪声
    noise = (np.random.randn(cfg.MN, cfg.L_snapshots) + 
             1j * np.random.randn(cfg.MN, cfg.L_snapshots)) / np.sqrt(2)
    
    # 计算噪声功率
    power_sig = np.mean(np.abs(X_clean) ** 2)
    power_noise = power_sig / (10 ** (snr_db / 10.0))
    
    # 含噪信号
    X = X_clean + np.sqrt(power_noise) * noise
    
    # 计算协方差矩阵 R = X * X^H / L
    R = X @ X.conj().T / cfg.L_snapshots
    
    # 归一化 (重要！神经网络对幅度敏感)
    R = R / (np.max(np.abs(R)) + 1e-10)
    
    # 转换为 Tensor [2, MN, MN] (Real, Imag)
    R_tensor = np.stack([R.real, R.imag], axis=0).astype(np.float32)
    return R_tensor


def generate_batch(batch_size, snr_db=None, snr_range=None):
    """
    生成一个批次的数据
    
    参数:
        batch_size: 批次大小
        snr_db: 固定SNR (如果指定则使用固定值)
        snr_range: SNR范围 (min, max)，用于随机采样
    
    返回:
        X: 输入数据，形状 [B, 2, MN, MN]
        Y: 标签，形状 [B, 2] (归一化的 r 和 theta)
    """
    X_batch = []
    Y_batch = []
    
    for _ in range(batch_size):
        # 随机生成目标参数
        r = np.random.uniform(cfg.r_min, cfg.r_max)
        theta = np.random.uniform(cfg.theta_min, cfg.theta_max)
        
        # SNR
        if snr_db is not None:
            snr = snr_db
        elif snr_range is not None:
            snr = np.random.uniform(snr_range[0], snr_range[1])
        else:
            snr = np.random.uniform(cfg.snr_train_min, cfg.snr_train_max)
        
        # 生成协方差矩阵
        R = generate_covariance_matrix(r, theta, snr)
        
        # 归一化标签到 [0, 1]
        r_norm = r / cfg.r_max
        theta_norm = (theta - cfg.theta_min) / (cfg.theta_max - cfg.theta_min)
        
        X_batch.append(R)
        Y_batch.append([r_norm, theta_norm])
    
    return np.array(X_batch), np.array(Y_batch, dtype=np.float32)


def denormalize_labels(y_norm):
    """
    将归一化标签转换回物理单位
    
    参数:
        y_norm: 归一化标签 [r_norm, theta_norm] 或 [B, 2]
    
    返回:
        y_physical: 物理单位 [r(m), theta(度)]
    """
    y_norm = np.array(y_norm)
    if y_norm.ndim == 1:
        r = y_norm[0] * cfg.r_max
        theta = y_norm[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
        return np.array([r, theta])
    else:
        r = y_norm[:, 0] * cfg.r_max
        theta = y_norm[:, 1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
        return np.stack([r, theta], axis=1)


if __name__ == "__main__":
    # 测试信号生成
    print("测试信号生成...")
    
    # 测试导向矢量
    r_test, theta_test = 1000.0, 30.0
    u = get_steering_vector(r_test, theta_test)
    print(f"导向矢量形状: {u.shape}")
    print(f"导向矢量模值范围: [{np.abs(u).min():.4f}, {np.abs(u).max():.4f}]")
    
    # 测试协方差矩阵
    R = generate_covariance_matrix(r_test, theta_test, snr_db=20)
    print(f"\n协方差矩阵形状: {R.shape}")
    print(f"实部范围: [{R[0].min():.4f}, {R[0].max():.4f}]")
    print(f"虚部范围: [{R[1].min():.4f}, {R[1].max():.4f}]")
    
    # 测试批次生成
    X, Y = generate_batch(4, snr_db=20)
    print(f"\n批次数据形状: X={X.shape}, Y={Y.shape}")
    
    # 测试反归一化
    Y_physical = denormalize_labels(Y)
    print(f"物理标签示例:")
    for i in range(4):
        print(f"  样本{i+1}: r={Y_physical[i,0]:.1f}m, θ={Y_physical[i,1]:.1f}°")
