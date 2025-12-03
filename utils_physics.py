"""
FDA-MIMO 物理信号生成与协方差矩阵计算
"""
import numpy as np
import torch
import config as cfg


def generate_batch_torch(batch_size, device=cfg.device, snr_range=None, L_range=None):
    """
    在GPU上直接生成批次数据 (极速模式)
    
    参数:
        batch_size: 批次大小
        device: 设备
        snr_range: (min, max) SNR范围
        L_range: (min, max) 快拍数范围，None则使用固定值 cfg.L_snapshots
                 设置此参数可训练出对不同快拍数都鲁棒的模型
    
    返回:
        R_tensor: [B, 2, MN, MN]
        labels: [B, 2]
    """
    # 物理常量
    c = cfg.c
    delta_f = cfg.delta_f
    d = cfg.d
    wavelength = cfg.wavelength
    M = cfg.M
    N = cfg.N
    
    # 快拍数：固定值或随机范围
    if L_range is not None:
        L_min, L_max = L_range
        L = np.random.randint(L_min, L_max + 1)  # 随机选择快拍数
    else:
        L = cfg.L_snapshots
    
    # 1. 随机生成参数 [B, 1]
    r = torch.rand(batch_size, 1, device=device) * (cfg.r_max - cfg.r_min) + cfg.r_min
    theta = torch.rand(batch_size, 1, device=device) * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
    
    if snr_range is None:
        snr_min, snr_max = cfg.snr_train_min, cfg.snr_train_max
    else:
        snr_min, snr_max = snr_range
        
    snr_db = torch.rand(batch_size, 1, device=device) * (snr_max - snr_min) + snr_min
    
    theta_rad = torch.deg2rad(theta)
    
    # 2. 构建导向矢量
    # m: [1, M], n: [1, N]
    m = torch.arange(M, device=device).float().unsqueeze(0)
    n = torch.arange(N, device=device).float().unsqueeze(0)
    
    # 发射导向矢量 a_tx: [B, M]
    # 相位1: 距离项 -4π * Δf * m * r / c
    phi_range = -4 * torch.pi * delta_f * m * r / c
    # 相位2: 角度项 2π * d * m * sin(θ) / λ
    phi_angle_tx = 2 * torch.pi * d * m * torch.sin(theta_rad) / wavelength
    a_tx = torch.exp(1j * (phi_range + phi_angle_tx))
    
    # 接收导向矢量 a_rx: [B, N]
    phi_angle_rx = 2 * torch.pi * d * n * torch.sin(theta_rad) / wavelength
    a_rx = torch.exp(1j * phi_angle_rx)
    
    # 联合导向矢量 u = a_tx ⊗ a_rx: [B, MN]
    # 利用广播机制: [B, M, 1] * [B, 1, N] -> [B, M, N] -> [B, MN]
    u = (a_tx.unsqueeze(2) * a_rx.unsqueeze(1)).view(batch_size, -1)
    u = u.unsqueeze(2) # [B, MN, 1]
    
    # 3. 生成信号 s: [B, 1, L]
    # 复高斯信号
    s_real = torch.randn(batch_size, 1, L, device=device)
    s_imag = torch.randn(batch_size, 1, L, device=device)
    s = (s_real + 1j * s_imag) / np.sqrt(2)
    
    # 纯信号 X_clean: [B, MN, L]
    X_clean = torch.matmul(u, s)
    
    # 4. 生成噪声 noise: [B, MN, L]
    n_real = torch.randn(batch_size, M*N, L, device=device)
    n_imag = torch.randn(batch_size, M*N, L, device=device)
    noise = (n_real + 1j * n_imag) / np.sqrt(2)
    
    # 5. 计算功率并混合
    # 信号功率 (对 MN 和 L 维度求平均)
    power_sig = torch.mean(torch.abs(X_clean)**2, dim=(1,2), keepdim=True)
    power_noise = power_sig / (10**(snr_db.unsqueeze(2)/10.0))
    
    X = X_clean + torch.sqrt(power_noise) * noise
    
    # 6. 计算协方差矩阵 R: [B, MN, MN]
    # R = X @ X^H / L
    R = torch.matmul(X, X.transpose(1, 2).conj()) / L
    
    # 7. 归一化
    max_val = torch.amax(torch.abs(R), dim=(1,2), keepdim=True)
    R = R / (max_val + 1e-10)
    
    # 8. 转换为实部/虚部通道: [B, 2, MN, MN]
    R_tensor = torch.stack([R.real, R.imag], dim=1).float()
    
    # 9. 标签归一化
    r_norm = r / cfg.r_max
    theta_norm = (theta - cfg.theta_min) / (cfg.theta_max - cfg.theta_min)
    labels = torch.cat([r_norm, theta_norm], dim=1).float()
    
    return R_tensor, labels


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
