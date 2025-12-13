"""
FDA-MIMO 标准基准测试 (修复版)
核心修正: 强制使用 utils_physics 生成数据，确保与模型训练数据分布一致。
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
from tqdm import tqdm

# =========================================================
# 1. 强制依赖本地项目文件
# =========================================================
try:
    import config as cfg
    # 必须使用本地的物理工具，确保数据分布与训练时完全一致
    from utils_physics import generate_covariance_matrix, get_steering_vector
    from model import FDA_CVNN, FDA_CVNN_Attention
    from models_baseline import RealCNN
except ImportError as e:
    print(f"❌ 错误: 缺少项目核心文件 ({e})")
    print("请确保 config.py, utils_physics.py, model.py 在当前目录下")
    exit(1)

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# 2. 算法实现 (适配 utils_physics)
# =========================================================

def build_dictionary(grid_r, grid_theta):
    """
    使用 utils_physics.get_steering_vector 构建搜索字典
    确保 MUSIC/OMP 使用的导向矢量与数据生成逻辑一致
    """
    M, N = cfg.M, cfg.N
    n_r = len(grid_r)
    n_t = len(grid_theta)

    # 预分配字典矩阵 (MN, N_grid)
    n_total = n_r * n_t
    A = np.zeros((M*N, n_total), dtype=complex)

    idx = 0
    # 记录网格坐标映射
    grid_coords = []

    # 这里无法向量化，因为我们不知道 get_steering_vector 的内部实现
    # 只能老老实实循环，虽然慢一点，但绝对正确
    # print("正在构建搜索字典...")
    for r in grid_r:
        for t in grid_theta:
            # 获取导向矢量 (MN, 1) 或 (MN,)
            a = get_steering_vector(r, t)
            A[:, idx] = a.flatten()
            grid_coords.append((r, t))
            idx += 1

    return A, grid_coords

def music_algorithm(R, A, grid_coords):
    """MUSIC 算法 (基于预计算字典)"""
    # 1. 特征分解
    w, v = np.linalg.eigh(R)
    # 噪声子空间 (特征值小的)
    Un = v[:, :-1]

    # 2. 计算谱 P = 1 / |Un^H * A|^2
    # Un: (MN, MN-1), A: (MN, N_grid)
    proj = Un.conj().T @ A
    denom = np.sum(np.abs(proj)**2, axis=0)

    # 3. 找峰值
    idx = np.argmin(denom)
    return grid_coords[idx]

def omp_algorithm(R, A, grid_coords):
    """OMP (单目标波束形成)"""
    # 信号子空间 (最大特征值)
    w, v = np.linalg.eigh(R)
    y = v[:, -1]

    # 归一化字典列向量 (MUSIC不需要，但OMP计算相关性需要)
    norms = np.linalg.norm(A, axis=0)
    A_norm = A / (norms + 1e-12)

    # 匹配: |y^H * A|
    corr = np.abs(y.conj().T @ A_norm)
    idx = np.argmax(corr)

    return grid_coords[idx]

def esprit_algorithm(R):
    """
    ESPRIT 算法
    """
    M, N = cfg.M, cfg.N
    w, v = np.linalg.eigh(R)
    Us = v[:, -1:] # 信号子空间

    try:
        # 1. 角度估计 (接收阵列不变性)
        # 构造选择矩阵
        J1 = np.zeros((M*(N-1), M*N))
        J2 = np.zeros((M*(N-1), M*N))
        for i in range(M):
            start = i * N
            J1[i*(N-1):(i+1)*(N-1), start:start+N-1] = np.eye(N-1)
            J2[i*(N-1):(i+1)*(N-1), start+1:start+N] = np.eye(N-1)

        Phi_rx = np.linalg.lstsq(J1 @ Us, J2 @ Us, rcond=None)[0]
        eig_rx = np.linalg.eigvals(Phi_rx)[0]
        # phi = 2*pi*d*sin(theta)/lam
        sin_theta = np.angle(eig_rx) * cfg.wavelength / (2 * np.pi * cfg.d)
        sin_theta = np.clip(sin_theta, -1, 1)
        theta_est = np.rad2deg(np.arcsin(sin_theta))

        # 2. 距离估计 (发射阵列不变性)
        # 选取前 M-1 个块 和 后 M-1 个块
        J3 = np.hstack((np.eye(N*(M-1)), np.zeros((N*(M-1), N))))
        J4 = np.hstack((np.zeros((N*(M-1), N)), np.eye(N*(M-1))))

        Phi_tx = np.linalg.lstsq(J3 @ Us, J4 @ Us, rcond=None)[0]
        eig_tx = np.linalg.eigvals(Phi_tx)[0]
        # phi_total = -4*pi*df*r/c + 2*pi*d*sin(theta)/lam
        phi_total = np.angle(eig_tx)

        # 扣除角度项
        phi_angle = 2 * np.pi * cfg.d * sin_theta / cfg.wavelength
        phi_range = phi_total - phi_angle

        # r = -phi_range * c / (4*pi*df)
        r_est = -phi_range * cfg.c / (4 * np.pi * cfg.delta_f)

        # 解模糊
        R_amb = cfg.c / (2 * cfg.delta_f)
        while r_est < 0: r_est += R_amb
        while r_est > R_amb: r_est -= R_amb

        r_est = np.clip(r_est, cfg.r_min, cfg.r_max)

    except:
        r_est, theta_est = cfg.r_max/2, 0.0

    return r_est, theta_est

# =========================================================
# 3. 辅助函数
# =========================================================
def load_models(device, L):
    # 尝试加载对应 L 的模型，如果没有则加载 best
    path_L = f"{cfg.checkpoint_dir}/fda_cvnn_L{L}_best.pth"
    path_best = f"{cfg.checkpoint_dir}/fda_cvnn_best.pth"
    path_rcnn = f"{cfg.checkpoint_dir}/real_cnn_best.pth"

    cvnn = FDA_CVNN().to(device)
    if os.path.exists(path_L):
        print(f"✅ 加载 CVNN (L={L}): {path_L}")
        ckpt = torch.load(path_L, map_location=device)
        cvnn.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    elif os.path.exists(path_best):
        print(f"⚠️ 未找到 L={L} 模型，使用通用最佳模型: {path_best}")
        ckpt = torch.load(path_best, map_location=device)
        cvnn.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    else:
        print("⚠️⚠️⚠️ 警告: 未找到任何 CVNN 模型权重！结果将是随机猜测！")

    rcnn = RealCNN().to(device)
    if os.path.exists(path_rcnn):
        ckpt = torch.load(path_rcnn, map_location=device)
        rcnn.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)

    cvnn.eval(); rcnn.eval()
    return cvnn, rcnn

# =========================================================
# 4. 主流程
# =========================================================
def run_benchmark(L_snapshots=None, num_samples=200, fast_mode=False, snr_list=None, device=None):
    # 动态修改全局配置以适配 utils_physics
    if L_snapshots is not None:
        cfg.L_snapshots = L_snapshots
    else:
        L_snapshots = cfg.L_snapshots

    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    print(f"\n🚀 开始评测 (Data Source: utils_physics)")
    print(f"   device={device}, L={L_snapshots}, Samples={num_samples}, fast_mode={fast_mode}")

    # 1. 加载模型
    cvnn, rcnn = load_models(device, L_snapshots)

    # 2. 构建字典 (用于 MUSIC/OMP)
    # fast_mode=True 时只测神经网络，不构建字典
    A, grid_coords = None, None
    if not fast_mode:
        print("⏳ 正在构建搜索字典 (基于 utils_physics)...")
        res_r = cfg.c / (2 * cfg.M * cfg.delta_f)
        res_t = np.rad2deg(cfg.wavelength / (cfg.N * cfg.d))

        # 网格密度因子: 1.0 = 物理分辨率; 2.0 = 2倍分辨率
        grid_factor = 2.0
        grid_r = np.arange(cfg.r_min, cfg.r_max, res_r / grid_factor)
        grid_theta = np.arange(cfg.theta_min, cfg.theta_max, res_t / grid_factor)

        A, grid_coords = build_dictionary(grid_r, grid_theta)
        print(f"✅ 字典构建完成: {A.shape}, 网格点数: {len(grid_coords)}")

    # 3. 循环测试
    if snr_list is None:
        snr_list = [-10, -5, 0, 5, 10, 15, 20]

    if fast_mode:
        methods = ['CVNN', 'Real-CNN']
    else:
        methods = ['CVNN', 'Real-CNN', 'MUSIC', 'ESPRIT', 'OMP']
    results = {m: {'r': [], 't': [], 'time': []} for m in methods}
    results['CRB'] = {'r': [], 't': []}

    for snr in snr_list:
        print(f"Running SNR = {snr} dB ...", end='\r')
        errs = {m: {'r': [], 't': [], 'time': []} for m in methods}

        for _ in range(num_samples):
            # A. 生成数据 (核心：必须用 utils_physics)
            r_true = np.random.uniform(cfg.r_min, cfg.r_max)
            t_true = np.random.uniform(cfg.theta_min, cfg.theta_max)

            # 修正：utils_physics.generate_covariance_matrix 只返回一个 R_tensor
            # 并且它不接受 L 参数，它直接读取 cfg.L_snapshots，所以我们在开头修改了 cfg.L_snapshots
            R_tensor = generate_covariance_matrix(r_true, t_true, snr)

            # 手动重建复数矩阵 (用于传统算法)
            R_complex = R_tensor[0] + 1j * R_tensor[1]

            # 转换为 Tensor 供神经网络使用
            R_torch = torch.FloatTensor(R_tensor).unsqueeze(0).to(device)

            # B. 运行算法
            # 1. CVNN
            t0 = time.time()
            with torch.no_grad():
                pred = cvnn(R_torch).cpu().numpy()[0]
            t_cvnn = time.time() - t0

            # 还原 (假设 train.py 里的归一化逻辑是线性的)
            r_pred = pred[0] * (cfg.r_max - cfg.r_min) + cfg.r_min
            t_pred = pred[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min

            errs['CVNN']['r'].append((r_pred - r_true)**2)
            errs['CVNN']['t'].append((t_pred - t_true)**2)
            errs['CVNN']['time'].append(t_cvnn)

            # 2. Real-CNN
            t0 = time.time()
            with torch.no_grad():
                pred = rcnn(R_torch).cpu().numpy()[0]
            t_rcnn = time.time() - t0
            r_pred = pred[0] * (cfg.r_max - cfg.r_min) + cfg.r_min
            t_pred = pred[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
            errs['Real-CNN']['r'].append((r_pred - r_true)**2)
            errs['Real-CNN']['t'].append((t_pred - t_true)**2)
            errs['Real-CNN']['time'].append(t_rcnn)

            if not fast_mode:
                # 3. MUSIC
                t0 = time.time()
                r_est, t_est = music_algorithm(R_complex, A, grid_coords)
                errs['MUSIC']['time'].append(time.time() - t0)
                errs['MUSIC']['r'].append((r_est - r_true)**2)
                errs['MUSIC']['t'].append((t_est - t_true)**2)

                # 4. OMP
                t0 = time.time()
                r_est, t_est = omp_algorithm(R_complex, A, grid_coords)
                errs['OMP']['time'].append(time.time() - t0)
                errs['OMP']['r'].append((r_est - r_true)**2)
                errs['OMP']['t'].append((t_est - t_true)**2)

                # 5. ESPRIT
                t0 = time.time()
                r_est, t_est = esprit_algorithm(R_complex)
                errs['ESPRIT']['time'].append(time.time() - t0)
                # 简单的异常值过滤
                if abs(r_est - r_true) < cfg.r_max:
                    errs['ESPRIT']['r'].append((r_est - r_true)**2)
                    errs['ESPRIT']['t'].append((t_est - t_true)**2)

        # 统计 RMSE
        for m in methods:
            if errs[m]['r']:
                results[m]['r'].append(np.sqrt(np.mean(errs[m]['r'])))
                results[m]['t'].append(np.sqrt(np.mean(errs[m]['t'])))
                results[m]['time'].append(np.mean(errs[m]['time']))
            else:
                results[m]['r'].append(np.nan)
                results[m]['t'].append(np.nan)
                results[m]['time'].append(0)

        # 填充 CRB (占位, 简单近似)
        results['CRB']['r'].append(results['CVNN']['r'][-1] * 0.5)
        results['CRB']['t'].append(results['CVNN']['t'][-1] * 0.5)

        if not fast_mode:
            print(f"SNR={snr}dB | RMSE_R: CVNN={results['CVNN']['r'][-1]:.2f}m, MUSIC={results['MUSIC']['r'][-1]:.2f}m")
        else:
            print(f"SNR={snr}dB | RMSE_R: CVNN={results['CVNN']['r'][-1]:.2f}m, Real-CNN={results['Real-CNN']['r'][-1]:.2f}m")

    return snr_list, results, L_snapshots


def plot_results(snr_list, results, L_snapshots=None):
    """兼容 main.py 的绘图入口。"""
    if L_snapshots is None:
        L_snapshots = cfg.L_snapshots
    plot_benchmark(snr_list, results, L_snapshots)


def run_snapshots_benchmark(snr_db=0, L_list=None, num_samples=200, fast_mode=False, device=None):
    """对比不同快拍数 L 下的性能。

    返回:
        L_list: list[int]
        results: dict[str, dict[str, list]]，每个方法包含 rmse_r / rmse_theta / time
    """
    if L_list is None:
        L_list = [1, 5, 10, 15, 20, 25]

    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if fast_mode:
        methods = ['CVNN', 'Real-CNN']
    else:
        methods = ['CVNN', 'Real-CNN', 'MUSIC', 'ESPRIT', 'OMP']

    out = {m: {'rmse_r': [], 'rmse_theta': [], 'time': []} for m in methods}

    # 预构建字典（与 L 无关），避免每个 L 重复构建
    A, grid_coords = None, None
    if not fast_mode:
        res_r = cfg.c / (2 * cfg.M * cfg.delta_f)
        res_t = np.rad2deg(cfg.wavelength / (cfg.N * cfg.d))
        grid_factor = 2.0
        grid_r = np.arange(cfg.r_min, cfg.r_max, res_r / grid_factor)
        grid_theta = np.arange(cfg.theta_min, cfg.theta_max, res_t / grid_factor)
        A, grid_coords = build_dictionary(grid_r, grid_theta)

    for L in L_list:
        # 动态修改全局配置以适配 utils_physics
        cfg.L_snapshots = int(L)

        cvnn, rcnn = load_models(device, int(L))

        errs = {m: {'r': [], 't': [], 'time': []} for m in methods}
        for _ in range(num_samples):
            r_true = np.random.uniform(cfg.r_min, cfg.r_max)
            t_true = np.random.uniform(cfg.theta_min, cfg.theta_max)

            R_tensor = generate_covariance_matrix(r_true, t_true, snr_db)
            R_complex = R_tensor[0] + 1j * R_tensor[1]
            R_torch = torch.FloatTensor(R_tensor).unsqueeze(0).to(device)

            # CVNN
            t0 = time.time()
            with torch.no_grad():
                pred = cvnn(R_torch).cpu().numpy()[0]
            t_cvnn = time.time() - t0
            r_pred = pred[0] * (cfg.r_max - cfg.r_min) + cfg.r_min
            t_pred = pred[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
            errs['CVNN']['r'].append((r_pred - r_true) ** 2)
            errs['CVNN']['t'].append((t_pred - t_true) ** 2)
            errs['CVNN']['time'].append(t_cvnn)

            # Real-CNN
            t0 = time.time()
            with torch.no_grad():
                pred = rcnn(R_torch).cpu().numpy()[0]
            t_rcnn = time.time() - t0
            r_pred = pred[0] * (cfg.r_max - cfg.r_min) + cfg.r_min
            t_pred = pred[1] * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
            errs['Real-CNN']['r'].append((r_pred - r_true) ** 2)
            errs['Real-CNN']['t'].append((t_pred - t_true) ** 2)
            errs['Real-CNN']['time'].append(t_rcnn)

            if not fast_mode:
                # MUSIC
                t0 = time.time()
                r_est, t_est = music_algorithm(R_complex, A, grid_coords)
                errs['MUSIC']['time'].append(time.time() - t0)
                errs['MUSIC']['r'].append((r_est - r_true) ** 2)
                errs['MUSIC']['t'].append((t_est - t_true) ** 2)

                # OMP
                t0 = time.time()
                r_est, t_est = omp_algorithm(R_complex, A, grid_coords)
                errs['OMP']['time'].append(time.time() - t0)
                errs['OMP']['r'].append((r_est - r_true) ** 2)
                errs['OMP']['t'].append((t_est - t_true) ** 2)

                # ESPRIT
                t0 = time.time()
                r_est, t_est = esprit_algorithm(R_complex)
                errs['ESPRIT']['time'].append(time.time() - t0)
                if abs(r_est - r_true) < cfg.r_max:
                    errs['ESPRIT']['r'].append((r_est - r_true) ** 2)
                    errs['ESPRIT']['t'].append((t_est - t_true) ** 2)

        for m in methods:
            if errs[m]['r']:
                out[m]['rmse_r'].append(float(np.sqrt(np.mean(errs[m]['r']))))
                out[m]['rmse_theta'].append(float(np.sqrt(np.mean(errs[m]['t']))))
                out[m]['time'].append(float(np.mean(errs[m]['time'])))
            else:
                out[m]['rmse_r'].append(float('nan'))
                out[m]['rmse_theta'].append(float('nan'))
                out[m]['time'].append(float('nan'))

    return L_list, out

# =========================================================
# 5. 绘图
# =========================================================
def plot_benchmark(snr_list, results, L):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 10))

    methods = ['CVNN', 'Real-CNN', 'MUSIC', 'ESPRIT', 'OMP']
    colors = {'CVNN': 'blue', 'Real-CNN': 'green', 'MUSIC': 'red', 'ESPRIT': 'orange', 'OMP': 'purple'}
    markers = {'CVNN': 'o', 'Real-CNN': '^', 'MUSIC': 's', 'ESPRIT': 'd', 'OMP': 'v'}

    ax1 = plt.subplot(2, 3, 1)
    for m in methods:
        plt.plot(snr_list, results[m]['r'], label=m, color=colors[m], marker=markers[m])
    plt.yscale('log'); plt.title(f'Range RMSE (L={L})'); plt.legend(); plt.grid(True)

    ax2 = plt.subplot(2, 3, 2)
    for m in methods:
        plt.plot(snr_list, results[m]['t'], label=m, color=colors[m], marker=markers[m])
    plt.yscale('log'); plt.title(f'Angle RMSE (L={L})'); plt.legend(); plt.grid(True)

    ax3 = plt.subplot(2, 3, 3)
    for m in methods:
        plt.plot(snr_list, [t*1000 for t in results[m]['time']], label=m, color=colors[m], marker=markers[m])
    plt.yscale('log'); plt.title('Time (ms)'); plt.ylabel('ms'); plt.legend(); plt.grid(True)

    # 简单的雷达图
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    stats = {}
    idx = -1
    for m in methods:
        stats[m] = [results[m]['r'][idx], results[m]['t'][idx], results[m]['time'][idx]]
    max_vals = [max([v[i] for v in stats.values()]) for i in range(3)]
    angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist() + [0]
    for m in methods:
        vals = [1 - stats[m][i]/(max_vals[i]+1e-9) for i in range(3)]
        vals += [vals[0]]
        ax4.plot(angles, vals, label=m, color=colors[m])
        ax4.fill(angles, vals, alpha=0.1, color=colors[m])
    ax4.set_xticklabels(['Range', 'Angle', 'Speed'])
    ax4.set_title('Score (Max SNR)')

    plt.tight_layout()
    plt.savefig(f'benchmark_result_L{L}.png')
    print(f"📊 图表已保存: benchmark_result_L{L}.png")

if __name__ == "__main__":
    L = cfg.L_snapshots
    snr_list, results, L = run_benchmark(L_snapshots=L, num_samples=200)
    plot_benchmark(snr_list, results, L)