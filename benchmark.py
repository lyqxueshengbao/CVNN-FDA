"""
FDA-MIMO 雷达参数估计对比实验脚本
对比算法：
1. Proposed CVNN (复数神经网络)
2. 2D-MUSIC (传统超分辨算法)
3. Real-CNN (实数神经网络基线)
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
from tqdm import tqdm

import config as cfg
from model import FDA_CVNN
from models_baseline import RealCNN
from utils_physics import generate_covariance_matrix, get_steering_vector

# ==========================================
# 1. 2D-MUSIC 算法实现
# ==========================================
def music_2d(R, r_search, theta_search):
    """
    简化的2D-MUSIC算法
    R: 协方差矩阵 (MN, MN)
    """
    # 1. 特征分解
    w, v = np.linalg.eigh(R)
    # 排序 (小到大)，取前 MN-K 个为噪声子空间
    # 假设单目标 K=1
    Un = v[:, :-1] 
    
    # 2. 构造空间谱
    # P(r, theta) = 1 / |a(r,theta)^H * Un * Un^H * a(r,theta)|
    
    # 预计算噪声投影矩阵 Pn = Un * Un^H
    Pn = Un @ Un.conj().T
    
    max_p = -1
    best_r = 0
    best_theta = 0
    
    # 网格搜索
    for r in r_search:
        for theta in theta_search:
            # 生成导向矢量
            a = get_steering_vector(r, theta)
            
            # 计算谱值
            # denom = a.conj().T @ Pn @ a
            # 优化计算: |Un^H * a|^2
            proj = Un.conj().T @ a
            denom = np.sum(np.abs(proj)**2)
            
            if denom < 1e-10:
                spectrum = 1e10
            else:
                spectrum = 1.0 / denom
            
            if spectrum > max_p:
                max_p = spectrum
                best_r = r
                best_theta = theta
                
    return best_r, best_theta

# ==========================================
# 2. 运行对比实验
# ==========================================
def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载 CVNN 模型
    cvnn = FDA_CVNN().to(device)
    cvnn_path = "checkpoints/fda_cvnn_best.pth"
    if os.path.exists(cvnn_path):
        try:
            checkpoint = torch.load(cvnn_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                cvnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                cvnn.load_state_dict(checkpoint)
            print(f"成功加载 CVNN 权重: {cvnn_path}")
        except Exception as e:
            print(f"加载 CVNN 权重失败: {e}，使用随机权重")
    else:
        print("未找到 CVNN 权重文件，使用随机权重演示...")
    cvnn.eval()
    
    # 2. 加载 Real-CNN 模型 (如果有)
    real_cnn = RealCNN().to(device)
    real_cnn_path = "checkpoints/real_cnn_best.pth"
    has_real_cnn = False
    if os.path.exists(real_cnn_path):
        try:
            checkpoint = torch.load(real_cnn_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                real_cnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                real_cnn.load_state_dict(checkpoint)
            print(f"成功加载 Real-CNN 权重: {real_cnn_path}")
            has_real_cnn = True
        except:
            pass
    if not has_real_cnn:
        print("未找到 Real-CNN 权重，将使用随机权重参与对比 (仅供参考速度)")
    real_cnn.eval()
    
    # 参数设置
    snr_list = [-10, 0, 10, 20, 30]
    num_samples = 50  # 每个SNR测试样本数 (MUSIC太慢，设小一点)
    
    # 结果存储
    results = {
        "CVNN": {"rmse_r": [], "time": []},
        "MUSIC": {"rmse_r": [], "time": []},
        "Real-CNN": {"rmse_r": [], "time": []}
    }
    
    # MUSIC 搜索网格 (降低精度以加快速度)
    # 真实应用中 MUSIC 需要更细的网格，速度会更慢
    r_grid = np.linspace(0, 2000, 50)   # 40m 步长
    theta_grid = np.linspace(-60, 60, 30) # 4度 步长
    
    print(f"\n开始对比实验 (样本数={num_samples})...")
    print(f"MUSIC 网格大小: {len(r_grid)}x{len(theta_grid)} = {len(r_grid)*len(theta_grid)} 点")
    
    for snr in snr_list:
        print(f"\n正在测试 SNR = {snr} dB ...")
        
        err_cvnn = []
        err_music = []
        err_real = []
        
        t_cvnn = []
        t_music = []
        t_real = []
        
        for _ in tqdm(range(num_samples)):
            # 生成数据
            r_true = np.random.uniform(0, 2000)
            theta_true = np.random.uniform(-60, 60)
            R = generate_covariance_matrix(r_true, theta_true, snr)
            
            # --- 1. 测试 CVNN ---
            t0 = time.time()
            R_tensor = torch.FloatTensor(R).unsqueeze(0).to(device) # [1, 2, MN, MN]
            with torch.no_grad():
                pred = cvnn(R_tensor).cpu().numpy()[0]
            r_pred_cvnn = pred[0] * 2000
            t1 = time.time()
            
            err_cvnn.append((r_pred_cvnn - r_true)**2)
            t_cvnn.append(t1 - t0)
            
            # --- 2. 测试 Real-CNN ---
            t0 = time.time()
            with torch.no_grad():
                pred_real = real_cnn(R_tensor).cpu().numpy()[0]
            r_pred_real = pred_real[0] * 2000
            t1 = time.time()
            
            err_real.append((r_pred_real - r_true)**2)
            t_real.append(t1 - t0)
            
            # --- 3. 测试 MUSIC ---
            # 恢复复数矩阵
            R_complex = R[0] + 1j * R[1]
            
            t0 = time.time()
            r_pred_music, _ = music_2d(R_complex, r_grid, theta_grid)
            t1 = time.time()
            
            err_music.append((r_pred_music - r_true)**2)
            t_music.append(t1 - t0)
            
        # 计算 RMSE
        rmse_cvnn = np.sqrt(np.mean(err_cvnn))
        rmse_real = np.sqrt(np.mean(err_real))
        rmse_music = np.sqrt(np.mean(err_music))
        
        results["CVNN"]["rmse_r"].append(rmse_cvnn)
        results["Real-CNN"]["rmse_r"].append(rmse_real)
        results["MUSIC"]["rmse_r"].append(rmse_music)
        
        results["CVNN"]["time"].append(np.mean(t_cvnn))
        results["Real-CNN"]["time"].append(np.mean(t_real))
        results["MUSIC"]["time"].append(np.mean(t_music))
        
        print(f"  CVNN     RMSE: {rmse_cvnn:.2f}m | Time: {np.mean(t_cvnn)*1000:.2f}ms")
        print(f"  Real-CNN RMSE: {rmse_real:.2f}m | Time: {np.mean(t_real)*1000:.2f}ms")
        print(f"  MUSIC    RMSE: {rmse_music:.2f}m | Time: {np.mean(t_music)*1000:.2f}ms")

    return snr_list, results

# ==========================================
# 3. 绘图
# ==========================================
def plot_results(snr_list, results):
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass
        
    plt.figure(figsize=(14, 6))
    
    # 图1: 精度对比
    plt.subplot(1, 2, 1)
    plt.plot(snr_list, results["CVNN"]["rmse_r"], 'b-o', label='Proposed CVNN', linewidth=2)
    plt.plot(snr_list, results["Real-CNN"]["rmse_r"], 'g--^', label='Real-CNN', linewidth=2)
    plt.plot(snr_list, results["MUSIC"]["rmse_r"], 'r--s', label='2D-MUSIC', linewidth=2)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('RMSE Range (m)', fontsize=12)
    plt.title('Range Estimation Accuracy vs. SNR', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # 图2: 耗时对比 (对数坐标)
    plt.subplot(1, 2, 2)
    # 转换为毫秒
    t_cvnn = [t * 1000 for t in results["CVNN"]["time"]]
    t_real = [t * 1000 for t in results["Real-CNN"]["time"]]
    t_music = [t * 1000 for t in results["MUSIC"]["time"]]
    
    plt.plot(snr_list, t_cvnn, 'b-o', label='Proposed CVNN')
    plt.plot(snr_list, t_real, 'g--^', label='Real-CNN')
    plt.plot(snr_list, t_music, 'r--s', label='2D-MUSIC')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Inference Time (ms)', fontsize=12)
    plt.title('Computation Efficiency (Log Scale)', fontsize=14)
    plt.yscale('log') # 关键：用对数坐标展示巨大的速度差异
    plt.grid(True, alpha=0.3, which="both")
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=300)
    print("\n图表已保存至 benchmark_comparison.png")
    # plt.show()

if __name__ == "__main__":
    snr_list, results = run_benchmark()
    plot_results(snr_list, results)
