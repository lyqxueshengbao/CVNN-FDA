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
    
    # 结果存储 (同时记录距离和角度)
    results = {
        "CVNN": {"rmse_r": [], "rmse_theta": [], "time": []},
        "MUSIC": {"rmse_r": [], "rmse_theta": [], "time": []},
        "Real-CNN": {"rmse_r": [], "rmse_theta": [], "time": []}
    }
    
    # MUSIC 搜索网格 (加密以提高精度)
    # 注意：网格越细，MUSIC 越慢，但精度越高
    r_grid = np.linspace(0, 2000, 200)    # 10m 步长 (原来40m)
    theta_grid = np.linspace(-60, 60, 60) # 2度 步长 (原来4度)
    
    print(f"\n开始对比实验 (样本数={num_samples})...")
    print(f"MUSIC 网格大小: {len(r_grid)}x{len(theta_grid)} = {len(r_grid)*len(theta_grid)} 点")
    
    for snr in snr_list:
        print(f"\n正在测试 SNR = {snr} dB ...")
        
        # 距离误差
        err_r_cvnn = []
        err_r_music = []
        err_r_real = []
        
        # 角度误差
        err_theta_cvnn = []
        err_theta_music = []
        err_theta_real = []
        
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
            theta_pred_cvnn = pred[1] * 120 - 60  # 反归一化: [0,1] -> [-60, 60]
            t1 = time.time()
            
            err_r_cvnn.append((r_pred_cvnn - r_true)**2)
            err_theta_cvnn.append((theta_pred_cvnn - theta_true)**2)
            t_cvnn.append(t1 - t0)
            
            # --- 2. 测试 Real-CNN ---
            t0 = time.time()
            with torch.no_grad():
                pred_real = real_cnn(R_tensor).cpu().numpy()[0]
            r_pred_real = pred_real[0] * 2000
            theta_pred_real = pred_real[1] * 120 - 60
            t1 = time.time()
            
            err_r_real.append((r_pred_real - r_true)**2)
            err_theta_real.append((theta_pred_real - theta_true)**2)
            t_real.append(t1 - t0)
            
            # --- 3. 测试 MUSIC ---
            # 恢复复数矩阵
            R_complex = R[0] + 1j * R[1]
            
            t0 = time.time()
            r_pred_music, theta_pred_music = music_2d(R_complex, r_grid, theta_grid)
            t1 = time.time()
            
            err_r_music.append((r_pred_music - r_true)**2)
            err_theta_music.append((theta_pred_music - theta_true)**2)
            t_music.append(t1 - t0)
            
        # 计算 RMSE
        rmse_r_cvnn = np.sqrt(np.mean(err_r_cvnn))
        rmse_r_real = np.sqrt(np.mean(err_r_real))
        rmse_r_music = np.sqrt(np.mean(err_r_music))
        
        rmse_theta_cvnn = np.sqrt(np.mean(err_theta_cvnn))
        rmse_theta_real = np.sqrt(np.mean(err_theta_real))
        rmse_theta_music = np.sqrt(np.mean(err_theta_music))
        
        results["CVNN"]["rmse_r"].append(rmse_r_cvnn)
        results["Real-CNN"]["rmse_r"].append(rmse_r_real)
        results["MUSIC"]["rmse_r"].append(rmse_r_music)
        
        results["CVNN"]["rmse_theta"].append(rmse_theta_cvnn)
        results["Real-CNN"]["rmse_theta"].append(rmse_theta_real)
        results["MUSIC"]["rmse_theta"].append(rmse_theta_music)
        
        results["CVNN"]["time"].append(np.mean(t_cvnn))
        results["Real-CNN"]["time"].append(np.mean(t_real))
        results["MUSIC"]["time"].append(np.mean(t_music))
        
        print(f"  CVNN     RMSE_r: {rmse_r_cvnn:6.2f}m | RMSE_θ: {rmse_theta_cvnn:5.2f}° | Time: {np.mean(t_cvnn)*1000:.2f}ms")
        print(f"  Real-CNN RMSE_r: {rmse_r_real:6.2f}m | RMSE_θ: {rmse_theta_real:5.2f}° | Time: {np.mean(t_real)*1000:.2f}ms")
        print(f"  MUSIC    RMSE_r: {rmse_r_music:6.2f}m | RMSE_θ: {rmse_theta_music:5.2f}° | Time: {np.mean(t_music)*1000:.2f}ms")

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
        
    plt.figure(figsize=(16, 10))
    
    # 图1: 距离精度对比
    plt.subplot(2, 2, 1)
    plt.plot(snr_list, results["CVNN"]["rmse_r"], 'b-o', label='Proposed CVNN', linewidth=2)
    plt.plot(snr_list, results["Real-CNN"]["rmse_r"], 'g--^', label='Real-CNN', linewidth=2)
    plt.plot(snr_list, results["MUSIC"]["rmse_r"], 'r--s', label='2D-MUSIC', linewidth=2)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('RMSE Range (m)', fontsize=12)
    plt.title('Range Estimation Accuracy vs. SNR', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # 图2: 角度精度对比
    plt.subplot(2, 2, 2)
    plt.plot(snr_list, results["CVNN"]["rmse_theta"], 'b-o', label='Proposed CVNN', linewidth=2)
    plt.plot(snr_list, results["Real-CNN"]["rmse_theta"], 'g--^', label='Real-CNN', linewidth=2)
    plt.plot(snr_list, results["MUSIC"]["rmse_theta"], 'r--s', label='2D-MUSIC', linewidth=2)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('RMSE Angle (°)', fontsize=12)
    plt.title('Angle Estimation Accuracy vs. SNR', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # 图3: 耗时对比 (对数坐标)
    plt.subplot(2, 2, 3)
    t_cvnn = [t * 1000 for t in results["CVNN"]["time"]]
    t_real = [t * 1000 for t in results["Real-CNN"]["time"]]
    t_music = [t * 1000 for t in results["MUSIC"]["time"]]
    
    plt.plot(snr_list, t_cvnn, 'b-o', label='Proposed CVNN', linewidth=2)
    plt.plot(snr_list, t_real, 'g--^', label='Real-CNN', linewidth=2)
    plt.plot(snr_list, t_music, 'r--s', label='2D-MUSIC', linewidth=2)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Inference Time (ms)', fontsize=12)
    plt.title('Computation Efficiency (Log Scale)', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which="both")
    plt.legend(fontsize=10)
    
    # 图4: 综合表格
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # 计算平均值
    avg_r_cvnn = np.mean(results["CVNN"]["rmse_r"])
    avg_r_real = np.mean(results["Real-CNN"]["rmse_r"])
    avg_r_music = np.mean(results["MUSIC"]["rmse_r"])
    
    avg_theta_cvnn = np.mean(results["CVNN"]["rmse_theta"])
    avg_theta_real = np.mean(results["Real-CNN"]["rmse_theta"])
    avg_theta_music = np.mean(results["MUSIC"]["rmse_theta"])
    
    avg_t_cvnn = np.mean(results["CVNN"]["time"]) * 1000
    avg_t_real = np.mean(results["Real-CNN"]["time"]) * 1000
    avg_t_music = np.mean(results["MUSIC"]["time"]) * 1000
    
    table_data = [
        ['Method', 'Avg RMSE_r (m)', 'Avg RMSE_θ (°)', 'Avg Time (ms)'],
        ['CVNN', f'{avg_r_cvnn:.2f}', f'{avg_theta_cvnn:.2f}', f'{avg_t_cvnn:.2f}'],
        ['Real-CNN', f'{avg_r_real:.2f}', f'{avg_theta_real:.2f}', f'{avg_t_real:.2f}'],
        ['2D-MUSIC', f'{avg_r_music:.2f}', f'{avg_theta_music:.2f}', f'{avg_t_music:.2f}'],
    ]
    
    table = plt.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Average Performance Summary', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存至 benchmark_comparison.png")

if __name__ == "__main__":
    snr_list, results = run_benchmark()
    plot_results(snr_list, results)
