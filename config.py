# -*- coding: utf-8 -*-
"""
FDA-MIMO 雷达参数估计 - 配置文件
Configuration for FDA-MIMO Radar Range-Angle Estimation using CVNN
"""

import torch

# =============================================================================
# 物理参数 (Physical Parameters)
# =============================================================================
M = 10                          # 发射阵元数 (Number of transmit antennas)
N = 10                          # 接收阵元数 (Number of receive antennas)
MN = M * N                      # 虚拟阵元数 (Virtual array size)

c = 3e8                         # 光速 (Speed of light) [m/s]
f0 = 1e9                        # 载频 (Carrier frequency) [Hz]
delta_f = 30e3                  # 频率偏移量 (Frequency increment) [Hz]
wavelength = c / f0             # 波长 (Wavelength) [m]
d = c / (2 * f0)                # 阵元间距 (Element spacing) [m], 即 0.15m

# =============================================================================
# 数据生成参数 (Data Generation Parameters)
# =============================================================================
L_snapshots = 500               # 快拍数 (Number of snapshots) - 增加以提高协方差矩阵质量
K_targets_max = 2               # 最大目标数 (Maximum number of targets)

# 目标参数范围 (Target parameter ranges)
r_min = 0                       # 最小距离 [m]
r_max = 2000                    # 最大距离 [m]
theta_min = -60                 # 最小角度 [degrees]
theta_max = 60                  # 最大角度 [degrees]

# =============================================================================
# 训练参数 (Training Parameters)
# =============================================================================
BATCH_SIZE = 32                 # 批大小 (减小以适应更大模型)
NUM_EPOCHS = 100                # 训练轮数
LEARNING_RATE = 5e-5            # 学习率 (降低以获得更稳定的训练)
WEIGHT_DECAY = 1e-4             # 权重衰减 (增大以防止过拟合)

# 数据集大小
TRAIN_SIZE = 30000              # 训练集样本数 (增大3倍)
VAL_SIZE = 5000                 # 验证集样本数
TEST_SIZE = 3000                # 测试集样本数

# SNR 范围 (用于训练和测试)
SNR_TRAIN_MIN = -10             # 训练时最小SNR [dB]
SNR_TRAIN_MAX = 20              # 训练时最大SNR [dB]
SNR_TEST_RANGE = range(-10, 25, 5)  # 测试SNR范围 [dB]

# =============================================================================
# 设备配置 (Device Configuration)
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 模型保存路径 (Model Save Path)
# =============================================================================
MODEL_SAVE_PATH = './checkpoints/'
RESULTS_PATH = './results/'
