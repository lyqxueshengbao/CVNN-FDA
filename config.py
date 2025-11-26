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
L_snapshots = 200               # 快拍数 (Number of snapshots)
K_targets_max = 2               # 最大目标数 (Maximum number of targets)

# 目标参数范围 (Target parameter ranges)
r_min = 0                       # 最小距离 [m]
r_max = 2000                    # 最大距离 [m]
theta_min = -60                 # 最小角度 [degrees]
theta_max = 60                  # 最大角度 [degrees]

# =============================================================================
# 训练参数 (Training Parameters)
# =============================================================================
BATCH_SIZE = 64                 # 批大小
NUM_EPOCHS = 100                # 训练轮数
LEARNING_RATE = 1e-4            # 学习率
WEIGHT_DECAY = 1e-5             # 权重衰减

# 数据集大小
TRAIN_SIZE = 10000              # 训练集样本数
VAL_SIZE = 2000                 # 验证集样本数
TEST_SIZE = 2000                # 测试集样本数

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
