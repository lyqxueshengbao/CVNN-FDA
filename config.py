"""
FDA-MIMO CVNN 配置文件
物理参数与训练配置（已修复模糊问题）
"""
import torch

# ================= 物理参数 (核心修复) =================
c = 3e8              # 光速
f0 = 1e9             # 1 GHz 载频
M = 10               # 发射阵元
N = 10               # 接收阵元
MN = M * N           # 总阵元数

# --- 关键修复：解决模糊问题 ---
# 设定最大探测距离 r_max = 2000m
# 最大不模糊距离 R_max = c / (2 * delta_f) >= r_max
# 因此 delta_f <= c / (2 * r_max) = 75,000 Hz
r_max = 2000.0
r_min = 0.0
delta_f = 70e3       # 70 kHz (保证 2000m 内无模糊, R_max = 2143m)

# 计算并验证
R_max = c / (2 * delta_f)  # 最大不模糊距离
assert R_max >= r_max, f"物理模糊！R_max={R_max:.0f}m < r_max={r_max:.0f}m"

d = c / (2 * f0)     # 半波长间距
wavelength = c / f0  # 波长

# 角度范围
theta_min = -60.0    # 最小角度 (度)
theta_max = 60.0     # 最大角度 (度)

# ================= 数据生成 =================
L_snapshots = 10     # 快拍数 (降低以测试鲁棒性)
num_targets = 1      # 单目标场景

# ================= 训练配置 =================
batch_size = 64
lr = 1e-4
epochs = 100
train_samples = 10000    # 训练样本数
val_samples = 2000       # 验证样本数
test_samples = 2000      # 测试样本数

# SNR 配置
snr_train_min = -10      # 训练时最小SNR (dB)，扩展到负SNR
snr_train_max = 30       # 训练时最大SNR (dB)
snr_test = 20            # 测试时固定SNR (dB)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 随机种子
seed = 42

# 模型保存路径
checkpoint_dir = "checkpoints"
model_save_path = f"{checkpoint_dir}/fda_cvnn_best.pth"


if __name__ == "__main__":
    print("=" * 50)
    print("FDA-MIMO CVNN 配置")
    print("=" * 50)
    print(f"载频 f0 = {f0/1e9:.1f} GHz")
    print(f"频率增量 delta_f = {delta_f/1e3:.0f} kHz")
    print(f"阵元数 M×N = {M}×{N} = {MN}")
    print(f"距离范围 [{r_min}, {r_max}] m")
    print(f"角度范围 [{theta_min}, {theta_max}]°")
    print(f"最大不模糊距离 R_max = {R_max:.0f} m")
    print(f"快拍数 L = {L_snapshots}")
    print(f"设备: {device}")
    print("=" * 50)
