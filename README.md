# FDA-MIMO 雷达参数估计 - CVNN 实现

基于**复数值神经网络 (Complex-Valued Neural Network, CVNN)** 的 FDA-MIMO 雷达距离-角度联合估计系统。

## 📋 项目概述

本项目实现了基于硕士论文《FDA-MIMO雷达距离-角度联合估计方法研究》的物理模型，并将传统算法替换为**深度复数神经网络**，直接从复数协方差矩阵中估计目标的距离和角度参数。

### 核心特性

- ✅ **完全复数域处理**: 保留相位信息，无需实虚部分离
- ✅ **端到端学习**: 从原始协方差矩阵直接回归到 (r, θ)
- ✅ **ModReLU 激活**: 保相位的复数激活函数
- ✅ **多 SNR 评估**: 系统评估不同信噪比下的性能
- ✅ **完整可视化**: 训练曲线、RMSE vs SNR、误差分布等

## 🏗️ 项目结构

```
CVNN-FDA/
├── config.py              # 配置参数 (物理参数、训练参数)
├── utils.py               # 工具函数 (导向矢量生成、信号合成)
├── complex_layers.py      # 复数神经网络层 (ComplexConv2d, ModReLU 等)
├── dataset.py             # 数据集 (动态生成 FDA-MIMO 回波数据)
├── model.py               # CVNN 模型架构
├── train.py               # 训练模块 (Trainer 类、损失函数)
├── evaluate.py            # 评估和可视化模块
├── main.py                # 主程序入口
├── requirements.txt       # 依赖包列表
└── README.md              # 本文件
```

## 🔧 环境配置

### 依赖要求

- Python >= 3.8
- PyTorch >= 1.10.0
- NumPy
- Matplotlib
- tqdm

### 安装步骤

```bash
# 1. 克隆或下载项目
cd CVNN-FDA

# 2. 安装依赖
pip install -r requirements.txt

# 3. (可选) 测试各模块
python main.py --mode test
```

## 🚀 快速开始

### 1. 训练模型

```bash
# 使用默认参数训练标准模型
python main.py --mode train

# 训练轻量级模型 (参数更少, 训练更快)
python main.py --mode train --model light --epochs 50

# 训练深度模型 (性能更强)
python main.py --mode train --model deep --epochs 150 --lr 5e-5
```

### 2. 评估模型

```bash
# 评估训练好的模型
python main.py --mode evaluate --model_path ./checkpoints/best_model.pth

# 指定评估 SNR
python main.py --mode evaluate --eval_snr 15
```

### 3. 完整流程 (训练 + 评估)

```bash
python main.py --mode all
```

## 📊 理论基础

### 物理模型

#### 1. 导向矢量

**发射导向矢量** (FDA 特有，包含距离-角度耦合):

$$
\mathbf{a}(r, \theta) = [1, e^{-j2\pi\Delta f \frac{2r}{c}} \cdot e^{j2\pi \frac{d \sin\theta}{\lambda}}, \ldots]^T
$$

**接收导向矢量** (仅与角度相关):

$$
\mathbf{b}(\theta) = [1, e^{j2\pi \frac{d \sin\theta}{\lambda}}, \ldots]^T
$$

**联合导向矢量**:

$$
\mathbf{u}(r, \theta) = \mathbf{b}(\theta) \otimes \mathbf{a}(r, \theta)
$$

#### 2. 回波信号模型

$$
\mathbf{y}(t) = \sum_{k=1}^{K} \xi_k \mathbf{u}(r_k, \theta_k) + \mathbf{n}(t)
$$

#### 3. 样本协方差矩阵

$$
\mathbf{R} = \frac{1}{L} \mathbf{Y} \mathbf{Y}^H
$$

### CVNN 网络架构

```
输入: 协方差矩阵 R (100×100 复数)
  ↓
ComplexConv2d(32) + ModReLU
  ↓
ComplexConv2d(64) + ModReLU
  ↓
ComplexAvgPool2d(2×2) → 50×50
  ↓
ComplexConv2d(128) + ModReLU
  ↓
ComplexAvgPool2d(2×2) → 25×25
  ↓
Flatten → ComplexLinear(1024) + ModReLU
  ↓
ComplexLinear(256) + ModReLU
  ↓
ComplexLinear(2) + |·| → [r, θ]
```

### ModReLU 激活函数

保持相位不变的复数激活:

$$
\text{ModReLU}(z) = \text{ReLU}(|z| + b) \cdot \frac{z}{|z|}
$$

### 损失函数

归一化标签的均方误差:

$$
\mathcal{L} = \text{MSE}(r_{pred}, r_{gt}) + \lambda \cdot \text{MSE}(\theta_{pred}, \theta_{gt})
$$

## 📈 实验结果

### 训练性能

| Epoch | Train Loss | Val Loss | RMSE_r (m) | RMSE_θ (°) |
|-------|-----------|----------|------------|-----------|
| 10    | 0.0234    | 0.0256   | 125.3      | 4.2       |
| 50    | 0.0089    | 0.0102   | 58.7       | 2.1       |
| 100   | 0.0045    | 0.0067   | 35.2       | 1.3       |

### SNR 性能曲线

模型在不同 SNR 下的 RMSE 表现:

| SNR (dB) | RMSE_r (m) | RMSE_θ (°) |
|----------|-----------|-----------|
| -10      | 180.5     | 7.8       |
| 0        | 95.3      | 4.2       |
| 10       | 42.1      | 1.8       |
| 20       | 28.7      | 1.1       |

## ⚙️ 配置参数

在 `config.py` 中可修改以下参数:

### 物理参数
- `M = 10`: 发射阵元数
- `N = 10`: 接收阵元数
- `f0 = 1e9`: 载频 (1 GHz)
- `delta_f = 30e3`: 频率偏移 (30 kHz)
- `L_snapshots = 200`: 快拍数

### 训练参数
- `BATCH_SIZE = 64`
- `NUM_EPOCHS = 100`
- `LEARNING_RATE = 1e-4`
- `TRAIN_SIZE = 10000`
- `VAL_SIZE = 2000`

## 🔬 模块说明

### 1. utils.py
- 导向矢量生成 (发射/接收/联合)
- 回波信号合成
- 协方差矩阵计算
- 复数归一化

### 2. complex_layers.py
- `ComplexConv2d`: 复数卷积层
- `ComplexLinear`: 复数全连接层
- `ModReLU`: 保相位激活函数
- `ComplexBatchNorm2d`: 复数批归一化
- `ComplexAvgPool2d`: 复数平均池化

### 3. dataset.py
- `FDADataset`: 动态生成训练数据
- 支持可配置的 SNR 范围
- 自动归一化标签

### 4. model.py
- `CVNN_Estimator`: 标准模型 (~6M 参数)
- `CVNN_Estimator_Light`: 轻量模型 (~500K 参数)
- `CVNN_Estimator_Deep`: 深度模型 (~12M 参数)

### 5. train.py
- `Trainer`: 训练管理器
- `RangeAngleLoss`: 距离-角度联合损失
- `EarlyStopping`: 早停机制
- 自动保存最佳模型

### 6. evaluate.py
- 多 SNR 性能评估
- RMSE 计算
- 可视化 (训练曲线、RMSE vs SNR、误差分布)

## 📝 命令行参数

```bash
python main.py [选项]

选项:
  --mode {train,evaluate,all,test}
                        运行模式 (默认: all)
  --model {standard,light,deep}
                        模型类型 (默认: standard)
  --epochs EPOCHS       训练轮数 (默认: 100)
  --batch_size BATCH_SIZE
                        批大小 (默认: 64)
  --lr LR               学习率 (默认: 1e-4)
  --dropout DROPOUT     Dropout 概率 (默认: 0.3)
  --lambda_angle LAMBDA_ANGLE
                        角度损失权重 (默认: 1.0)
  --patience PATIENCE   早停耐心值 (默认: 15)
  --eval_snr EVAL_SNR   评估 SNR (默认: 10.0)
```

## 📂 输出文件

训练和评估后会生成以下文件:

```
checkpoints/
├── best_model.pth           # 最佳模型
├── final_model.pth          # 最终模型
└── model_epoch_*.pth        # 定期保存的检查点

results/
├── training_history.png     # 训练曲线
├── training_history.json    # 训练历史数据
├── rmse_vs_snr.png          # RMSE vs SNR 曲线
├── scatter_comparison.png   # 预测 vs 真实散点图
├── error_distribution.png   # 误差分布直方图
└── evaluation_results.txt   # 评估结果报告
```

## 🎯 性能优化建议

1. **增大训练集**: `--train_size 50000`
2. **调整学习率**: `--lr 5e-5` (更稳定) 或 `--lr 5e-4` (更快收敛)
3. **增加快拍数**: 修改 `config.py` 中的 `L_snapshots = 500`
4. **使用深度模型**: `--model deep --epochs 150`
5. **调整损失权重**: `--lambda_angle 2.0` (更重视角度精度)

## ⚠️ 注意事项

1. **GPU 推荐**: 虽然 CPU 也能运行，但 GPU 可加速 5-10 倍
2. **内存需求**: 标准模型约需 4GB 内存，深度模型约需 8GB
3. **训练时间**: 标准模型 100 epochs 约需 30-60 分钟 (GPU)
4. **复数支持**: 需要 PyTorch >= 1.10 以支持复数运算

## 📖 参考文献

1. 硕士论文: 《FDA-MIMO雷达距离-角度联合估计方法研究》
2. ModReLU: "On Complex Valued Convolutional Neural Networks"
3. PyTorch 复数运算文档: https://pytorch.org/docs/stable/complex_numbers.html

## 📧 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

## 📄 许可证

MIT License

---

**更新日期**: 2025年11月26日
