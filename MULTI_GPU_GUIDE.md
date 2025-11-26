# FDA-MIMO CVNN 多GPU训练指南 (6x 2080Ti 服务器)

## 🚀 快速开始

### 1. 上传代码到服务器
```bash
# 在本地打包
tar -czf CVNN-FDA.tar.gz CVNN-FDA/

# 上传到服务器
scp CVNN-FDA.tar.gz user@server:/path/to/workspace/

# 在服务器上解压
ssh user@server
cd /path/to/workspace/
tar -xzf CVNN-FDA.tar.gz
cd CVNN-FDA
```

### 2. 检查GPU状态
```bash
nvidia-smi
# 确认6张2080Ti都在线
```

### 3. 安装依赖
```bash
pip install torch torchvision numpy matplotlib tqdm
```

### 4. 开始训练

#### 方案A: Pro模型 (推荐) - 6卡全开
```bash
chmod +x train_multi_gpu.sh
nohup bash train_multi_gpu.sh > training.log 2>&1 &
```

#### 方案B: Standard模型 (更快) - 3卡
```bash
chmod +x train_standard.sh
nohup bash train_standard.sh > training_std.log 2>&1 &
```

#### 方案C: 自定义训练
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

python main.py \
    --mode train \
    --model pro \
    --epochs 100 \
    --batch_size 192 \
    --lr 5e-5 \
    --train_size 30000 \
    --num_workers 16 \
    --multi_gpu
```

## 📊 性能预期

### Pro模型 (19M参数)
- **总Batch Size**: 192 (32 per GPU × 6)
- **训练时间**: ~2-3小时 (100 epochs)
- **目标性能**:
  - SNR=10dB: RMSE_r < 5m, RMSE_θ < 0.5°
  - SNR=0dB: RMSE_r < 10m, RMSE_θ < 1.0°

### Standard模型 (6M参数)
- **总Batch Size**: 96 (32 per GPU × 3)
- **训练时间**: ~1-2小时 (80 epochs)
- **目标性能**:
  - SNR=10dB: RMSE_r < 8m, RMSE_θ < 0.8°

## 🔧 配置说明

### 批大小计算
- **单GPU最大**: 32 (2080Ti 11GB显存)
- **6卡总批大小**: 192
- **3卡总批大小**: 96

### Worker数量
- **推荐**: 2-3 × GPU数量
- **6卡**: 16-18 workers
- **3卡**: 8-12 workers

### 学习率调整
- **多GPU加速**: batch size越大，学习率可略微提高
- **推荐**: 5e-5 (稳定) 或 1e-4 (快速)

## 📝 监控训练

### 实时查看日志
```bash
tail -f training.log
```

### 检查GPU使用率
```bash
watch -n 1 nvidia-smi
```

### 查看训练历史
```bash
python -c "import json; print(json.load(open('results/training_history.json')))"
```

## 🎯 训练完成后

### 1. 查看结果
```bash
ls checkpoints/  # 模型文件
ls results/      # 图表和报告
cat results/evaluation_results.txt
```

### 2. 下载结果到本地
```bash
# 在本地执行
scp -r user@server:/path/to/workspace/CVNN-FDA/checkpoints ./
scp -r user@server:/path/to/workspace/CVNN-FDA/results ./
```

### 3. 可视化
打开 `results/` 目录下的PNG图片：
- `training_history.png` - 训练曲线
- `rmse_vs_snr.png` - 性能曲线
- `scatter_comparison.png` - 预测对比
- `error_distribution.png` - 误差分布

## ✅ 性能优化 (CPU占用100% → 20%)

**问题**: 动态数据生成导致CPU满载，GPU等待数据
**解决**: 使用 `--use_cache` 预生成数据到内存

| 模式 | CPU占用 | GPU利用率 | 启动时间 | 内存占用 | 训练速度 |
|------|---------|-----------|----------|----------|----------|
| 动态生成 | ~100% | 60-70% | 立即 | ~2GB | 基准 |
| **缓存模式** | **20-30%** | **85-95%** | +1-2分钟 | ~6GB | **+30-50%** |

**推荐配置**:
```bash
# 缓存模式 (推荐)
python main.py --use_cache --num_workers 4

# 动态模式 (内存受限时)
python main.py --num_workers 16
```

**对比测试**:
```bash
bash benchmark_cpu.sh  # 运行CPU占用对比
```

## ✅ DataParallel 兼容性修复

本实现已修复PyTorch DataParallel对复数张量的兼容性问题：

**修复方案**:
- Dataset返回2通道实数张量 `[real, imag]` 而非原生复数
- 模型forward入口自动转换为复数张量
- 所有中间层正常使用复数运算

**测试**:
```bash
python test_multi_gpu.py  # 验证多GPU兼容性
```

## 🐛 常见问题

### Q1: 显存不足 (CUDA out of memory)
**解决**: 减小batch_size
```bash
python main.py --batch_size 96  # 从192降到96
```

### Q2: 只想用部分GPU
**解决**: 设置CUDA_VISIBLE_DEVICES
```bash
export CUDA_VISIBLE_DEVICES=0,1,2  # 只用前3张卡
```

### Q3: 训练速度慢
**检查**:
- `num_workers` 是否设置 (推荐12-16)
- 数据是否在SSD上
- GPU利用率是否接近100%

### Q4: 中断后恢复训练
代码会自动保存检查点，但目前不支持自动恢复。
可以通过修改代码添加 `--resume` 选项。

## 💡 优化建议

### 1. 数据预加载
如果磁盘I/O是瓶颈，可以将数据预生成到内存:
```python
# 修改dataset.py，添加数据缓存
```

### 2. 混合精度训练
可以进一步加速（但PyTorch复数运算对AMP支持有限）

### 3. 学习率调度
当前使用ReduceLROnPlateau，可以尝试:
- CosineAnnealingLR
- OneCycleLR

## 📞 联系

如有问题，检查:
1. `training.log` - 训练日志
2. `results/evaluation_results.txt` - 评估报告
3. GPU使用率: `nvidia-smi`
