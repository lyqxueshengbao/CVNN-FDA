#!/bin/bash
# CPU占用对比测试
# 测试动态生成 vs 缓存模式的性能差异

echo "=========================================="
echo "CPU占用对比测试"
echo "=========================================="

# 设置小规模测试
TRAIN_SIZE=3000
VAL_SIZE=500
EPOCHS=3
BATCH_SIZE=96

echo ""
echo "测试配置:"
echo "  训练集: $TRAIN_SIZE 样本"
echo "  验证集: $VAL_SIZE 样本"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo ""

# 测试1: 动态生成模式 (原始)
echo "=========================================="
echo "测试1: 动态生成模式 (实时计算)"
echo "=========================================="
echo "预期: CPU ~100%, GPU利用率 60-70%"
echo ""

time python main.py \
    --mode train \
    --model light \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --train_size $TRAIN_SIZE \
    --val_size $VAL_SIZE \
    --test_size 500 \
    --num_workers 8 \
    --multi_gpu

echo ""
echo "动态生成模式测试完成!"
echo ""
echo "请记录CPU和GPU使用率..."
read -p "按Enter继续测试缓存模式..."

# 测试2: 缓存模式 (优化)
echo ""
echo "=========================================="
echo "测试2: 缓存模式 (预生成到内存)"
echo "=========================================="
echo "预期: CPU ~20-30%, GPU利用率 85-95%"
echo ""

time python main.py \
    --mode train \
    --model light \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --train_size $TRAIN_SIZE \
    --val_size $VAL_SIZE \
    --test_size 500 \
    --num_workers 2 \
    --use_cache \
    --multi_gpu

echo ""
echo "=========================================="
echo "对比测试完成!"
echo "=========================================="
echo ""
echo "预期改进:"
echo "  ✓ CPU占用: 100% → 20-30%"
echo "  ✓ GPU利用率: 60-70% → 85-95%"
echo "  ✓ 训练速度: 提升 30-50%"
echo "  ✓ Workers: 16 → 4 (降低进程开销)"
echo ""
echo "注意: 缓存模式会在启动时预生成数据 (~1-2分钟)"
echo "      但训练时CPU占用显著降低，GPU效率提升"
