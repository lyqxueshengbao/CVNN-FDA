#!/bin/bash
# FDA-MIMO CVNN 6卡训练脚本
# 
# 使用方法:
#   chmod +x run_6gpu.sh
#   ./run_6gpu.sh

# 指定使用的GPU (0-5)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# 设置环境变量优化多GPU训练
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 训练参数
EPOCHS=100
TRAIN_SIZE=100000
VAL_SIZE=20000
TEST_SIZE=10000
BATCH_SIZE=128        # 每个GPU的batch_size，实际 = 128 × 6 = 768
LR=1e-3
NUM_WORKERS=16        # 数据加载线程数

echo "======================================"
echo "FDA-MIMO CVNN 6-GPU 训练"
echo "======================================"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Epochs: $EPOCHS"
echo "Train Size: $TRAIN_SIZE"
echo "Batch Size: $BATCH_SIZE × 6 = $((BATCH_SIZE * 6))"
echo "======================================"

python train_multi_gpu.py \
    --model cvnn \
    --epochs $EPOCHS \
    --train_size $TRAIN_SIZE \
    --val_size $VAL_SIZE \
    --test_size $TEST_SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --num_workers $NUM_WORKERS \
    --save_dir ./checkpoints \
    --log_interval 1 \
    --save_interval 20

echo "训练完成!"
