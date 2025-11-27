#!/bin/bash
# FDA-MIMO CVNN 6卡训练脚本 (预缓存版)
# 
# 使用方法:
#   chmod +x run_6gpu.sh
#   ./run_6gpu.sh

# 指定使用的GPU (0-5)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# 设置环境变量
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 训练参数
EPOCHS=100
TRAIN_SIZE=50000      # 预生成到内存，注意内存占用
VAL_SIZE=10000
TEST_SIZE=5000
BATCH_SIZE=64         # 每个GPU的batch_size
ACCUM_STEPS=2         # 梯度累积，实际batch = 64 × 6 × 2 = 768
LR=5e-4
NUM_WORKERS=4         # 数据已在内存，不需要太多worker

echo "======================================"
echo "FDA-MIMO CVNN 6-GPU 训练 (预缓存版)"
echo "======================================"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Epochs: $EPOCHS"
echo "Train Size: $TRAIN_SIZE"
echo "Batch Size: $BATCH_SIZE × 6 GPU × $ACCUM_STEPS accum = $((BATCH_SIZE * 6 * ACCUM_STEPS))"
echo "======================================"

python train_multi_gpu.py \
    --model cvnn \
    --epochs $EPOCHS \
    --train_size $TRAIN_SIZE \
    --val_size $VAL_SIZE \
    --test_size $TEST_SIZE \
    --batch_size $BATCH_SIZE \
    --accumulation_steps $ACCUM_STEPS \
    --lr $LR \
    --num_workers $NUM_WORKERS \
    --save_dir ./checkpoints \
    --log_interval 1 \
    --save_interval 20

echo "训练完成!"
