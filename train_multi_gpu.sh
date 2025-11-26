#!/bin/bash
# FDA-MIMO CVNN 多GPU训练脚本 (6x 2080Ti)
# 使用方法: bash train_multi_gpu.sh

# 设置CUDA设备可见性 (使用所有6张卡)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# 设置日志文件
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"

echo "日志将保存到: $LOG_FILE"
echo "可以使用 'tail -f $LOG_FILE' 实时查看训练进度"

# 训练Pro模型 (推荐配置)
echo "=========================================="
echo "开始训练 CVNN Pro 模型 (6x 2080Ti)"
echo "=========================================="

nohup python main.py \
    --mode train \
    --model pro \
    --epochs 100 \
    --batch_size 192 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --dropout 0.2 \
    --lambda_angle 1.0 \
    --patience 20 \
    --train_size 30000 \
    --val_size 5000 \
    --test_size 3000 \
    --num_workers 4 \
    --use_cache \
    --multi_gpu > $LOG_FILE 2>&1 &

# 获取训练进程PID
TRAIN_PID=$!
echo "训练进程 PID: $TRAIN_PID"
echo $TRAIN_PID > train.pid

# 等待训练完成
wait $TRAIN_PID

# 训练完成后自动评估
echo ""
echo "=========================================="
echo "开始评估模型"
echo "=========================================="

EVAL_LOG_FILE="evaluation_$(date +%Y%m%d_%H%M%S).log"

nohup python main.py \
    --mode evaluate \
    --model pro \
    --model_path ./checkpoints/best_model.pth \
    --eval_snr 10 \
    --test_size 3000 \
    --batch_size 192 \
    --num_workers 8 > $EVAL_LOG_FILE 2>&1 &

EVAL_PID=$!
echo "评估进程 PID: $EVAL_PID"
wait $EVAL_PID

echo ""
echo "=========================================="
echo "训练和评估完成!"
echo "日志文件: $LOG_FILE 和 $EVAL_LOG_FILE"
echo "=========================================="
