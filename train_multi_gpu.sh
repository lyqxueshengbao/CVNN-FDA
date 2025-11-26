#!/bin/bash
# FDA-MIMO CVNN 多GPU训练脚本 (6x 2080Ti)
# 使用方法: bash train_multi_gpu.sh

# 设置CUDA设备可见性 (使用所有6张卡)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# 训练Pro模型 (推荐配置)
echo "=========================================="
echo "开始训练 CVNN Pro 模型 (6x 2080Ti)"
echo "=========================================="

python main.py \
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
    --num_workers 16 \
    --multi_gpu

# 训练完成后自动评估
echo ""
echo "=========================================="
echo "开始评估模型"
echo "=========================================="

python main.py \
    --mode evaluate \
    --model pro \
    --model_path ./checkpoints/best_model.pth \
    --eval_snr 10 \
    --test_size 3000 \
    --batch_size 192 \
    --num_workers 8

echo ""
echo "=========================================="
echo "训练和评估完成!"
echo "=========================================="
