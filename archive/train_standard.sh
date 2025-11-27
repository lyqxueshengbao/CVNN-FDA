#!/bin/bash
# FDA-MIMO CVNN 快速训练脚本 (标准模型，3卡)
# 使用方法: bash train_standard.sh

export CUDA_VISIBLE_DEVICES=0,1,2

echo "=========================================="
echo "开始训练 CVNN Standard 模型 (3x 2080Ti)"
echo "=========================================="

python main.py \
    --mode train \
    --model standard \
    --epochs 80 \
    --batch_size 96 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --dropout 0.3 \
    --train_size 20000 \
    --val_size 4000 \
    --num_workers 12 \
    --multi_gpu

python main.py \
    --mode evaluate \
    --model standard \
    --model_path ./checkpoints/best_model.pth \
    --eval_snr 10 \
    --test_size 3000

echo "训练完成!"
