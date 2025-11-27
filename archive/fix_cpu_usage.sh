#!/bin/bash
# 快速修复脚本: 停止当前训练，切换到缓存模式重新启动
# 使用方法: bash fix_cpu_usage.sh

echo "=========================================="
echo "CPU占用优化 - 切换到缓存模式"
echo "=========================================="

# 1. 停止当前训练
echo ""
echo "[1/3] 停止当前训练进程..."
if [ -f train.pid ]; then
    PID=$(cat train.pid)
    if ps -p $PID > /dev/null; then
        echo "  发现训练进程 PID: $PID"
        kill $PID
        echo "  ✓ 已停止"
        sleep 2
    else
        echo "  进程已结束"
    fi
else
    echo "  未找到 train.pid, 尝试直接查找..."
    pkill -f "main.py.*--mode train"
    sleep 2
fi

# 2. 备份当前检查点
echo ""
echo "[2/3] 备份当前训练进度..."
if [ -d checkpoints ]; then
    BACKUP_DIR="checkpoints_backup_$(date +%Y%m%d_%H%M%S)"
    cp -r checkpoints $BACKUP_DIR
    echo "  ✓ 备份到: $BACKUP_DIR"
fi

# 3. 使用缓存模式重新启动
echo ""
echo "[3/3] 启动缓存模式训练..."
echo "  配置: num_workers=4, use_cache=True"
echo ""

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

LOG_FILE="training_cached_$(date +%Y%m%d_%H%M%S).log"

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

NEW_PID=$!
echo $NEW_PID > train.pid

echo ""
echo "=========================================="
echo "✓ 训练已重启 (缓存模式)"
echo "=========================================="
echo ""
echo "进程 PID: $NEW_PID"
echo "日志文件: $LOG_FILE"
echo ""
echo "预期改进:"
echo "  • CPU占用: 100% → 20-30%"
echo "  • GPU利用率: 60-70% → 85-95%"
echo "  • 训练速度: 提升 30-50%"
echo ""
echo "监控命令:"
echo "  watch -n 1 nvidia-smi      # GPU状态"
echo "  tail -f $LOG_FILE          # 训练日志"
echo "  htop                       # CPU占用"
echo ""
echo "注意: 启动时会预生成30K样本到内存 (~1-2分钟)"
echo "      请等待进度条完成后再查看GPU利用率"
