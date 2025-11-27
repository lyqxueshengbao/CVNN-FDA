@echo off
REM FDA-MIMO CVNN 6卡训练脚本 (Windows)
REM 
REM 使用方法:
REM   run_6gpu.bat

REM 指定使用的GPU (0-5)
set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

REM 训练参数
set EPOCHS=100
set TRAIN_SIZE=100000
set VAL_SIZE=20000
set TEST_SIZE=10000
set BATCH_SIZE=32
set ACCUM_STEPS=4
set LR=1e-3
set NUM_WORKERS=8

echo ======================================
echo FDA-MIMO CVNN 6-GPU Training
echo ======================================
echo GPUs: %CUDA_VISIBLE_DEVICES%
echo Epochs: %EPOCHS%
echo Train Size: %TRAIN_SIZE%
echo Batch: %BATCH_SIZE% x 6 GPU x %ACCUM_STEPS% accum
echo ======================================

python train_multi_gpu.py ^
    --model cvnn ^
    --epochs %EPOCHS% ^
    --train_size %TRAIN_SIZE% ^
    --val_size %VAL_SIZE% ^
    --test_size %TEST_SIZE% ^
    --batch_size %BATCH_SIZE% ^
    --accumulation_steps %ACCUM_STEPS% ^
    --lr %LR% ^
    --num_workers %NUM_WORKERS% ^
    --save_dir ./checkpoints ^
    --log_interval 1 ^
    --save_interval 20

echo Training completed!
pause
