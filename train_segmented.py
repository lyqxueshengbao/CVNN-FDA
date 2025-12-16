"""
分段SNR训练方案
- 低SNR模型: [-20, 0] dB
- 高SNR模型: [0, 30] dB
- 推理时根据估计的SNR自动选择模型

与原 train.py 保持一致:
- RangeAngleLoss (L1, range_weight=2.0)
- Adam 优化器
- CosineAnnealingLR 调度器
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import argparse

import config as cfg
from model import FDA_CVNN, FDA_CVNN_Attention
from dataset import FastDataLoader
from utils_physics import generate_covariance_matrix, denormalize_labels


class RangeAngleLoss(nn.Module):
    """
    距离-角度联合损失函数 (与 train.py 完全一致)
    使用 L1Loss 对小误差更敏感
    """
    def __init__(self, lambda_angle=1.0, range_weight=2.0):
        super().__init__()
        self.lambda_angle = lambda_angle
        self.range_weight = range_weight
        self.criterion = nn.L1Loss()
    
    def forward(self, pred, target):
        r_pred, theta_pred = pred[:, 0], pred[:, 1]
        r_target, theta_target = target[:, 0], target[:, 1]
        
        loss_r = self.criterion(r_pred, r_target)
        loss_theta = self.criterion(theta_pred, theta_target)
        
        return self.range_weight * loss_r + self.lambda_angle * loss_theta


def generate_batch_snr_range(batch_size, device, snr_min, snr_max):
    """在指定SNR范围内生成训练数据"""
    c = cfg.c
    delta_f = cfg.delta_f
    d = cfg.d
    wavelength = cfg.wavelength
    M = cfg.M
    N = cfg.N
    L = cfg.L_snapshots
    
    # 随机生成参数
    r = torch.rand(batch_size, 1, device=device) * (cfg.r_max - cfg.r_min) + cfg.r_min
    theta = torch.rand(batch_size, 1, device=device) * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
    snr_db = torch.rand(batch_size, 1, device=device) * (snr_max - snr_min) + snr_min
    
    theta_rad = torch.deg2rad(theta)
    
    # 构建导向矢量
    m = torch.arange(M, device=device).float().unsqueeze(0)
    n = torch.arange(N, device=device).float().unsqueeze(0)
    
    phi_range = -4 * torch.pi * delta_f * m * r / c
    phi_angle_tx = 2 * torch.pi * d * m * torch.sin(theta_rad) / wavelength
    a_tx = torch.exp(1j * (phi_range + phi_angle_tx))
    
    phi_angle_rx = 2 * torch.pi * d * n * torch.sin(theta_rad) / wavelength
    a_rx = torch.exp(1j * phi_angle_rx)
    
    u = (a_tx.unsqueeze(2) * a_rx.unsqueeze(1)).view(batch_size, -1)
    u = u.unsqueeze(2)
    
    # 生成信号
    s_real = torch.randn(batch_size, 1, L, device=device)
    s_imag = torch.randn(batch_size, 1, L, device=device)
    s = (s_real + 1j * s_imag) / np.sqrt(2)
    
    X_clean = torch.matmul(u, s)
    
    # 生成噪声
    n_real = torch.randn(batch_size, M*N, L, device=device)
    n_imag = torch.randn(batch_size, M*N, L, device=device)
    noise = (n_real + 1j * n_imag) / np.sqrt(2)
    
    # 计算功率并混合
    power_sig = torch.mean(torch.abs(X_clean)**2, dim=(1,2), keepdim=True)
    power_noise = power_sig / (10**(snr_db.unsqueeze(2)/10.0))
    
    X = X_clean + torch.sqrt(power_noise) * noise
    
    # 计算协方差矩阵
    R = torch.matmul(X, X.transpose(1, 2).conj()) / L
    
    # 归一化
    max_val = torch.amax(torch.abs(R), dim=(1,2), keepdim=True)
    R = R / (max_val + 1e-10)
    
    # 转换格式
    R_tensor = torch.stack([R.real, R.imag], dim=1).float()
    
    # 标签归一化
    r_norm = r / cfg.r_max
    theta_norm = (theta - cfg.theta_min) / (cfg.theta_max - cfg.theta_min)
    labels = torch.cat([r_norm, theta_norm], dim=1).float()
    
    return R_tensor, labels


def train_snr_specific(model, device, snr_min, snr_max, epochs=100, 
                       batch_size=64, model_name="model", train_samples=50000, lr=5e-4):
    """
    训练特定SNR范围的模型 (与 train.py 保持一致)
    
    参数与 main.py 训练命令对应:
    --epochs 300 -> epochs=300
    --samples 50000 -> train_samples=50000
    --batch 64 -> batch_size=64
    --lr 5e-4 -> lr (默认)
    """
    # 与 train.py 一致: 使用 Adam (不是 AdamW)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 与 train.py 一致: CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    # 与 train.py 一致: RangeAngleLoss
    criterion = RangeAngleLoss(lambda_angle=1.0, range_weight=2.0)
    
    best_val_loss = float('inf')
    best_rmse_r = float('inf')
    
    # 使用 FastDataLoader (与 train.py 一致，GPU直接生成)
    print("启用 GPU 加速数据生成 (FastDataLoader)...")
    train_loader = FastDataLoader(
        batch_size=batch_size,
        num_samples=train_samples,
        snr_range=(snr_min, snr_max),
        device=device
    )
    
    print("="*60)
    print(f"训练 {model_name}")
    print(f"SNR范围: [{snr_min}, {snr_max}] dB")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    print(f"训练样本数: {train_samples}")
    print(f"学习率: {lr}")
    print(f"模型参数量: {model.count_parameters():,}")
    print("="*60)
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        all_preds_train = []
        all_labels_train = []
        
        # 使用 tqdm 显示进度
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', leave=False)
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            all_preds_train.append(preds.detach().cpu())
            all_labels_train.append(batch_y.detach().cpu())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss /= len(train_loader)
        
        # 计算训练集RMSE
        all_preds_train = torch.cat(all_preds_train, dim=0).numpy()
        all_labels_train = torch.cat(all_labels_train, dim=0).numpy()
        preds_phys_train = denormalize_labels(all_preds_train)
        labels_phys_train = denormalize_labels(all_labels_train)
        train_rmse_r = np.sqrt(np.mean((preds_phys_train[:, 0] - labels_phys_train[:, 0])**2))
        train_rmse_theta = np.sqrt(np.mean((preds_phys_train[:, 1] - labels_phys_train[:, 1])**2))
        
        # 验证 (使用在线生成的验证数据)
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        val_batches = 30
        
        with torch.no_grad():
            for _ in range(val_batches):
                R, labels = generate_batch_snr_range(batch_size, device, snr_min, snr_max)
                preds = model(R)
                val_loss += criterion(preds, labels).item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        val_loss /= val_batches
        
        # 计算验证集RMSE
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        preds_phys = denormalize_labels(all_preds)
        labels_phys = denormalize_labels(all_labels)
        rmse_r = np.sqrt(np.mean((preds_phys[:, 0] - labels_phys[:, 0])**2))
        rmse_theta = np.sqrt(np.mean((preds_phys[:, 1] - labels_phys[:, 1])**2))
        
        scheduler.step()
        
        # 保存最佳 (基于验证集 RMSE_r)
        if rmse_r < best_rmse_r:
            best_rmse_r = rmse_r
            best_val_loss = val_loss
            save_path = f'checkpoints/{model_name}_L{cfg.L_snapshots}_best.pth'
            torch.save({
                'epoch': epoch,
                'snr_range': (snr_min, snr_max),
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'rmse_r': float(rmse_r),
                'rmse_theta': float(rmse_theta)
            }, save_path)
        
        # 打印进度 (每10个epoch或第1个epoch)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} "
                  f"Train: loss={epoch_loss:.4f} RMSE_r={train_rmse_r:.1f}m | "
                  f"Val: loss={val_loss:.4f} RMSE_r={rmse_r:.1f}m RMSE_θ={rmse_theta:.2f}°"
                  + (" ⭐" if rmse_r == best_rmse_r else ""))
    
    print(f"\n训练完成! 最佳 RMSE_r: {best_rmse_r:.2f}m")
    print(f"模型保存至: checkpoints/{model_name}_L{cfg.L_snapshots}_best.pth")
    return best_rmse_r


def evaluate_segmented(models, snr_boundaries, device, snr_list, num_samples=100):
    """
    分段评估
    models: dict, 键为SNR范围名称，值为模型
    snr_boundaries: list, SNR边界点，如[-10, 0] 表示<=-10用低SNR模型，>=-10用高SNR模型
    """
    print("\n" + "="*60)
    print("分段模型评估")
    print(f"SNR边界: {snr_boundaries}")
    print("="*60)
    
    results = {}
    
    for snr in snr_list:
        # 根据SNR选择模型
        if snr <= snr_boundaries[0]:
            model = models['low']
            model_used = "低SNR模型"
        else:
            model = models['high']
            model_used = "高SNR模型"
        
        model.eval()
        errors_r, errors_theta = [], []
        
        with torch.no_grad():
            for _ in range(num_samples):
                r_true = np.random.uniform(cfg.r_min, cfg.r_max)
                theta_true = np.random.uniform(cfg.theta_min, cfg.theta_max)
                
                R = generate_covariance_matrix(r_true, theta_true, snr)
                R_tensor = torch.from_numpy(R).unsqueeze(0).to(device)
                
                pred = model(R_tensor).cpu().numpy()[0]
                pred_phys = denormalize_labels(pred.reshape(1, -1))[0]
                
                errors_r.append((pred_phys[0] - r_true)**2)
                errors_theta.append((pred_phys[1] - theta_true)**2)
        
        rmse_r = np.sqrt(np.mean(errors_r))
        rmse_theta = np.sqrt(np.mean(errors_theta))
        results[snr] = {'rmse_r': rmse_r, 'rmse_theta': rmse_theta, 'model': model_used}
        print(f"SNR={snr:+3d}dB [{model_used}]: RMSE_r={rmse_r:.1f}m, RMSE_θ={rmse_theta:.2f}°")
    
    return results


class SegmentedCVNN:
    """分段CVNN封装类，自动根据SNR选择模型"""
    
    def __init__(self, device, snr_boundary=-5, use_attention=True):
        self.device = device
        self.snr_boundary = snr_boundary
        ModelClass = FDA_CVNN_Attention if use_attention else FDA_CVNN
        self.model_low = ModelClass().to(device)
        self.model_high = ModelClass().to(device)
        
    def load(self, low_path, high_path):
        """加载两个模型"""
        ckpt_low = torch.load(low_path, map_location=self.device, weights_only=False)
        self.model_low.load_state_dict(ckpt_low['model_state_dict'])
        self.model_low.eval()
        
        ckpt_high = torch.load(high_path, map_location=self.device, weights_only=False)
        self.model_high.load_state_dict(ckpt_high['model_state_dict'])
        self.model_high.eval()
        
        print(f"低SNR模型: {low_path}")
        print(f"高SNR模型: {high_path}")
        print(f"切换边界: {self.snr_boundary} dB")
    
    def predict(self, R_tensor, snr_db):
        """根据SNR选择模型进行预测"""
        if snr_db <= self.snr_boundary:
            return self.model_low(R_tensor)
        else:
            return self.model_high(R_tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分段SNR训练 - 与原 main.py 参数保持一致')
    # 与原 main.py 一致的参数
    parser.add_argument('--snapshots', type=int, default=30, help='快拍数 L (原 main.py: --snapshots)')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数 (原 main.py: --epochs)')
    parser.add_argument('--samples', type=int, default=50000, help='训练样本数 (原 main.py: --samples)')
    parser.add_argument('--batch', type=int, default=64, help='批次大小 (原 main.py: --batch)')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    # 分段训练特有参数
    parser.add_argument('--train-low', action='store_true', help='训练低SNR模型 [-20, 0]dB')
    parser.add_argument('--train-high', action='store_true', help='训练高SNR模型 [-5, 30]dB')
    parser.add_argument('--train-all', action='store_true', help='训练所有模型')
    parser.add_argument('--eval', action='store_true', help='评估分段模型')
    # 模型选择 (与 main.py 的 --model 参数一致)
    parser.add_argument('--model', type=str, default='dual',
                        choices=['standard', 'dual', 'se', 'cbam', 'far'],
                        help='模型类型 (原 main.py: --model)')
    parser.add_argument('--se_reduction', type=int, default=4, help='注意力压缩比')
    parser.add_argument('--deep_only', action='store_true', help='只在深层使用注意力')
    args = parser.parse_args()
    
    cfg.L_snapshots = args.snapshots
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("FDA-CVNN 分段SNR训练")
    print("="*60)
    print(f"设备: {device}")
    print(f"快拍数: L={args.snapshots}")
    
    # 选择模型类型 (与 train.py 一致的逻辑)
    if args.model == 'standard':
        ModelClass = FDA_CVNN
        model_suffix = ""
        print("模型: FDA_CVNN (基础)")
    elif args.model == 'dual':
        ModelClass = lambda: FDA_CVNN_Attention(attention_type='dual', se_reduction=args.se_reduction, deep_only=args.deep_only)
        model_suffix = "_dual"
        print(f"模型: FDA_CVNN_Attention (PP-DSA 双尺度注意力: SE+FAR) ⭐")
    elif args.model == 'se':
        ModelClass = lambda: FDA_CVNN_Attention(attention_type='se', se_reduction=args.se_reduction, deep_only=args.deep_only)
        model_suffix = "_se"
        print(f"模型: FDA_CVNN_Attention (SE 通道注意力)")
    elif args.model == 'cbam':
        ModelClass = lambda: FDA_CVNN_Attention(attention_type='cbam', se_reduction=args.se_reduction, deep_only=args.deep_only)
        model_suffix = "_cbam"
        print(f"模型: FDA_CVNN_Attention (CBAM)")
    elif args.model == 'far':
        ModelClass = lambda: FDA_CVNN_Attention(attention_type='far', se_reduction=args.se_reduction, deep_only=args.deep_only)
        model_suffix = "_far"
        print(f"模型: FDA_CVNN_Attention (FAR 局部注意力)")
    else:
        ModelClass = FDA_CVNN
        model_suffix = ""
        print("模型: FDA_CVNN (基础)")
    
    # SNR范围设置
    LOW_SNR_RANGE = (-20, 0)    # 低SNR模型范围
    HIGH_SNR_RANGE = (-5, 30)   # 高SNR模型范围 (有一点重叠)
    SNR_BOUNDARY = -5           # 切换边界
    
    if args.train_low or args.train_all:
        print("\n" + "="*60)
        print("训练低SNR专用模型")
        print("="*60)
        model_low = ModelClass().to(device)
        train_snr_specific(model_low, device, 
                          snr_min=LOW_SNR_RANGE[0], 
                          snr_max=LOW_SNR_RANGE[1],
                          epochs=args.epochs,
                          batch_size=args.batch,
                          train_samples=args.samples,
                          lr=args.lr,
                          model_name=f"cvnn_low_snr{model_suffix}")
    
    if args.train_high or args.train_all:
        print("\n" + "="*60)
        print("训练高SNR专用模型")
        print("="*60)
        model_high = ModelClass().to(device)
        train_snr_specific(model_high, device,
                          snr_min=HIGH_SNR_RANGE[0],
                          snr_max=HIGH_SNR_RANGE[1],
                          epochs=args.epochs,
                          batch_size=args.batch,
                          train_samples=args.samples,
                          lr=args.lr,
                          model_name=f"cvnn_high_snr{model_suffix}")
    
    if args.eval:
        print("\n" + "="*60)
        print("评估分段模型")
        print("="*60)
        
        # 加载模型
        low_path = f'checkpoints/cvnn_low_snr{model_suffix}_L{args.snapshots}_best.pth'
        high_path = f'checkpoints/cvnn_high_snr{model_suffix}_L{args.snapshots}_best.pth'
        
        models = {}
        
        if os.path.exists(low_path):
            model_low = ModelClass().to(device)
            ckpt = torch.load(low_path, map_location=device, weights_only=False)
            model_low.load_state_dict(ckpt['model_state_dict'])
            models['low'] = model_low
            print(f"加载低SNR模型: {low_path}")
        else:
            print(f"警告: {low_path} 不存在!")
        
        if os.path.exists(high_path):
            model_high = ModelClass().to(device)
            ckpt = torch.load(high_path, map_location=device, weights_only=False)
            model_high.load_state_dict(ckpt['model_state_dict'])
            models['high'] = model_high
            print(f"加载高SNR模型: {high_path}")
        else:
            print(f"警告: {high_path} 不存在!")
        
        if len(models) == 2:
            snr_list = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
            results = evaluate_segmented(models, [SNR_BOUNDARY], device, snr_list)
            
            # 保存结果
            results_save = {str(k): v for k, v in results.items()}
            with open(f'results/segmented{model_suffix}_eval_L{args.snapshots}.json', 'w') as f:
                json.dump(results_save, f, indent=2)
