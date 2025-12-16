"""
改进的 CVNN 训练脚本 - 专门针对低 SNR 优化

改进点:
1. 扩展 SNR 范围到 -20dB
2. 对低 SNR 样本过采样 (重采样策略)
3. 课程学习: 先学高SNR，再逐渐加入低SNR
4. SNR-aware 损失函数: 对低SNR样本加权
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json

import config as cfg
from model import FDA_CVNN, FDA_CVNN_Attention
from utils_physics import generate_batch_torch, denormalize_labels


class SNRAwareLoss(nn.Module):
    """
    SNR感知损失函数
    对低SNR样本给予更高权重
    """
    def __init__(self, snr_min=-20, snr_max=30, low_snr_weight=3.0):
        super().__init__()
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.low_snr_weight = low_snr_weight
        self.criterion = nn.L1Loss(reduction='none')
    
    def forward(self, pred, target, snr_db=None):
        """
        Args:
            pred: [B, 2] 预测值
            target: [B, 2] 真实值  
            snr_db: [B] 每个样本的SNR (可选)
        """
        # 基础损失
        loss = self.criterion(pred, target)  # [B, 2]
        loss = loss.mean(dim=1)  # [B]
        
        if snr_db is not None:
            # 计算权重: 低SNR -> 高权重
            # 使用 sigmoid 平滑过渡
            # SNR=-20 -> weight=3.0, SNR=10 -> weight=1.0
            normalized_snr = (snr_db - self.snr_min) / (self.snr_max - self.snr_min)
            weights = self.low_snr_weight - (self.low_snr_weight - 1.0) * normalized_snr
            weights = weights.clamp(min=1.0, max=self.low_snr_weight)
            loss = loss * weights
        
        return loss.mean()


def generate_batch_with_snr_bias(batch_size, device, snr_range, low_snr_prob=0.5):
    """
    生成带有低SNR偏向的批次数据
    
    Args:
        batch_size: 批次大小
        device: 设备
        snr_range: (min, max) SNR范围
        low_snr_prob: 低SNR样本的概率 (0.5表示一半样本来自低SNR区间)
    """
    snr_min, snr_max = snr_range
    snr_mid = (snr_min + snr_max) / 2
    
    # 物理常量
    c = cfg.c
    delta_f = cfg.delta_f
    d = cfg.d
    wavelength = cfg.wavelength
    M = cfg.M
    N = cfg.N
    L = cfg.L_snapshots
    
    # 1. 随机生成参数 [B, 1]
    r = torch.rand(batch_size, 1, device=device) * (cfg.r_max - cfg.r_min) + cfg.r_min
    theta = torch.rand(batch_size, 1, device=device) * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
    
    # 2. 偏向低SNR的采样
    # 一部分样本从低SNR区间采样，一部分从高SNR区间采样
    low_snr_mask = torch.rand(batch_size, 1, device=device) < low_snr_prob
    snr_low = torch.rand(batch_size, 1, device=device) * (snr_mid - snr_min) + snr_min
    snr_high = torch.rand(batch_size, 1, device=device) * (snr_max - snr_mid) + snr_mid
    snr_db = torch.where(low_snr_mask, snr_low, snr_high)
    
    theta_rad = torch.deg2rad(theta)
    
    # 3. 构建导向矢量
    m = torch.arange(M, device=device).float().unsqueeze(0)
    n = torch.arange(N, device=device).float().unsqueeze(0)
    
    phi_range = -4 * torch.pi * delta_f * m * r / c
    phi_angle_tx = 2 * torch.pi * d * m * torch.sin(theta_rad) / wavelength
    a_tx = torch.exp(1j * (phi_range + phi_angle_tx))
    
    phi_angle_rx = 2 * torch.pi * d * n * torch.sin(theta_rad) / wavelength
    a_rx = torch.exp(1j * phi_angle_rx)
    
    u = (a_tx.unsqueeze(2) * a_rx.unsqueeze(1)).view(batch_size, -1)
    u = u.unsqueeze(2)
    
    # 4. 生成信号
    s_real = torch.randn(batch_size, 1, L, device=device)
    s_imag = torch.randn(batch_size, 1, L, device=device)
    s = (s_real + 1j * s_imag) / np.sqrt(2)
    
    X_clean = torch.matmul(u, s)
    
    # 5. 生成噪声
    n_real = torch.randn(batch_size, M*N, L, device=device)
    n_imag = torch.randn(batch_size, M*N, L, device=device)
    noise = (n_real + 1j * n_imag) / np.sqrt(2)
    
    # 6. 计算功率并混合
    power_sig = torch.mean(torch.abs(X_clean)**2, dim=(1,2), keepdim=True)
    power_noise = power_sig / (10**(snr_db.unsqueeze(2)/10.0))
    
    X = X_clean + torch.sqrt(power_noise) * noise
    
    # 7. 计算协方差矩阵
    R = torch.matmul(X, X.transpose(1, 2).conj()) / L
    
    # 8. 归一化
    max_val = torch.amax(torch.abs(R), dim=(1,2), keepdim=True)
    R = R / (max_val + 1e-10)
    
    # 9. 转换格式
    R_tensor = torch.stack([R.real, R.imag], dim=1).float()
    
    # 10. 标签归一化
    r_norm = r / cfg.r_max
    theta_norm = (theta - cfg.theta_min) / (cfg.theta_max - cfg.theta_min)
    labels = torch.cat([r_norm, theta_norm], dim=1).float()
    
    return R_tensor, labels, snr_db.squeeze(1)


def train_with_curriculum(model, device, epochs=100, batch_size=64,
                          snr_range_final=(-20, 30), curriculum_epochs=30):
    """
    课程学习训练
    
    前 curriculum_epochs: 逐渐降低 SNR 下限
    后续: 使用完整 SNR 范围 + 低SNR过采样
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SNRAwareLoss(snr_min=snr_range_final[0], snr_max=snr_range_final[1])
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_rmse_r': [], 'val_rmse_theta': []}
    
    batches_per_epoch = 200
    val_batches = 50
    
    print("="*70)
    print("开始课程学习训练")
    print("="*70)
    print(f"设备: {device}")
    print(f"Epochs: {epochs}")
    print(f"课程学习阶段: {curriculum_epochs} epochs")
    print(f"最终SNR范围: {snr_range_final}")
    
    for epoch in range(epochs):
        model.train()
        
        # 课程学习: 逐渐降低SNR下限
        if epoch < curriculum_epochs:
            progress = epoch / curriculum_epochs
            snr_min = 0 - progress * (0 - snr_range_final[0])  # 从0逐渐降到-20
            snr_max = snr_range_final[1]
            low_snr_prob = 0.3 + 0.2 * progress  # 从0.3逐渐增加到0.5
        else:
            snr_min = snr_range_final[0]
            snr_max = snr_range_final[1]
            low_snr_prob = 0.5
        
        snr_range = (snr_min, snr_max)
        
        # 训练
        epoch_loss = 0
        for _ in range(batches_per_epoch):
            R, labels, snr_db = generate_batch_with_snr_bias(
                batch_size, device, snr_range, low_snr_prob
            )
            
            optimizer.zero_grad()
            preds = model(R)
            loss = criterion(preds, labels, snr_db)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= batches_per_epoch
        history['train_loss'].append(epoch_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for _ in range(val_batches):
                R, labels, snr_db = generate_batch_with_snr_bias(
                    batch_size, device, snr_range_final, low_snr_prob=0.5
                )
                preds = model(R)
                val_loss += criterion(preds, labels, snr_db).item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        val_loss /= val_batches
        history['val_loss'].append(val_loss)
        
        # 计算RMSE
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        preds_phys = denormalize_labels(all_preds)
        labels_phys = denormalize_labels(all_labels)
        
        rmse_r = np.sqrt(np.mean((preds_phys[:, 0] - labels_phys[:, 0])**2))
        rmse_theta = np.sqrt(np.mean((preds_phys[:, 1] - labels_phys[:, 1])**2))
        history['val_rmse_r'].append(rmse_r)
        history['val_rmse_theta'].append(rmse_theta)
        
        scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'rmse_r': rmse_r,
                'rmse_theta': rmse_theta
            }, f'checkpoints/fda_cvnn_lowsnr_L{cfg.L_snapshots}_best.pth')
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            phase = "课程" if epoch < curriculum_epochs else "完整"
            print(f"Epoch {epoch+1:3d}/{epochs} [{phase}] SNR=[{snr_min:.0f},{snr_max:.0f}] "
                  f"Train={epoch_loss:.4f} Val={val_loss:.4f} "
                  f"RMSE_r={rmse_r:.2f}m RMSE_θ={rmse_theta:.2f}°")
    
    # 保存训练历史
    with open(f'results/training_lowsnr_L{cfg.L_snapshots}.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存至: checkpoints/fda_cvnn_lowsnr_L{cfg.L_snapshots}_best.pth")
    
    return history


def evaluate_by_snr(model, device, snr_list, num_samples=100):
    """按SNR分别评估模型"""
    model.eval()
    results = {}
    
    print("\n" + "="*70)
    print("按SNR评估")
    print("="*70)
    
    for snr in snr_list:
        errors_r = []
        errors_theta = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                R, labels, _ = generate_batch_with_snr_bias(
                    1, device, (snr, snr), low_snr_prob=1.0
                )
                preds = model(R)
                
                pred_phys = denormalize_labels(preds.cpu().numpy())
                label_phys = denormalize_labels(labels.cpu().numpy())
                
                errors_r.append((pred_phys[0, 0] - label_phys[0, 0])**2)
                errors_theta.append((pred_phys[0, 1] - label_phys[0, 1])**2)
        
        rmse_r = np.sqrt(np.mean(errors_r))
        rmse_theta = np.sqrt(np.mean(errors_theta))
        results[snr] = {'rmse_r': rmse_r, 'rmse_theta': rmse_theta}
        print(f"SNR={snr:+3d}dB: RMSE_r={rmse_r:.2f}m, RMSE_θ={rmse_theta:.2f}°")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, default=1, help='快拍数')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--eval-only', action='store_true', help='只评估不训练')
    args = parser.parse_args()
    
    # 设置快拍数
    cfg.L_snapshots = args.L
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"快拍数: L={args.L}")
    
    # 创建模型
    model = FDA_CVNN().to(device)
    
    # 加载已有权重(如果存在)
    model_path = f'checkpoints/fda_cvnn_lowsnr_L{args.L}_best.pth'
    if os.path.exists(model_path) and args.eval_only:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"加载模型: {model_path}")
    
    if not args.eval_only:
        # 训练
        history = train_with_curriculum(
            model, device,
            epochs=args.epochs,
            snr_range_final=(-20, 30),
            curriculum_epochs=30
        )
    
    # 评估
    snr_list = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    results = evaluate_by_snr(model, device, snr_list)
