"""
PyTorch Dataset 封装
支持在线生成和离线加载
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config as cfg
from utils_physics import generate_covariance_matrix, generate_batch_torch


class FastDataLoader:
    """
    极速数据加载器 - 直接在GPU上生成数据
    替代标准的 DataLoader，避免 CPU->GPU 传输瓶颈
    """
    def __init__(self, batch_size, num_samples, snr_range=None, device=cfg.device):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_batches = num_samples // batch_size
        self.snr_range = snr_range
        self.device = device
        
    def __iter__(self):
        for _ in range(self.num_batches):
            yield generate_batch_torch(self.batch_size, self.device, self.snr_range)
            
    def __len__(self):
        return self.num_batches


class FDADataset(Dataset):
    """
    FDA-MIMO 数据集
    
    支持两种模式:
    1. 在线生成 (online=True): 每次访问生成新数据，防止过拟合
    2. 离线模式 (online=False): 预生成固定数据集
    """
    def __init__(self, num_samples, snr_db=None, snr_range=None, online=True, seed=None):
        """
        参数:
            num_samples: 样本数量
            snr_db: 固定SNR值 (dB)
            snr_range: SNR范围 (min, max)，用于随机采样
            online: 是否在线生成数据
            seed: 随机种子 (仅离线模式有效)
        """
        self.num_samples = num_samples
        self.snr_db = snr_db
        self.snr_range = snr_range
        self.online = online
        
        if not online:
            # 离线模式：预生成所有数据
            if seed is not None:
                np.random.seed(seed)
            self._generate_offline_data()
            
    def _generate_offline_data(self):
        """预生成离线数据"""
        print(f"预生成 {self.num_samples} 个样本...")
        self.data = []
        self.labels = []
        self.params = []  # 保存真实参数用于评估
        
        for i in range(self.num_samples):
            # 随机参数
            r = np.random.uniform(cfg.r_min, cfg.r_max)
            theta = np.random.uniform(cfg.theta_min, cfg.theta_max)
            
            # SNR
            if self.snr_db is not None:
                snr = self.snr_db
            elif self.snr_range is not None:
                snr = np.random.uniform(self.snr_range[0], self.snr_range[1])
            else:
                snr = np.random.uniform(cfg.snr_train_min, cfg.snr_train_max)
            
            # 生成数据
            R = generate_covariance_matrix(r, theta, snr)
            
            # 归一化标签
            r_norm = r / cfg.r_max
            theta_norm = (theta - cfg.theta_min) / (cfg.theta_max - cfg.theta_min)
            
            self.data.append(R)
            self.labels.append([r_norm, theta_norm])
            self.params.append([r, theta, snr])
            
            if (i + 1) % 1000 == 0:
                print(f"  已生成 {i+1}/{self.num_samples}")
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.params = np.array(self.params, dtype=np.float32)
        print("数据生成完成！")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.online:
            # 在线生成
            r = np.random.uniform(cfg.r_min, cfg.r_max)
            theta = np.random.uniform(cfg.theta_min, cfg.theta_max)
            
            if self.snr_db is not None:
                snr = self.snr_db
            elif self.snr_range is not None:
                snr = np.random.uniform(self.snr_range[0], self.snr_range[1])
            else:
                snr = np.random.uniform(cfg.snr_train_min, cfg.snr_train_max)
            
            R = generate_covariance_matrix(r, theta, snr)
            
            r_norm = r / cfg.r_max
            theta_norm = (theta - cfg.theta_min) / (cfg.theta_max - cfg.theta_min)
            
            return torch.FloatTensor(R), torch.FloatTensor([r_norm, theta_norm])
        else:
            # 离线模式
            return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.labels[idx])
    
    def get_params(self, idx):
        """获取真实参数 (仅离线模式)"""
        if not self.online:
            return self.params[idx]
        return None


def create_dataloaders(train_samples=None, val_samples=None, test_samples=None,
                       batch_size=None, snr_train_range=None, snr_test=None,
                       online_train=True, num_workers=0):
    """
    创建训练、验证、测试数据加载器
    
    参数:
        train_samples: 训练样本数
        val_samples: 验证样本数
        test_samples: 测试样本数
        batch_size: 批次大小
        snr_train_range: 训练SNR范围
        snr_test: 测试固定SNR
        online_train: 训练数据是否在线生成
        num_workers: 数据加载线程数
    
    返回:
        train_loader, val_loader, test_loader
    """
    train_samples = train_samples or cfg.train_samples
    val_samples = val_samples or cfg.val_samples
    test_samples = test_samples or cfg.test_samples
    batch_size = batch_size or cfg.batch_size
    snr_train_range = snr_train_range or (cfg.snr_train_min, cfg.snr_train_max)
    snr_test = snr_test if snr_test is not None else cfg.snr_test
    
    # 训练集 (在线生成，每个epoch数据不同)
    train_dataset = FDADataset(
        train_samples, 
        snr_range=snr_train_range,
        online=online_train
    )
    
    # 验证集 (离线，固定数据便于比较)
    val_dataset = FDADataset(
        val_samples,
        snr_range=snr_train_range,
        online=False,
        seed=cfg.seed
    )
    
    # 测试集 (离线，固定SNR)
    test_dataset = FDADataset(
        test_samples,
        snr_db=snr_test,
        online=False,
        seed=cfg.seed + 1
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("测试数据集...")
    
    # 测试在线数据集
    print("\n1. 测试在线数据集")
    dataset_online = FDADataset(100, snr_db=20, online=True)
    x, y = dataset_online[0]
    print(f"   样本形状: x={x.shape}, y={y.shape}")
    
    # 测试离线数据集
    print("\n2. 测试离线数据集")
    dataset_offline = FDADataset(100, snr_db=20, online=False, seed=42)
    x, y = dataset_offline[0]
    print(f"   样本形状: x={x.shape}, y={y.shape}")
    print(f"   真实参数: {dataset_offline.get_params(0)}")
    
    # 测试DataLoader
    print("\n3. 测试 DataLoader")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples=200,
        val_samples=50,
        test_samples=50,
        batch_size=16
    )
    
    for batch_x, batch_y in train_loader:
        print(f"   训练批次: x={batch_x.shape}, y={batch_y.shape}")
        break
    
    print("\n测试完成！")
