# 快速测试脚本
import torch
from dataset import FDADataset
from model import get_model

print("测试数据集...")
ds = FDADataset(10, fixed_snr=30.0, seed=42)
item = ds[0]
print(f"数据集返回: {type(item)}")
print(f"长度: {len(item)}")
if len(item) == 2:
    X, y = item
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"y value: {y}")

print("\n测试模型...")
model = get_model('cvnn')
x_test = torch.randn(2, 2, 100, 100)
out = model(x_test)
print(f"输出: {out}")
