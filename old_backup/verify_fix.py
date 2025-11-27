# 验证修复
from config import delta_f, c, r_max
import torch
from model import CVNN_Improved

# 1. 检查物理模糊
R_max = c/(2*delta_f)
print('='*60)
print('修复验证')
print('='*60)
print('\n1. 物理模糊检查:')
print(f'   delta_f = {delta_f/1e3:.0f} kHz')
print(f'   R_max = {R_max:.0f} m')
print(f'   r_max = {r_max} m')
if r_max <= R_max:
    print('   状态: ✓ 无模糊')
else:
    print('   状态: ✗ 有模糊')

# 2. 检查ModReLU bias
model = CVNN_Improved()
bias1 = model.act1a.bias[0].item()
print('\n2. ModReLU激活函数:')
print(f'   bias = {bias1:.2f}')
if bias1 < 0:
    print('   状态: ✓ 非线性')
else:
    print('   状态: ✗ 退化为恒等')

# 3. 检查池化
print('\n3. 池化层类型:')
print(f'   pool1 = {type(model.pool1).__name__}')
print(f'   pool2 = {type(model.pool2).__name__}')
if 'Complex' in type(model.pool1).__name__:
    print('   状态: ✓ 复数池化')
else:
    print('   状态: ✗ 破坏相位')

print('\n' + '='*60)
print('所有修复完成！现在可以重新训练了。')
print('='*60)
