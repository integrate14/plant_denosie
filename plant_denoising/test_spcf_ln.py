"""
测试 StraightPCF LN 训练 - 只运行1个epoch的少量batch
"""
import os
os.environ['TORCH_DISABLE_CUDA_INIT'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USERNAME'] = 'User'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\torch_cache'

import sys
sys.path.insert(0, r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

from data.synthetic_data_generator import load_dataset
from models.straight_pcf_improved import create_straightpcf_improved_model

print('开始测试训练...')

# 加载数据
DATA_PATH = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl'
data = load_dataset(DATA_PATH)
SHAPES = data['metadata']['shapes']
NOISE_LEVEL = 0.02

class SingleShapeDataset(Dataset):
    def __init__(self, clean_shapes, noisy_shapes, shape, noise_level):
        self.clean = clean_shapes[shape]
        self.noisy = noisy_shapes[shape][noise_level]

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.noisy[idx]).float(),
                torch.from_numpy(self.clean[idx]['points']).float())

class MixedShapeDataset(Dataset):
    def __init__(self, clean_shapes, noisy_shapes, shapes, noise_level):
        self.datasets = []
        self.shape_indices = []
        for si, shape in enumerate(shapes):
            ds = SingleShapeDataset(clean_shapes, noisy_shapes, shape, noise_level)
            self.datasets.append(ds)
            self.shape_indices.extend([si] * len(ds))
        self.total_length = sum(len(ds) for ds in self.datasets)
        self.all_noisy = []
        self.all_clean = []
        for ds in self.datasets:
            for i in range(len(ds)):
                n, c = ds[i]
                self.all_noisy.append(n)
                self.all_clean.append(c)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        return (self.all_noisy[idx], self.all_clean[idx], self.shape_indices[idx])

full_dataset = MixedShapeDataset(data['clean'], data['noisy'], SHAPES, NOISE_LEVEL)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

gen = torch.Generator().manual_seed(42)
train_subset, val_subset, test_subset = random_split(
    full_dataset, [train_size, val_size, test_size], generator=gen
)

train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, drop_last=True)

print(f'训练集: {len(train_subset)}, 验证集: {len(val_subset)}')
print(f'训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}')

# 创建模型
device = 'cpu'
model = create_straightpcf_improved_model(num_points=2048, feature_dim=256, hidden_dim=128)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

print('\n测试训练的前3个batch...')
model.train()
for i, batch in enumerate(train_loader):
    if len(batch) == 3:
        noisy, clean, _ = batch
    else:
        noisy, clean = batch
    noisy, clean = noisy.to(device), clean.to(device)
    
    print(f'  Batch {i+1}: noisy={noisy.shape}, clean={clean.shape}')
    
    optimizer.zero_grad()
    cleaned = model(noisy)
    if isinstance(cleaned, tuple):
        cleaned = cleaned[0]
    loss = model.get_loss(cleaned, clean)
    print(f'  Loss: {loss.item():.6f}')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print(f'  Batch {i+1} 完成')
    
    if i >= 2:  # 只测试3个batch
        break

print('\n测试验证的前2个batch...')
model.eval()
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if len(batch) == 3:
            noisy, clean, _ = batch
        else:
            noisy, clean = batch
        noisy, clean = noisy.to(device), clean.to(device)
        
        print(f'  Val Batch {i+1}: noisy={noisy.shape}, clean={clean.shape}')
        
        cleaned = model(noisy)
        if isinstance(cleaned, tuple):
            cleaned = cleaned[0]
        loss = model.get_loss(cleaned, clean)
        print(f'  Val Loss: {loss.item():.6f}')
        print(f'  Val Batch {i+1} 完成')
        
        if i >= 1:  # 只测试2个batch
            break

print('\n测试完成！训练脚本应该可以正常运行。')
