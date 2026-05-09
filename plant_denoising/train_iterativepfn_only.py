"""
单独训练 IterativePFN
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
from datetime import datetime

from data.synthetic_data_generator import load_dataset
from models.iterative_pfn_improved import create_iterativepfn_improved_model

print('='*60)
print('Training IterativePFN')
print('='*60)

device = 'cpu'
DATA_PATH = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl'
CHECKPOINT_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\checkpoints'

# 加载数据
data = load_dataset(DATA_PATH)

class PointCloudDataset(Dataset):
    def __init__(self, clean_shapes, noisy_shapes, shape='sphere', noise_level=0.02):
        self.clean = clean_shapes[shape]
        self.noisy = noisy_shapes[shape][noise_level]
    def __len__(self):
        return len(self.clean)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.noisy[idx]).float(),
                torch.from_numpy(self.clean[idx]['points']).float())

dataset = PointCloudDataset(data['clean'], data['noisy'], shape='sphere', noise_level=0.02)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)
print(f'Data: Train {len(train_subset)}, Val {len(val_subset)}, Test {test_size}\n')

# 创建模型
model = create_iterativepfn_improved_model(num_points=2048, num_iterations=3, feature_dim=256, hidden_dim=128)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

best_val_loss = float('inf')
best_epoch = 0
num_epochs = 100

print('Training IterativePFN...')
start = datetime.now()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    n_batches = 0

    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()

        # 关键：使用 cleaned (模型输出) 而不是 noisy
        cleaned = model(noisy)
        loss = model.get_loss(cleaned, clean)  # 正确：cleaned 需要梯度

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1

    train_loss /= max(n_batches, 1)

    # 验证
    model.eval()
    val_loss = 0.0
    n_val = 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            cleaned = model(noisy)
            loss = model.get_loss(cleaned, clean)
            val_loss += loss.item()
            n_val += 1

    val_loss /= max(n_val, 1)
    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(CHECKPOINT_DIR, 'IterativePFN_best.pth'))

    if (epoch+1) % 10 == 0:
        print(f'  Epoch {epoch+1}/{num_epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}')

elapsed = (datetime.now() - start).total_seconds()
print(f'\nBest: epoch {best_epoch}, Val={best_val_loss:.6f}, Time={elapsed:.0f}s')

# 测试评估
print('\nEvaluating on validation set...')
model.eval()
total_cd = 0
total_p2p = 0
n = 0
with torch.no_grad():
    for noisy, clean in val_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        cleaned = model(noisy)
        cd = model.chamfer_distance(cleaned, clean)
        p2p = torch.mean(torch.sqrt(torch.sum((cleaned - clean) ** 2, dim=-1) + 1e-8))
        total_cd += cd.item()
        total_p2p += p2p.item()
        n += 1

print(f'  Val CD: {total_cd/n:.6f}, P2P: {total_p2p/n:.6f}')
print('\nDone! Checkpoint saved to IterativePFN_best.pth')
