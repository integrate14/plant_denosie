"""
StraightPCF LayerNorm 版本训练脚本
- 替换 BatchNorm1d → LayerNorm，消除混合形状训练时的 batch 统计偏移
- 使用 CosineAnnealingLR 学习率调度
- 使用梯度裁剪防止梯度爆炸
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
import json
from datetime import datetime

from data.synthetic_data_generator import load_dataset
from models.straight_pcf_improved import create_straightpcf_improved_model

print('='*70)
print('STRAIGHTPCF LAYERNORM TRAINING (Mixed Shape)')
print('='*70)
print(f'PyTorch: {torch.__version__}\n')

# ============================================================
# 数据准备
# ============================================================
DATA_PATH = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl'
CHECKPOINT_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\checkpoints'
RESULTS_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results'
NOISE_LEVEL = 0.02

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

data = load_dataset(DATA_PATH)
SHAPES = data['metadata']['shapes']
print(f'形状类型: {SHAPES}, 噪声级别: {NOISE_LEVEL}')


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


def build_data_loaders(clean_shapes, noisy_shapes, shapes, noise_level, batch_size=4,
                       train_ratio=0.7, val_ratio=0.15, seed=42):
    full_dataset = MixedShapeDataset(clean_shapes, noisy_shapes, shapes, noise_level)
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    print(f'混合数据集: Total={total_size}, Train={train_size}, Val={val_size}, Test={test_size}')

    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=gen
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            drop_last=True)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = build_data_loaders(
    data['clean'], data['noisy'], SHAPES, NOISE_LEVEL, batch_size=8
)


device = 'cpu'


def train_model(model, train_loader, val_loader, num_epochs, model_name, save_suffix=''):
    print(f'\n{"="*60}')
    print(f'Training: {model_name}{save_suffix}')
    print(f'{"="*60}')

    model = model.to(device)

    # CosineAnnealingLR 学习率调度
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    start_time = datetime.now()

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if len(batch) == 3:
                noisy, clean, _ = batch
            else:
                noisy, clean = batch
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            cleaned = model(noisy)
            if isinstance(cleaned, tuple):
                cleaned = cleaned[0]
            loss = model.get_loss(cleaned, clean)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    noisy, clean, _ = batch
                else:
                    noisy, clean = batch
                noisy, clean = noisy.to(device), clean.to(device)
                cleaned = model(noisy)
                if isinstance(cleaned, tuple):
                    cleaned = cleaned[0]
                loss = model.get_loss(cleaned, clean)
                val_loss += loss.item()
                n_val += 1

        val_loss /= max(n_val, 1)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(scheduler.get_last_lr()[0])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            ckpt_name = f'{model_name}{save_suffix}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'training_type': 'mixed_shape_ln',
                'shapes': SHAPES,
                'noise_level': NOISE_LEVEL,
            }, os.path.join(CHECKPOINT_DIR, ckpt_name))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f'  Epoch {epoch+1:3d}/{num_epochs}: '
                  f'Train={train_loss:.6f}, Val={val_loss:.6f} '
                  f'(best={best_val_loss:.6f}@ep{best_epoch}) [{elapsed:.0f}s]',
                  flush=True)

    total_elapsed = (datetime.now() - start_time).total_seconds()
    print(f'\n  Done! Best Val={best_val_loss:.6f} @ Epoch {best_epoch}, Total={total_elapsed:.0f}s',
          flush=True)

    hist_path = os.path.join(CHECKPOINT_DIR, f'{model_name}{save_suffix}_history.json')
    with open(hist_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'training_type': 'mixed_shape_ln',
            'shapes': SHAPES,
            'noise_level': NOISE_LEVEL,
            'num_epochs': num_epochs,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'total_time_seconds': total_elapsed,
            'history': history
        }, f, indent=2)

    return best_val_loss, history, best_epoch


# ============================================================
# 训练 StraightPCF LN
# ============================================================
NUM_EPOCHS = 50  # 已完成，重新训练时使用
SAVE_SUFFIX = '_spcf_ln'

print('\n\n' + '='*70)
print('TRAINING StraightPCF LayerNorm')
print('='*70)

spcf_ln_model = create_straightpcf_improved_model(
    num_points=2048, feature_dim=256, hidden_dim=128
)
best_val, history, best_ep = train_model(
    spcf_ln_model, train_loader, val_loader,
    num_epochs=NUM_EPOCHS, model_name='StraightPCF',
    save_suffix=SAVE_SUFFIX
)

print(f'\n训练完成! Best Val Loss = {best_val:.6f} @ Epoch {best_ep}')
print(f'检查点: checkpoints/StraightPCF{SAVE_SUFFIX}_best.pth')
