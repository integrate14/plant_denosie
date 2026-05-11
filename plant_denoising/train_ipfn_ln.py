"""
IterativePFN LayerNorm版训练脚本
BN -> LN 优化测试

核心改动：
1. SimplePointNet: 所有 BatchNorm1d 替换为 LayerNorm
2. IterationModule.mlp: BatchNorm1d 替换为 LayerNorm
3. 梯度裁剪 + CosineAnnealingLR
4. 训练 IterativePFN + PointFilter 对比基线
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
from models.pointfilter import create_pointfilter_model
from models.iterative_pfn_improved import create_iterativepfn_improved_model

print('=' * 70)
print('IPFN LayerNorm Training: Mixed Shapes')
print('=' * 70)
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
print(f'形状: {SHAPES}, 噪声级别: {NOISE_LEVEL}')


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
        self.all_noisy = []
        self.all_clean = []
        self.shape_indices = []
        for si, shape in enumerate(shapes):
            ds = SingleShapeDataset(clean_shapes, noisy_shapes, shape, noise_level)
            for i in range(len(ds)):
                n, c = ds[i]
                self.all_noisy.append(n)
                self.all_clean.append(c)
                self.shape_indices.append(si)
        self.total_length = len(self.all_noisy)

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
    full_dataset, [train_size, val_size, test_size], generator=gen)

train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, drop_last=True)
test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

print(f'数据划分: Train={train_size}, Val={val_size}, Test={test_size}\n')

device = 'cpu'


# ============================================================
# 评估函数
# ============================================================
def compute_metrics(pred, clean, noise_level=NOISE_LEVEL):
    """计算 CD / P2P / GR"""
    threshold = 2 * noise_level
    # CD (平方距离版)
    p1e = pred.unsqueeze(2)
    p2e = clean.unsqueeze(1)
    dist = torch.sum((p1e - p2e) ** 2, dim=-1)
    cd1 = torch.mean(torch.min(dist, dim=2)[0], dim=1)
    cd2 = torch.mean(torch.min(dist, dim=1)[0], dim=1)
    cd = torch.mean(cd1 + cd2)
    # P2P
    p2p = torch.mean(torch.sqrt(torch.sum((pred - clean) ** 2, dim=-1) + 1e-8))
    # GR (Recall方向, dim=0)
    dist_l2 = torch.cdist(pred, clean)
    min_per_clean, _ = torch.min(dist_l2, dim=0)
    gr = (min_per_clean <= threshold).float().mean()
    return cd.item(), p2p.item(), gr.item()


def evaluate_model(model, shapes, noise_level):
    """按形状分别评估"""
    results = {}
    for shape in shapes:
        ds = SingleShapeDataset(data['clean'], data['noisy'], shape, noise_level)
        total = len(ds)
        tr, va, te = random_split(ds, [int(0.7*total), int(0.15*total),
                                       total - int(0.7*total) - int(0.15*total)],
                                  generator=gen)
        loader = DataLoader(te, batch_size=8, shuffle=False)

        total_cd, total_p2p, total_gr, n = 0, 0, 0, 0
        model.eval()
        with torch.no_grad():
            for noisy, clean in loader:
                noisy, clean = noisy.to(device), clean.to(device)
                cleaned = model(noisy)
                if isinstance(cleaned, tuple):
                    cleaned = cleaned[0]
                cd, p2p, gr = compute_metrics(cleaned[0], clean[0], noise_level)
                total_cd += cd
                total_p2p += p2p
                total_gr += gr
                n += 1

        results[shape] = {
            'CD': total_cd / max(n, 1),
            'P2P': total_p2p / max(n, 1),
            'GR': total_gr / max(n, 1),
        }
    return results


# ============================================================
# 训练函数
# ============================================================
def train_model(model, train_loader, val_loader, num_epochs, model_name, save_suffix=''):
    print(f'\nTraining: {model_name}{save_suffix}')
    print('-' * 50)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    start_time = datetime.now()

    for epoch in range(num_epochs):
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

            if model_name == 'IterativePFN':
                loss = model.get_iterative_loss(noisy, clean)
            else:
                cleaned = model(noisy)
                if isinstance(cleaned, tuple):
                    cleaned = cleaned[0]
                loss = model.get_loss(cleaned, clean)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

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
                if model_name == 'IterativePFN':
                    loss = model.get_iterative_loss(noisy, clean)
                else:
                    cleaned = model(noisy)
                    if isinstance(cleaned, tuple):
                        cleaned = cleaned[0]
                    loss = model.get_loss(cleaned, clean)
                val_loss += loss.item()
                n_val += 1

        val_loss /= max(n_val, 1)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            ckpt_name = f'{model_name}{save_suffix}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'training_type': 'mixed_ln',
                'shapes': SHAPES,
                'noise_level': NOISE_LEVEL,
            }, os.path.join(CHECKPOINT_DIR, ckpt_name))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f'  Ep {epoch+1:3d}/{num_epochs}: Train={train_loss:.6f}, '
                  f'Val={val_loss:.6f} (best={best_val_loss:.6f}@ep{best_epoch}) '
                  f'[{elapsed:.0f}s] LR={current_lr:.2e}', flush=True)

    total_elapsed = (datetime.now() - start_time).total_seconds()
    print(f'  Done! Val={best_val_loss:.6f} @ ep{best_epoch}, Total={total_elapsed:.0f}s')
    return best_val_loss, best_epoch, history


# ============================================================
# 训练配置
# ============================================================
NUM_EPOCHS = 50
SAVE_SUFFIX = '_ln'

models_config = {
    'IPFN_LN': lambda: create_iterativepfn_improved_model(
        num_points=2048, num_iterations=3, feature_dim=256, hidden_dim=128),
}

all_results = {}

for name, model_fn in models_config.items():
    model = model_fn()
    best_val, best_ep, history = train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, model_name='IterativePFN',
        save_suffix=SAVE_SUFFIX)

    # 加载最佳检查点评估
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'IterativePFN{SAVE_SUFFIX}_best.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)['model_state_dict'])

    per_shape = evaluate_model(model, SHAPES, NOISE_LEVEL)
    all_results[name] = {
        'best_val_loss': best_val,
        'best_epoch': best_ep,
        'per_shape': per_shape,
        'history': history,
    }

    print(f'\n--- {name} 各形状评估 ---')
    print(f'  {"Shape":10s} | {"CD":>12s} | {"P2P":>12s} | {"GR":>10s}')
    print(f'  {"-"*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}')
    for shape in SHAPES:
        r = per_shape[shape]
        print(f'  {shape:10s} | {r["CD"]:>12.6f} | {r["P2P"]:>12.6f} | {r["GR"]:>10.4f}')

    # Mean
    mean_cd = np.mean([per_shape[s]['CD'] for s in SHAPES])
    mean_p2p = np.mean([per_shape[s]['P2P'] for s in SHAPES])
    mean_gr = np.mean([per_shape[s]['GR'] for s in SHAPES])
    print(f'  {"MEAN":10s} | {mean_cd:>12.6f} | {mean_p2p:>12.6f} | {mean_gr:>10.4f}')

# ============================================================
# 保存结果
# ============================================================
output_path = os.path.join(RESULTS_DIR, 'ipfn_ln_results.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'model': 'IterativePFN_LayerNorm',
        'epochs': NUM_EPOCHS,
        'lr': 0.0005,
        'noise_level': NOISE_LEVEL,
        'shapes': SHAPES,
        'results': all_results,
    }, f, indent=2, ensure_ascii=False)

print(f'\n结果已保存: {output_path}')
print('\n' + '=' * 70)
print('TRAINING COMPLETE!')
print('=' * 70)
