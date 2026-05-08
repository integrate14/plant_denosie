import os
os.environ['TORCH_DISABLE_CUDA_INIT'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USERNAME'] = 'User'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\torch_cache'

import sys
sys.path.insert(0, r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising')

print('='*70)
print('Plant Point Cloud Denoising - Complete Training & Evaluation')
print('='*70)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from datetime import datetime

from data.synthetic_data_generator import SyntheticPointCloudDataset, load_dataset
from models.pointfilter import PointFilter, create_pointfilter_model
from models.iterative_pfn import IterativePFN, create_iterativepfn_model
from models.straight_pcf import StraightPCF, create_straightpcf_model

print(f'PyTorch: {torch.__version__}')
print()

# 数据准备
print('Preparing data...')
data_path = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl'
checkpoint_dir = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

data = load_dataset(data_path)
clean_shapes = data['clean']
noisy_shapes = data['noisy']

class PointCloudDataset(Dataset):
    def __init__(self, clean_shapes, noisy_shapes, shape='sphere', noise_level=0.02):
        self.clean = clean_shapes[shape]
        self.noisy = noisy_shapes[shape][noise_level]
    def __len__(self):
        return len(self.clean)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.noisy[idx]).float(),
                torch.from_numpy(self.clean[idx]['points']).float())

dataset = PointCloudDataset(clean_shapes, noisy_shapes, shape='sphere', noise_level=0.02)
train_size = int(0.8 * len(dataset))
train_subset, val_subset = random_split(dataset, [train_size, len(dataset)-train_size])
train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)
print(f'Data: Train {len(train_subset)}, Val {len(val_subset)}\n')

device = 'cpu'


def train_model(model, train_loader, val_loader, num_epochs, model_name):
    """训练单个模型"""
    print(f'{model_name}')
    print('-'*50)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_val_loss = float('inf')
    best_epoch = 0
    start = datetime.now()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()

            cleaned = model(noisy)
            if isinstance(cleaned, tuple):
                cleaned = cleaned[0]

            # 根据模型类型计算损失
            if model_name == 'PointFilter':
                loss = model.get_loss(cleaned, clean)
            elif model_name == 'IterativePFN':
                loss = model.get_loss(noisy, clean)
            elif model_name == 'StraightPCF':
                cd = model.chamfer_distance(cleaned, clean)
                ptp = torch.mean(torch.sqrt(torch.sum((cleaned - clean)**2, dim=-1) + 1e-8))
                loss = cd + 0.1 * ptp

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
                if isinstance(cleaned, tuple):
                    cleaned = cleaned[0]
                if model_name == 'PointFilter':
                    loss = model.get_loss(cleaned, clean)
                elif model_name == 'IterativePFN':
                    loss = model.get_loss(noisy, clean)
                elif model_name == 'StraightPCF':
                    cd = model.chamfer_distance(cleaned, clean)
                    ptp = torch.mean(torch.sqrt(torch.sum((cleaned - clean)**2, dim=-1) + 1e-8))
                    loss = cd + 0.1 * ptp
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
            }, os.path.join(checkpoint_dir, f'{model_name}_best.pth'))

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1}/{num_epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}')

    elapsed = (datetime.now() - start).total_seconds()
    print(f'  Best: epoch {best_epoch}, Val={best_val_loss:.6f}, Time={elapsed:.0f}s\n')
    return best_val_loss


def evaluate_model(model, val_loader, model_name):
    """评估模型"""
    model.eval()
    total_cd = 0.0
    total_ptp = 0.0
    n = 0

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            cleaned = model(noisy)
            if isinstance(cleaned, tuple):
                cleaned = cleaned[0]

            cd = model.chamfer_distance(cleaned, clean)
            ptp = torch.mean(torch.sqrt(torch.sum((cleaned - clean)**2, dim=-1) + 1e-8))

            total_cd += cd.item()
            total_ptp += ptp.item()
            n += 1

    avg_cd = total_cd / max(n, 1)
    avg_ptp = total_ptp / max(n, 1)
    return avg_cd, avg_ptp


# 训练模型
print('='*70)
print('TRAINING')
print('='*70)

# PointFilter
model_pf = create_pointfilter_model(num_points=2048)
train_model(model_pf, train_loader, val_loader, num_epochs=50, model_name='PointFilter')

# IterativePFN (使用feature_dim=256, hidden_dim=128)
model_ipfn = create_iterativepfn_model(num_points=2048, num_iterations=3, feature_dim=256, hidden_dim=128)
train_model(model_ipfn, train_loader, val_loader, num_epochs=50, model_name='IterativePFN')

# StraightPCF (使用feature_dim=256, hidden_dim=128)
model_spcf = create_straightpcf_model(num_points=2048, feature_dim=256, hidden_dim=128)
train_model(model_spcf, train_loader, val_loader, num_epochs=50, model_name='StraightPCF')

# 评估
print('='*70)
print('EVALUATION')
print('='*70)

results = {}
for model, name in [(model_pf, 'PointFilter'), (model_ipfn, 'IterativePFN'), (model_spcf, 'StraightPCF')]:
    # 加载最佳检查点
    ckpt_path = os.path.join(checkpoint_dir, f'{name}_best.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    cd, ptp = evaluate_model(model, val_loader, name)
    results[name] = {'chamfer': cd, 'ptp': ptp}
    print(f'{name}: Chamfer={cd:.6f}, P2P={ptp:.6f}')

# 保存结果
import json
results_path = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results\experiment_results.json'
os.makedirs(os.path.dirname(results_path), exist_ok=True)

# 加载现有结果
existing_results = {}
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        existing_results = json.load(f)

# 更新结果
existing_results['final_comparison'] = {
    'timestamp': datetime.now().isoformat(),
    'dataset': 'synthetic (sphere)',
    'noise_level': 0.02,
    'models': results
}

with open(results_path, 'w') as f:
    json.dump(existing_results, f, indent=2)

print(f'\nResults saved to: {results_path}')
print('\nDone!')
