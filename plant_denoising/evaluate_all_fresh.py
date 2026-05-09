"""
评估所有模型 - 使用新训练的 IterativePFN
"""
import os
os.environ['TORCH_DISABLE_CUDA_INIT'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USERNAME'] = 'User'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\torch_cache'

import sys
sys.path.insert(0, r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising')

import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime

from data.synthetic_data_generator import load_dataset
from models.pointfilter import create_pointfilter_model
from models.iterative_pfn_improved import create_iterativepfn_improved_model
from models.straight_pcf_improved import create_straightpcf_improved_model

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
test_loader = DataLoader(test_subset, batch_size=4, shuffle=False)

print('='*60)
print('Evaluating All Models (70/15/15 Split)')
print('='*60)
print(f'Test set: {len(test_subset)} samples\n')

# 加载模型
def load_checkpoint(model, name):
    path = os.path.join(CHECKPOINT_DIR, f'{name}_best.pth')
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'  {name}: loaded from {path}')
    return model

# 加载三个模型
model_pf = create_pointfilter_model(num_points=2048)
model_pf = load_checkpoint(model_pf, 'PointFilter')
model_pf = model_pf.to(device).eval()

model_ipfn = create_iterativepfn_improved_model(num_points=2048, num_iterations=3, feature_dim=256, hidden_dim=128)
model_ipfn = load_checkpoint(model_ipfn, 'IterativePFN')
model_ipfn = model_ipfn.to(device).eval()

model_pcf = create_straightpcf_improved_model(num_points=2048, feature_dim=256, hidden_dim=128)
model_pcf = load_checkpoint(model_pcf, 'StraightPCF')
model_pcf = model_pcf.to(device).eval()

# 评估函数
def evaluate_model(model, loader, model_name, model_type='pf'):
    total_cd = 0
    total_p2p = 0
    total_gr = 0
    n = 0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)

            if model_type == 'pf':
                output = model(noisy)
                cleaned = output[0] if isinstance(output, tuple) else output
            elif model_type == 'ipfn':
                output = model(noisy)
                cleaned = output[0] if isinstance(output, tuple) else output
            elif model_type == 'pcf':
                output = model(noisy)
                cleaned = output[0] if isinstance(output, tuple) else output
            else:
                cleaned = model(noisy)

            # Chamfer Distance
            cd = model.chamfer_distance(cleaned, clean)

            # P2P Distance
            p2p = torch.mean(torch.sqrt(torch.sum((cleaned - clean) ** 2, dim=-1) + 1e-8))

            # Geometric Recall (τ = 0.01)
            dists = torch.cdist(cleaned, clean)
            min_dists_cleaned, _ = torch.min(dists, dim=2)
            gr = torch.mean((min_dists_cleaned < 0.01).float())

            total_cd += cd.item()
            total_p2p += p2p.item()
            total_gr += gr.item()
            n += 1

    return {
        'Chamfer Distance': total_cd / n,
        'P2P Distance': total_p2p / n,
        'Geometric Recall': total_gr / n
    }

print('\n--- Test Set Results ---')
results = {}

print('PointFilter...')
results['PointFilter'] = evaluate_model(model_pf, test_loader, 'PointFilter', 'pf')

print('IterativePFN (重新训练)...')
results['IterativePFN'] = evaluate_model(model_ipfn, test_loader, 'IterativePFN', 'ipfn')

print('StraightPCF...')
results['StraightPCF'] = evaluate_model(model_pcf, test_loader, 'StraightPCF', 'pcf')

# 打印结果
print('\n' + '='*60)
print(f'Test Set Evaluation Results (n={len(test_subset)})')
print('='*60)
print(f'{"Model":<15} {"CD ↓":<12} {"P2P ↓":<12} {"GR ↑":<10}')
print('-'*50)
for name, r in sorted(results.items(), key=lambda x: x[1]['Chamfer Distance']):
    print(f'{name:<15} {r["Chamfer Distance"]:<12.6f} {r["P2P Distance"]:<12.6f} {r["Geometric Recall"]:<10.4f}')

# 保存结果
output = {
    'timestamp': datetime.now().isoformat(),
    'dataset_split': {'train': train_size, 'val': val_size, 'test': test_size},
    'results': results
}

with open(os.path.join(r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising', 'results', 'experiment_results.json'), 'w') as f:
    json.dump(output, f, indent=2)

print('\nResults saved to results/experiment_results.json')
