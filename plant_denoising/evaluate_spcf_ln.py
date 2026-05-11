"""
评估 StraightPCF LN 模型 - 综合指标测试
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
from torch.utils.data import Dataset, DataLoader, random_split
import json
from datetime import datetime

from data.synthetic_data_generator import load_dataset
from models.straight_pcf_improved import create_straightpcf_improved_model

print('='*70)
print('STRAIGHTPCF LAYERNORM EVALUATION')
print('='*70)

DATA_PATH = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl'
CHECKPOINT_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\checkpoints'
NOISE_LEVEL = 0.02

data = load_dataset(DATA_PATH)
SHAPES = data['metadata']['shapes']

class SingleShapeDataset(Dataset):
    def __init__(self, clean_shapes, noisy_shapes, shape, noise_level):
        self.clean = clean_shapes[shape]
        self.noisy = noisy_shapes[shape][noise_level]

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.noisy[idx]).float(),
                torch.from_numpy(self.clean[idx]['points']).float())

device = 'cpu'

# 加载模型
model = create_straightpcf_improved_model(num_points=2048, feature_dim=256, hidden_dim=128)
ckpt_path = os.path.join(CHECKPOINT_DIR, 'StraightPCF_spcf_ln_best.pth')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()

print(f'模型加载: {ckpt_path}')
print(f'Val Loss: {ckpt["val_loss"]:.6f} @ Epoch {ckpt["epoch"]+1}')
print()

# 评估指标计算（与 evaluate_comprehensive.py 一致）
THRESHOLD = 0.04  # GR 阈值

results = {}
all_cd = []
all_p2p = []
all_gr = []

for shape in SHAPES:
    shape_dataset = SingleShapeDataset(data['clean'], data['noisy'], shape, NOISE_LEVEL)
    total_size = len(shape_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    gen = torch.Generator().manual_seed(42)
    _, _, test_subset = random_split(shape_dataset, [train_size, val_size, test_size], generator=gen)

    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    total_cd = 0; total_p2p = 0; total_gr = 0; n = 0
    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            cleaned = model(noisy)
            if isinstance(cleaned, tuple):
                cleaned = cleaned[0]

            # CD (平方距离)
            p1e = cleaned.unsqueeze(2)
            p2e = clean.unsqueeze(1)
            dist_sq = torch.sum((p1e - p2e) ** 2, dim=-1)
            cd = torch.mean(torch.min(dist_sq, dim=2)[0]) + torch.mean(torch.min(dist_sq, dim=1)[0])

            # P2P (欧氏距离)
            p2p = torch.mean(torch.sqrt(torch.sum((cleaned - clean) ** 2, dim=-1) + 1e-8))

            # GR (Recall - dim=0: 每个clean点被pred覆盖的比例)
            dist = torch.cdist(cleaned, clean)
            min_dists, _ = torch.min(dist, dim=0)
            gr = torch.mean((min_dists < THRESHOLD).float())

            total_cd += cd.item()
            total_p2p += p2p.item()
            total_gr += gr.item()
            n += 1

    mean_cd = total_cd / max(n, 1)
    mean_p2p = total_p2p / max(n, 1)
    mean_gr = total_gr / max(n, 1)

    results[shape] = {
        'CD': mean_cd,
        'P2P': mean_p2p,
        'GR': mean_gr,
        'samples': n
    }
    all_cd.append(mean_cd)
    all_p2p.append(mean_p2p)
    all_gr.append(mean_gr)

    print(f'{shape:10s}: CD={mean_cd:.6f}, P2P={mean_p2p:.6f}, GR={mean_gr:.4f}')

# 汇总
print()
print('='*50)
print('SUMMARY')
print('='*50)
print(f'Mean CD:  {np.mean(all_cd):.6f}')
print(f'Mean P2P: {np.mean(all_p2p):.6f}')
print(f'Mean GR:  {np.mean(all_gr):.4f}')

# 保存结果
output = {
    'model': 'StraightPCF_LN',
    'checkpoint': ckpt_path,
    'val_loss': float(ckpt['val_loss']),
    'threshold': THRESHOLD,
    'per_shape': results,
    'summary': {
        'mean_cd': float(np.mean(all_cd)),
        'mean_p2p': float(np.mean(all_p2p)),
        'mean_gr': float(np.mean(all_gr)),
        'std_cd': float(np.std(all_cd)),
        'std_p2p': float(np.std(all_p2p)),
        'std_gr': float(np.std(all_gr))
    },
    'timestamp': datetime.now().isoformat()
}

out_path = os.path.join(CHECKPOINT_DIR, '..', 'results', 'spcf_ln_evaluation.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f'\n结果已保存: {out_path}')
print('\n=== VIS_DATA ===')
vis = {
    'model': 'StraightPCF_LN',
    'mean_cd': float(np.mean(all_cd)),
    'mean_p2p': float(np.mean(all_p2p)),
    'mean_gr': float(np.mean(all_gr)),
    'per_shape': results
}
print(json.dumps(vis, indent=2))
