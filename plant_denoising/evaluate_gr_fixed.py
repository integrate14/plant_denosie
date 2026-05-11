"""
GR修复版评估脚本
Bug修复:
  1. 原代码用平方距离(d^2)与线性阈值(0.04)比较 → 改用 torch.cdist 获取真实欧氏距离
  2. 原代码 dim=1 (Precision方向) → 改为 dim=0 (Recall方向: 每个clean点能否被pred覆盖)
  3. 统一 THRESHOLD = 2 * NOISE_LEVEL (与噪声尺度对齐)
评估范围: 混合训练检查点 (PointFilter / IterativePFN / StraightPCF) 在 sphere+cube+cone 上
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
from datetime import datetime
import json

from data.synthetic_data_generator import load_dataset
from models.pointfilter import create_pointfilter_model
from models.iterative_pfn_improved import create_iterativepfn_improved_model
from models.straight_pcf_improved import StraightPCFImproved

print('='*70)
print('GR修复版评估  —  混合训练检查点  (sphere + cube + cone)')
print('='*70)

device = 'cpu'
DATA_PATH     = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl'
CHECKPOINT_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\checkpoints'
RESULTS_DIR   = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results'
NOISE_LEVEL   = 0.02
THRESHOLD     = 2 * NOISE_LEVEL   # 0.04 — 线性欧氏距离

print(f'THRESHOLD = {THRESHOLD}  (2 x noise_level={NOISE_LEVEL})')

data   = load_dataset(DATA_PATH)
SHAPES = ['sphere', 'cube', 'cone']
torch.manual_seed(42)


class PointCloudDataset(Dataset):
    def __init__(self, clean_shapes, noisy_shapes, shape, noise_level=0.02):
        self.clean = clean_shapes[shape]
        self.noisy = noisy_shapes[shape][noise_level]

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.noisy[idx]).float(),
                torch.from_numpy(self.clean[idx]['points']).float())


def compute_metrics_fixed(pred, clean):
    """
    正确版本:
      CD  — 平方 Chamfer (与之前保持一致)
      P2P — 逐点欧氏距离均值
      GR  — torch.cdist 线性距离, dim=0 (Recall: 每个clean点能否被pred覆盖)
    """
    if pred.dim() == 3:
        pred  = pred.squeeze(0)
    if clean.dim() == 3:
        clean = clean.squeeze(0)

    # Chamfer Distance (保持与历史一致: 平方)
    dist_sq = torch.sum((pred.unsqueeze(1) - clean.unsqueeze(0)) ** 2, dim=2)
    cd = torch.mean(torch.min(dist_sq, dim=1)[0]) + torch.mean(torch.min(dist_sq, dim=0)[0])

    # P2P (逐点, 要求pred与clean点数相同)
    p2p = torch.mean(torch.sqrt(torch.sum((pred - clean) ** 2, dim=-1) + 1e-8))

    # GR — 修复版
    # torch.cdist 返回真实欧氏距离 (N_pred, N_clean)
    dist_l2 = torch.cdist(pred, clean)                # (N_pred, N_clean)
    min_dist_per_clean, _ = torch.min(dist_l2, dim=0) # (N_clean,) — 每个clean点的最近pred点
    gr = (min_dist_per_clean <= THRESHOLD).float().mean()

    return cd.item(), p2p.item(), gr.item()


def evaluate_model(model, shape, noise_level=0.02, test_ratio=0.15, seed=42):
    """按 70/15/15 划分，取测试集评估"""
    dataset = PointCloudDataset(data['clean'], data['noisy'], shape, noise_level)
    n       = len(dataset)
    train_n = int(0.7 * n)
    val_n   = int(0.15 * n)
    test_n  = n - train_n - val_n
    generator = torch.Generator().manual_seed(seed)
    _, _, test_subset = torch.utils.data.random_split(
        dataset, [train_n, val_n, test_n], generator=generator
    )
    loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    model.eval()
    total_cd = total_p2p = total_gr = 0.0
    cnt = 0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            cleaned = model(noisy)
            if isinstance(cleaned, tuple):
                cleaned = cleaned[0]
            if cleaned.dim() == 3:
                cleaned = cleaned.squeeze(0)
            if clean.dim() == 3:
                clean = clean.squeeze(0)

            cd, p2p, gr = compute_metrics_fixed(cleaned, clean)
            total_cd  += cd
            total_p2p += p2p
            total_gr  += gr
            cnt       += 1

    return {
        'CD':  total_cd  / cnt,
        'P2P': total_p2p / cnt,
        'GR':  total_gr  / cnt,
        'n':   cnt,
    }


# ============================================================
# 加载混合训练检查点
# ============================================================
print('\n加载混合训练检查点...')

model_pf = create_pointfilter_model(num_points=2048)
ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'PointFilter_mixed_best.pth'),
                   map_location=device, weights_only=False)
model_pf.load_state_dict(ckpt['model_state_dict'])
print(f'  PointFilter   loaded  (val_loss={ckpt.get("val_loss", "?"):.6f})')

model_ipfn = create_iterativepfn_improved_model(
    num_points=2048, num_iterations=3, feature_dim=256, hidden_dim=128)
ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'IterativePFN_mixed_best.pth'),
                   map_location=device, weights_only=False)
model_ipfn.load_state_dict(ckpt['model_state_dict'])
print(f'  IterativePFN  loaded  (val_loss={ckpt.get("val_loss", "?"):.6f})')

model_spcf = StraightPCFImproved(num_points=2048, feature_dim=256, hidden_dim=128,
                                   num_iterations=3, use_dgcnn=False)
ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'StraightPCF_mixed_best.pth'),
                   map_location=device, weights_only=False)
model_spcf.load_state_dict(ckpt['model_state_dict'])
print(f'  StraightPCF   loaded  (val_loss={ckpt.get("val_loss", "?"):.6f})')

MODELS = {
    'PointFilter':  model_pf,
    'IterativePFN': model_ipfn,
    'StraightPCF':  model_spcf,
}

# ============================================================
# 评估
# ============================================================
print(f'\n评估 (THRESHOLD={THRESHOLD}, 召回方向=clean, 开方距离)...\n')

all_results = {}
for model_name, model in MODELS.items():
    all_results[model_name] = {}
    for shape in SHAPES:
        r = evaluate_model(model, shape)
        all_results[model_name][shape] = r
        print(f'  {model_name:14s} | {shape:6s} | CD={r["CD"]:.6f}  P2P={r["P2P"]:.6f}  GR={r["GR"]:.4f}  (n={r["n"]})')
    print()

# ============================================================
# 汇总表
# ============================================================
print('='*80)
print('SUMMARY  (GR修复后)')
print('='*80)
print(f'{"Model":<14} | {"Shape":<6} | {"CD ↓":>10} | {"P2P ↓":>10} | {"GR_old(≈)":>10} | {"GR_fixed ↑":>10}')
print('-'*75)

GR_OLD = {
    'PointFilter':  {'sphere': 0.0072, 'cube': 0.1458, 'cone': 0.1023},
    'IterativePFN': {'sphere': 0.0135, 'cube': 0.0349, 'cone': 0.0336},
    'StraightPCF':  {'sphere': 0.0015, 'cube': 0.1437, 'cone': 0.0877},
}

for model_name in MODELS:
    cds, p2ps, grs = [], [], []
    for shape in SHAPES:
        r      = all_results[model_name][shape]
        gr_old = GR_OLD.get(model_name, {}).get(shape, float('nan'))
        print(f'{model_name:<14} | {shape:<6} | {r["CD"]:>10.6f} | {r["P2P"]:>10.6f} | {gr_old:>10.4f} | {r["GR"]:>10.4f}')
        cds.append(r["CD"]); p2ps.append(r["P2P"]); grs.append(r["GR"])
    mean_cd, mean_p2p, mean_gr = np.mean(cds), np.mean(p2ps), np.mean(grs)
    print(f'{"  → Mean":<14} | {"":6} | {mean_cd:>10.6f} | {mean_p2p:>10.6f} | {"":>10} | {mean_gr:>10.4f}')
    print()

# ============================================================
# 保存结果
# ============================================================
output = {
    'timestamp':    datetime.now().isoformat(),
    'note':         'GR修复版: torch.cdist线性距离, dim=0召回方向, THRESHOLD=2*noise_level=0.04',
    'threshold':    THRESHOLD,
    'noise_level':  NOISE_LEVEL,
    'checkpoints':  'mixed_best',
    'results':      {
        model_name: {
            shape: {k: v for k, v in r.items()}
            for shape, r in shape_results.items()
        }
        for model_name, shape_results in all_results.items()
    }
}

out_path = os.path.join(RESULTS_DIR, 'gr_fixed_evaluation.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f'结果已保存: {out_path}')
