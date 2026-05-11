"""调试 MLS 在 sphere 上 GR 低的问题"""
import os
os.environ['TORCH_DISABLE_CUDA_INIT'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USERNAME'] = 'User'

import sys
sys.path.insert(0, r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising')

import numpy as np
import torch
from data.synthetic_data_generator import load_dataset
from models.traditional_methods import mls_denoise, gaussian_filter

data = load_dataset(r'data\synthetic_dataset.pkl')
noisy = data['noisy']['sphere'][0.02][0]
clean = data['clean']['sphere'][0]['points']

print('=== MLS vs Gaussian debug on sphere ===')
print(f'noisy shape: {noisy.shape}')
print(f'clean shape: {clean.shape}')

mls_out = mls_denoise(noisy, k=16, sigma=0.02)
gauss_out = gaussian_filter(noisy, k=16, sigma=0.02)

pts_pred = torch.from_numpy(mls_out)
pts_clean = torch.from_numpy(clean)
d = torch.sum((pts_pred.unsqueeze(1) - pts_clean.unsqueeze(0)) ** 2, dim=2)
cd_mls = (torch.mean(torch.min(d, dim=1)[0]) + torch.mean(torch.min(d, dim=0)[0])).item()

dist_matrix = torch.cdist(pts_pred, pts_clean)
min_dists = torch.min(dist_matrix, dim=0)[0]
gr_mls = (min_dists <= 0.04).float().mean().item()

print(f'\nMLS   -> CD={cd_mls:.6f}, GR={gr_mls:.4f}')
print(f'MLS输出范围: min={mls_out.min():.4f}, max={mls_out.max():.4f}')
print(f'clean范围:    min={clean.min():.4f}, max={clean.max():.4f}')

radii_mls = np.sqrt(np.sum(mls_out ** 2, axis=1))
radii_clean = np.sqrt(np.sum(clean ** 2, axis=1))
print(f'\nMLS输出点半径: mean={radii_mls.mean():.4f}, std={radii_mls.std():.4f}')
print(f'clean点半径:    mean={radii_clean.mean():.4f}, std={radii_clean.std():.4f}')

print(f'\nMLS最近clean点距离: mean={min_dists.mean():.6f}, median={min_dists.median():.6f}, max={min_dists.max():.6f}')
print(f'距离<0.04的比例(GR): {(min_dists <= 0.04).float().mean().item():.4f}')
print(f'距离<0.10的比例: {(min_dists <= 0.10).float().mean().item():.4f}')

# 检查 MLS 是否把点移向了球心（内缩）
print(f'\n诊断: MLS 输出半径均值={radii_mls.mean():.4f}, clean半径均值={radii_clean.mean():.4f}')
if radi_mls.mean() < radi_clean.mean() - 0.01:
    print('  => MLS 输出点向内收缩了! 这是BUG!')
else:
    print('  => MLS 输出半径正常')
