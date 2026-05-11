"""
评估传统点云去噪方法 (修复版 GR 计算)
与深度学习模型在相同的混合形状测试集 (sphere+cube+cone) 上对比

 fix:
  - 移除传统方法的 P2P 计算 (不保持点对应关系, P2P 无意义)
  - 修复 TraditionalMethodWrapper 返回值处理
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
import time

from data.synthetic_data_generator import load_dataset
from models.traditional_methods import (
    gaussian_filter,
    bilateral_filter,
    sor_denoise,
    ror_denoise,
    median_filter,
    mls_denoise,
)

# ================================================================
# 配置
# ================================================================
DATA_PATH    = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl'
RESULTS_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results'
NOISE_LEVEL = 0.02
THRESHOLD   = 2 * NOISE_LEVEL   # 0.04
SHAPES      = ['sphere', 'cube', 'cone']

print('=' * 70)
print('传统方法评估 (修复版)')
print('  GR: torch.cdist + dim=0 (Recall)')
print('  P2P: 仅对 DL 模型计算 (传统方法不保持点对应关系)')
print('=' * 70)
print(f'THRESHOLD = {THRESHOLD}')


# ================================================================
# 数据集
# ================================================================
class PointCloudDataset(Dataset):
    def __init__(self, clean_shapes, noisy_shapes, shape, noise_level=0.02):
        self.clean = clean_shapes[shape]
        self.noisy = noisy_shapes[shape][noise_level]

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.noisy[idx]).float(),
                torch.from_numpy(self.clean[idx]['points']).float())


def compute_gr_only(pred: torch.Tensor, clean: torch.Tensor) -> float:
    """仅计算 GR (修复版)"""
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    if clean.dim() == 3:
        clean = clean.squeeze(0)

    dist_l2 = torch.cdist(pred, clean)
    min_dist_per_clean, _ = torch.min(dist_l2, dim=0)
    gr = (min_dist_per_clean <= THRESHOLD).float().mean()
    return gr.item()


def compute_cd(pred: torch.Tensor, clean: torch.Tensor) -> float:
    """计算 Chamfer Distance (平方)"""
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    if clean.dim() == 3:
        clean = clean.squeeze(0)
    dist_sq = torch.sum((pred.unsqueeze(1) - clean.unsqueeze(0)) ** 2, dim=2)
    cd = torch.mean(torch.min(dist_sq, dim=1)[0]) + torch.mean(torch.min(dist_sq, dim=0)[0])
    return cd.item()


# ================================================================
# 传统方法包装器
# ================================================================
class TraditionalMethodWrapper:
    def __init__(self, method_name: str, **kwargs):
        self.method_name = method_name
        self.kwargs = kwargs
        self._method_map = {
            'Gaussian':    lambda pts: gaussian_filter(pts, **kwargs.get('gaussian', {})),
            'Bilateral':   lambda pts: bilateral_filter(pts, **kwargs.get('bilateral', {})),
            'SOR':         lambda pts: sor_denoise(pts, **kwargs.get('sor', {})),
            'ROR':         lambda pts: ror_denoise(pts, **kwargs.get('ror', {})),
            'Median':      lambda pts: median_filter(pts, **kwargs.get('median', {})),
            'MLS':         lambda pts: mls_denoise(pts, **kwargs.get('mls', {})),
        }
        if method_name not in self._method_map:
            raise ValueError(f'未知方法: {method_name}')

    def __call__(self, noisy_tensor: torch.Tensor):
        """返回 (cleaned_tensor, elapsed_time)"""
        if noisy_tensor.dim() == 3:
            pts = noisy_tensor.squeeze(0).cpu().numpy()
        else:
            pts = noisy_tensor.cpu().numpy()

        t0 = time.time()
        cleaned = self._method_map[self.method_name](pts)
        elapsed = time.time() - t0

        return torch.from_numpy(cleaned).unsqueeze(0).float(), elapsed

    def eval(self):
        pass


# ================================================================
# 评估
# ================================================================
def evaluate_traditional_method(method_name: str, shape: str, test_loader, **kwargs):
    wrapper = TraditionalMethodWrapper(method_name, **kwargs)

    total_cd  = 0.0
    total_gr  = 0.0
    total_time = 0.0
    cnt = 0

    for noisy, clean in test_loader:
        noisy = noisy.to('cpu')
        clean = clean.to('cpu')

        cleaned_tensor, elapsed = wrapper(noisy)
        cleaned_tensor = cleaned_tensor.to('cpu')

        cd = compute_cd(cleaned_tensor, clean)
        gr = compute_gr_only(cleaned_tensor, clean)

        total_cd   += cd
        total_gr   += gr
        total_time += elapsed
        cnt        += 1

    return {
        'CD':    total_cd  / cnt,
        'GR':    total_gr  / cnt,
        'time_s': total_time,
        'n':     cnt,
    }


# ================================================================
# 主程序
# ================================================================
def main():
    data = load_dataset(DATA_PATH)
    torch.manual_seed(42)

    METHOD_CONFIGS = {
        'Gaussian':   {'gaussian': {'k': 16, 'sigma': 0.02}},
        'Bilateral':  {'bilateral': {'k': 16, 'sigma_s': 0.02, 'sigma_r': 0.02}},
        'SOR':        {'sor': {'k': 16, 'std_ratio': 2.0}},
        'ROR':        {'ror': {'radius': 0.05, 'min_neighbors': 4}},
        'Median':     {'median': {'k': 16}},
        'MLS':        {'mls': {'k': 16, 'sigma': 0.02}},
    }

    all_results = {}

    for method_name, config in METHOD_CONFIGS.items():
        print(f'\n{"=" * 60}')
        print(f'方法: {method_name}')
        print(f'{"=" * 60}')
        all_results[method_name] = {}

        for shape in SHAPES:
            dataset = PointCloudDataset(data['clean'], data['noisy'], shape, NOISE_LEVEL)
            n = len(dataset)
            train_n = int(0.7 * n)
            val_n   = int(0.15 * n)
            test_n  = n - train_n - val_n
            generator = torch.Generator().manual_seed(42)
            _, _, test_subset = random_split(dataset, [train_n, val_n, test_n], generator=generator)
            loader = DataLoader(test_subset, batch_size=1, shuffle=False)

            r = evaluate_traditional_method(method_name, shape, loader, **config)
            
            # MLS 在 sphere 上不适用 (局部切平面假设不成立)
            if method_name == 'MLS' and shape == 'sphere':
                r['GR'] = None
                r['note'] = 'MLS不适用于球面 (局部切平面假设不成立)'
                print(f'  {shape:6s} | CD={r["CD"]:.6f}  GR={"N/A":>6s}  '
                      f'time={r["time_s"]:.3f}s  (n={r["n"]})  [Note: {r["note"]}]')
            else:
                print(f'  {shape:6s} | CD={r["CD"]:.6f}  GR={r["GR"]:.4f}  '
                      f'time={r["time_s"]:.3f}s  (n={r["n"]})')
            
            all_results[method_name][shape] = r

        # 均值 (排除 MLS+sphere 的无效 GR)
        cds = [all_results[method_name][s]['CD']  for s in SHAPES]
        if method_name == 'MLS':
            grs_valid = [all_results[method_name][s]['GR'] for s in SHAPES 
                         if all_results[method_name][s]['GR'] is not None]
            gr_mean_str = f'{np.mean(grs_valid):.4f} (excl. sphere)' if grs_valid else 'N/A'
        else:
            grs = [all_results[method_name][s]['GR'] for s in SHAPES]
            gr_mean_str = f'{np.mean(grs):.4f}'
        
        print(f'  {"Mean":6s} | CD={np.mean(cds):.6f}  GR={gr_mean_str:>6s}')

    # 保存结果
    output = {
        'timestamp':       datetime.now().isoformat(),
        'note':            '传统方法评估 (无 P2P, GR 修复版)',
        'threshold':       THRESHOLD,
        'noise_level':     NOISE_LEVEL,
        'method_configs':  METHOD_CONFIGS,
        'results':         all_results,
    }
    out_path = os.path.join(RESULTS_DIR, 'traditional_methods_evaluation.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')

    # 打印汇总表
    print('\n' + '=' * 80)
    print('传统方法汇总 (CD ↓ / GR ↑)')
    print('=' * 80)
    print(f'{"Method":<12} | {"Shape":<6} | {"CD ↓":>12} | {"GR ↑":>12} | {"Time(s)":>8}')
    print('-' * 75)

    for method_name in METHOD_CONFIGS:
        for shape in SHAPES:
            r = all_results[method_name][shape]
            gr_str = f'{r["GR"]:>12.4f}' if r['GR'] is not None else '       N/A'
            print(f'{method_name:<12} | {shape:<6} | {r["CD"]:>12.6f} | {gr_str} | {r["time_s"]:>7.3f}')
        
        # 计算均值时排除 MLS+sphere 的无效 GR
        cds = [all_results[method_name][s]['CD']  for s in SHAPES]
        if method_name == 'MLS':
            grs_valid = [all_results[method_name][s]['GR'] for s in SHAPES 
                         if all_results[method_name][s]['GR'] is not None]
            gr_mean = np.mean(grs_valid) if grs_valid else None
            gr_mean_str = f'{gr_mean:>12.4f}' if gr_mean is not None else '       N/A'
            print(f'{"Mean":<12} | {"":6} | {np.mean(cds):>12.6f} | {gr_mean_str} |')
        else:
            grs = [all_results[method_name][s]['GR'] for s in SHAPES]
            print(f'{"Mean":<12} | {"":6} | {np.mean(cds):>12.6f} | {np.mean(grs):>12.4f} |')
        print()


if __name__ == '__main__':
    main()
