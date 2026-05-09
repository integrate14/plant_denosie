"""
评估所有模型并更新结果
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

print('='*60)
print('Evaluating All Models')
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

# 数据集划分: 70% 训练集, 15% 验证集, 15% 测试集
dataset = PointCloudDataset(data['clean'], data['noisy'], shape='sphere', noise_level=0.02)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_subset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
print(f'Data: Train {len(train_subset)}, Val {len(val_subset)}, Test {len(test_subset)}')

NOISE_LEVEL = 0.02
THRESHOLD = 2 * NOISE_LEVEL

def compute_metrics(pred, clean):
    """计算 CD, P2P, GR 指标"""
    # 确保是 2D 张量 (N, 3)
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    if clean.dim() == 3:
        clean = clean.squeeze(0)

    # 确保是 2D
    assert pred.dim() == 2 and clean.dim() == 2, f"Expected 2D, got {pred.dim()}D and {clean.dim()}D"
    assert pred.shape[1] == 3 and clean.shape[1] == 3, f"Expected 3 channels, got {pred.shape[1]} and {clean.shape[1]}"

    N = pred.shape[0]

    # Chamfer Distance
    # dist[i,j] = ||pred[i] - clean[j]||^2
    dist = torch.sum((pred.unsqueeze(1) - clean.unsqueeze(0)) ** 2, dim=2)  # (N, N)
    cd1 = torch.mean(torch.min(dist, dim=1)[0])  # mean over pred points
    cd2 = torch.mean(torch.min(dist, dim=0)[0])  # mean over clean points
    cd = cd1 + cd2

    # Point-to-Point Distance
    p2p = torch.mean(torch.sqrt(torch.sum((pred - clean) ** 2, dim=-1) + 1e-8))

    # Geometric Recall (GR)
    min_dist, _ = torch.min(dist, dim=1)  # (N,) - 每个pred点到最近clean点的距离
    within_threshold = torch.sum(min_dist <= THRESHOLD).float()
    gr = within_threshold / N

    return cd.item(), p2p.item(), gr.item()

def evaluate_model_on_loader(model, loader):
    """评估单个模型在指定数据加载器上"""
    model.eval()
    total_cd = 0.0
    total_p2p = 0.0
    total_gr = 0.0
    n = 0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            cleaned = model(noisy)
            if isinstance(cleaned, tuple):
                cleaned = cleaned[0]

            # 处理批次维度 (B, N, 3) -> (N, 3)
            if cleaned.dim() == 3:
                cleaned = cleaned.squeeze(0)
            if clean.dim() == 3:
                clean = clean.squeeze(0)

            # 确保维度正确
            assert cleaned.shape == clean.shape, f"Shape mismatch: {cleaned.shape} vs {clean.shape}"

            cd, p2p, gr = compute_metrics(cleaned, clean)
            total_cd += cd
            total_p2p += p2p
            total_gr += gr
            n += 1

    return {
        'chamfer': total_cd / n,
        'p2p': total_p2p / n,
        'geometric_recall': total_gr / n
    }

# 评估所有模型 - 分别在验证集和测试集上评估
results = {}
val_results = {}
test_results = {}

# PointFilter
print('\nEvaluating PointFilter...')
model_pf = create_pointfilter_model(num_points=2048)
ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'PointFilter_best.pth'), map_location=device, weights_only=False)
model_pf.load_state_dict(ckpt['model_state_dict'])
val_results['PointFilter'] = evaluate_model_on_loader(model_pf, val_loader)
test_results['PointFilter'] = evaluate_model_on_loader(model_pf, test_loader)
print(f"  Val: CD={val_results['PointFilter']['chamfer']:.6f}, P2P={val_results['PointFilter']['p2p']:.6f}, GR={val_results['PointFilter']['geometric_recall']:.4f}")
print(f"  Test: CD={test_results['PointFilter']['chamfer']:.6f}, P2P={test_results['PointFilter']['p2p']:.6f}, GR={test_results['PointFilter']['geometric_recall']:.4f}")

# IterativePFN
print('Evaluating IterativePFN...')
model_ipfn = create_iterativepfn_improved_model(num_points=2048, num_iterations=3, feature_dim=256, hidden_dim=128)
ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'IterativePFN_best.pth'), map_location=device, weights_only=False)
model_ipfn.load_state_dict(ckpt['model_state_dict'])
val_results['IterativePFN'] = evaluate_model_on_loader(model_ipfn, val_loader)
test_results['IterativePFN'] = evaluate_model_on_loader(model_ipfn, test_loader)
print(f"  Val: CD={val_results['IterativePFN']['chamfer']:.6f}, P2P={val_results['IterativePFN']['p2p']:.6f}, GR={val_results['IterativePFN']['geometric_recall']:.4f}")
print(f"  Test: CD={test_results['IterativePFN']['chamfer']:.6f}, P2P={test_results['IterativePFN']['p2p']:.6f}, GR={test_results['IterativePFN']['geometric_recall']:.4f}")

# StraightPCF
print('Evaluating StraightPCF...')
model_spcf = StraightPCFImproved(num_points=2048, feature_dim=256, hidden_dim=128, num_iterations=3, use_dgcnn=False)
ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'StraightPCF_best.pth'), map_location=device, weights_only=False)
model_spcf.load_state_dict(ckpt['model_state_dict'])
val_results['StraightPCF'] = evaluate_model_on_loader(model_spcf, val_loader)
test_results['StraightPCF'] = evaluate_model_on_loader(model_spcf, test_loader)
print(f"  Val: CD={val_results['StraightPCF']['chamfer']:.6f}, P2P={val_results['StraightPCF']['p2p']:.6f}, GR={val_results['StraightPCF']['geometric_recall']:.4f}")
print(f"  Test: CD={test_results['StraightPCF']['chamfer']:.6f}, P2P={test_results['StraightPCF']['p2p']:.6f}, GR={test_results['StraightPCF']['geometric_recall']:.4f}")

# 传统方法
print('\nEvaluating Traditional Methods...')

def bilateral_filter(points, sigma_s=0.1, sigma_r=0.1):
    """双边滤波器"""
    n = points.shape[0]
    filtered = torch.zeros_like(points)
    for i in range(n):
        p = points[i]
        # 简化的双边滤波
        weights = torch.zeros(n)
        for j in range(n):
            dist = torch.norm(p - points[j])
            weights[j] = torch.exp(-dist**2 / (2 * sigma_s**2))
        weights = weights / weights.sum()
        filtered[i] = (points * weights.unsqueeze(1)).sum(dim=0)
    return filtered

def laplacian_smooth(points, k=6, iterations=3):
    """拉普拉斯平滑"""
    filtered = points.clone()
    for _ in range(iterations):
        new_filtered = torch.zeros_like(filtered)
        for i in range(len(filtered)):
            p = filtered[i]
            # 找 k 个最近邻
            dists = torch.norm(filtered - p, dim=1)
            _, idx = torch.topk(dists, min(k+1, len(dists)))
            neighbors = filtered[idx[1:]]  # 排除自己
            new_filtered[i] = neighbors.mean(dim=0)
        filtered = new_filtered
    return filtered

def evaluate_traditional_on_loader(filter_fn, loader):
    """评估传统方法在指定数据加载器上"""
    total_cd, total_p2p, total_gr = 0, 0, 0
    n = len(loader)
    for noisy, clean in loader:
        noisy, clean = noisy.squeeze(0).to(device), clean.squeeze(0).to(device)
        denoised = filter_fn(noisy.unsqueeze(0)).squeeze(0)
        cd, p2p, gr = compute_metrics(denoised, clean)
        total_cd += cd
        total_p2p += p2p
        total_gr += gr
    return {
        'chamfer': total_cd / n,
        'p2p': total_p2p / n,
        'geometric_recall': total_gr / n
    }

# 评估双边滤波
val_results['Bilateral'] = evaluate_traditional_on_loader(bilateral_filter, val_loader)
test_results['Bilateral'] = evaluate_traditional_on_loader(bilateral_filter, test_loader)
print(f"  Bilateral - Val: CD={val_results['Bilateral']['chamfer']:.6f}, P2P={val_results['Bilateral']['p2p']:.6f}, GR={val_results['Bilateral']['geometric_recall']:.4f}")
print(f"  Bilateral - Test: CD={test_results['Bilateral']['chamfer']:.6f}, P2P={test_results['Bilateral']['p2p']:.6f}, GR={test_results['Bilateral']['geometric_recall']:.4f}")

# 评估拉普拉斯
val_results['Laplacian'] = evaluate_traditional_on_loader(laplacian_smooth, val_loader)
test_results['Laplacian'] = evaluate_traditional_on_loader(laplacian_smooth, test_loader)
print(f"  Laplacian - Val: CD={val_results['Laplacian']['chamfer']:.6f}, P2P={val_results['Laplacian']['p2p']:.6f}, GR={val_results['Laplacian']['geometric_recall']:.4f}")
print(f"  Laplacian - Test: CD={test_results['Laplacian']['chamfer']:.6f}, P2P={test_results['Laplacian']['p2p']:.6f}, GR={test_results['Laplacian']['geometric_recall']:.4f}")

# 合并结果（使用测试集结果作为最终结果）
results = test_results

# 保存结果
output = {
    'timestamp': datetime.now().isoformat(),
    'dataset': 'synthetic (sphere)',
    'noise_level': NOISE_LEVEL,
    'threshold': THRESHOLD,
    'data_split': {
        'train': '70%',
        'validation': '15%',
        'test': '15%'
    },
    'validation_results': val_results,
    'test_results': test_results,
    'models': results  # 兼容旧格式，指向测试集结果
}

results_path = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results\experiment_results.json'
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f'\nResults saved to: {results_path}')

# 打印汇总表 - 分别显示验证集和测试集结果
print('\n' + '='*80)
print('FINAL RESULTS SUMMARY')
print('='*80)
print('\nValidation Set Results:')
print(f"{'Model':<15} {'CD':>12} {'P2P':>12} {'GR':>12}")
print('-'*55)
for name, metrics in val_results.items():
    print(f"{name:<15} {metrics['chamfer']:>12.6f} {metrics['p2p']:>12.6f} {metrics['geometric_recall']:>12.4f}")

print('\nTest Set Results:')
print(f"{'Model':<15} {'CD':>12} {'P2P':>12} {'GR':>12}")
print('-'*55)
for name, metrics in test_results.items():
    print(f"{name:<15} {metrics['chamfer']:>12.6f} {metrics['p2p']:>12.6f} {metrics['geometric_recall']:>12.4f}")
