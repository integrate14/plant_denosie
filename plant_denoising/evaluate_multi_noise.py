"""
多噪声级别评估 - 评估三个去噪模型在不同噪声级别下的性能
数据集预生成4个噪声级别: 0.01, 0.02, 0.05, 0.1
模型在 noise_level=0.02 上训练，测试泛化能力
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
noise_levels = sorted(data['metadata']['noise_levels'])
print(f"可用噪声级别: {noise_levels}")


class MultiNoiseDataset(Dataset):
    """支持指定噪声级别的点云数据集"""
    def __init__(self, clean_shapes, noisy_shapes, shape='sphere', noise_level=0.02):
        self.clean = clean_shapes[shape]
        self.noisy = noisy_shapes[shape][noise_level]
    def __len__(self):
        return len(self.clean)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.noisy[idx]).float(),
                torch.from_numpy(self.clean[idx]['points']).float())


def load_checkpoint(model, name):
    """加载模型检查点"""
    path = os.path.join(CHECKPOINT_DIR, f'{name}_best.pth')
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'  {name}: loaded from {path}')
    else:
        print(f'  WARNING: {path} not found!')
    return model


def evaluate_model(model, loader, model_type='pf'):
    """评估单个模型的CD/P2P/GR指标"""
    total_cd = 0
    total_p2p = 0
    total_gr = 0
    n = 0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)

            output = model(noisy)
            cleaned = output[0] if isinstance(output, tuple) else output

            # Chamfer Distance
            cd = model.chamfer_distance(cleaned, clean)

            # P2P Distance (point-to-point Euclidean distance)
            p2p = torch.mean(torch.sqrt(torch.sum((cleaned - clean) ** 2, dim=-1) + 1e-8))

            # Geometric Recall (tau = 0.01)
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


# 使用固定随机种子保证每个噪声级别的测试集划分一致
def get_test_loader(noise_level):
    """获取指定噪声级别的测试集DataLoader（固定划分）"""
    dataset = MultiNoiseDataset(data['clean'], data['noisy'], shape='sphere', noise_level=noise_level)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # 固定generator确保每次划分一致
    gen = torch.Generator().manual_seed(42)
    _, _, test_subset = random_split(dataset, [train_size, val_size, test_size], generator=gen)

    return DataLoader(test_subset, batch_size=4, shuffle=False), len(test_subset)


# ============================================================
# 加载三个模型
# ============================================================
print('\n' + '='*60)
print('加载模型')
print('='*60)

model_pf = create_pointfilter_model(num_points=2048)
model_pf = load_checkpoint(model_pf, 'PointFilter')
model_pf = model_pf.to(device).eval()

model_ipfn = create_iterativepfn_improved_model(num_points=2048, num_iterations=3, feature_dim=256, hidden_dim=128)
model_ipfn = load_checkpoint(model_ipfn, 'IterativePFN')
model_ipfn = model_ipfn.to(device).eval()

model_pcf = create_straightpcf_improved_model(num_points=2048, feature_dim=256, hidden_dim=128)
model_pcf = load_checkpoint(model_pcf, 'StraightPCF')
model_pcf = model_pcf.to(device).eval()

models = {
    'PointFilter': (model_pf, 'pf'),
    'IterativePFN': (model_ipfn, 'ipfn'),
    'StraightPCF': (model_pcf, 'pcf')
}

# ============================================================
# 对每个噪声级别进行评估
# ============================================================
print('\n' + '='*60)
print('多噪声级别评估')
print('='*60)

all_results = {}
test_set_size = None

for nl in noise_levels:
    print(f'\n--- 噪声级别: {nl} ---')
    loader, tsize = get_test_loader(nl)
    if test_set_size is None:
        test_set_size = tsize
    print(f'  测试集样本数: {tsize}')

    all_results[nl] = {}
    for name, (model_obj, mtype) in models.items():
        print(f'  评估 {name}...', end=' ', flush=True)
        result = evaluate_model(model_obj, loader, mtype)
        all_results[nl][name] = result
        print(f"CD={result['Chamfer Distance']:.6f}, P2P={result['P2P Distance']:.6f}")

# ============================================================
# 汇总结果
# ============================================================
print('\n' + '='*70)
print(f'多噪声级别评估汇总 (测试集 n={test_set_size})')
print('='*70)
print(f'{"噪声级别":^10}', end='')
for name in models:
    print(f' | {"CD ↓":>10}', end='')
print()
print('-'*70)

for nl in noise_levels:
    print(f'{nl:^10.3f}', end='')
    for name in models:
        cd = all_results[nl][name]['Chamfer Distance']
        print(f' | {cd:>10.6f}', end='')
    print()

print()
print(f'{"噪声级别":^10}', end='')
for name in models:
    print(f' | {"P2P ↓":>10}', end='')
print()
print('-'*70)

for nl in noise_levels:
    print(f'{nl:^10.3f}', end='')
    for name in models:
        p2p = all_results[nl][name]['P2P Distance']
        print(f' | {p2p:>10.6f}', end='')
    print()

print()
print(f'{"噪声级别":^10}', end='')
for name in models:
    print(f' | {"GR ↑":>10}', end='')
print()
print('-'*70)

for nl in noise_levels:
    print(f'{nl:^10.3f}', end='')
    for name in models:
        gr = all_results[nl][name]['Geometric Recall']
        print(f' | {gr:>10.4f}', end='')
    print()

# ============================================================
# 计算额外统计：相对于输入噪声的改善率
# ============================================================
print('\n--- 相对于输入噪声的改善率 (vs Noisy Input) ---')

# 先计算各噪声级别下原始噪声点云的CD和P2P
noisy_baselines = {}
for nl in noise_levels:
    loader, _ = get_test_loader(nl)
    total_cd_noisy = 0
    total_p2p_noisy = 0
    n = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            # 用简单的chamfer distance计算
            dists_xy = torch.cdist(noisy, clean)  # (B, N, M)
            dists_yx = torch.cdist(clean, noisy)   # (B, M, N)
            cd1, _ = torch.min(dists_xy, dim=2)    # (B, N)
            cd2, _ = torch.min(dists_yx, dim=2)    # (B, M)
            cd_val = (cd1.mean(dim=1).mean() + cd2.mean(dim=1).mean()).item() / 2
            p2p_val = torch.mean(torch.sqrt(torch.sum((noisy - clean) ** 2, dim=-1) + 1e-8)).item()
            total_cd_noisy += cd_val
            total_p2p_noisy += p2p_val
            n += 1
    noisy_baselines[nl] = {
        'CD': total_cd_noisy / n,
        'P2P': total_p2p_noisy / n
    }

for nl in noise_levels:
    base_cd = noisy_baselines[nl]['CD']
    base_p2p = noisy_baselines[nl]['P2P']
    print(f'\n  Noise={nl}: 输入CD={base_cd:.6f}, 输入P2P={base_p2p:.6f}')
    for name in models:
        res = all_results[nl][name]
        cd_improve = (base_cd - res['Chamfer Distance']) / base_cd * 100
        p2p_improve = (base_p2p - res['P2P Distance']) / base_p2p * 100
        print(f'    {name}: CD改善 {cd_improve:+.1f}%, P2P改善 {p2p_improve:+.1f}%')

# 保存完整结果
output = {
    'timestamp': datetime.now().isoformat(),
    'test_set_size': test_set_size,
    'noise_levels': noise_levels,
    'noisy_baselines': noisy_baselines,
    'results': all_results
}

output_path = os.path.join(r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising', 'results', 'multi_noise_evaluation.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f'\n\n结果已保存到: {output_path}')

# 打印最终JSON供可视化使用
print('\n=== VISUALIZATION_DATA ===')
viz_data = {
    'noise_levels': noise_levels,
    'models': list(models.keys()),
    'metrics': {}
}
for metric in ['Chamfer Distance', 'P2P Distance', 'Geometric Recall']:
    viz_data['metrics'][metric] = {}
    for name in models:
        viz_data['metrics'][metric][name] = [all_results[nl][name][metric] for nl in noise_levels]

# 也保存noisy baseline用于对比
viz_data['noisy_baseline_CD'] = [noisy_baselines[nl]['CD'] for nl in noise_levels]
viz_data['noisy_baseline_P2P'] = [noisy_baselines[nl]['P2P'] for nl in noise_levels]

print(json.dumps(viz_data, indent=2))
