"""
按形状分析模型性能差异 - 评估 PointFilter/IterativePFN/StraightPCF
在 sphere / cube / cone 三种几何形状上的表现差异

核心问题：
1. 哪个模型在哪种形状上表现最好？
2. 模型对几何形状的敏感度如何？
3. 各模型的优势/劣势形状是什么？
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
shapes = data['metadata']['shapes']  # ['sphere', 'cube', 'cone']
noise_levels = sorted(data['metadata']['noise_levels'])
print(f"形状类型: {shapes}")
print(f"噪声级别: {noise_levels}")


class ShapeDataset(Dataset):
    """按指定形状和噪声级别构建数据集"""
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
        print(f'  已加载: {name}')
    else:
        print(f'  警告: {path} 不存在!')
    return model


def evaluate_model(model, loader):
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

            # P2P Distance
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


def evaluate_noisy_baseline(loader):
    """计算原始噪声点云的基线指标"""
    total_cd = 0
    total_p2p = 0
    n = 0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            dists_xy = torch.cdist(noisy, clean)
            dists_yx = torch.cdist(clean, noisy)
            cd1, _ = torch.min(dists_xy, dim=2)
            cd2, _ = torch.min(dists_yx, dim=2)
            cd_val = (cd1.mean(dim=1).mean() + cd2.mean(dim=1).mean()).item() / 2
            p2p_val = torch.mean(torch.sqrt(torch.sum((noisy - clean) ** 2, dim=-1) + 1e-8)).item()
            total_cd += cd_val
            total_p2p += p2p_val
            n += 1

    return {'CD': total_cd / n, 'P2P': total_p2p / n}


# ============================================================
# 加载三个模型（使用训练噪声 0.02 的检查点）
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

# 使用训练时的噪声级别 0.02 进行主评估
TRAIN_NOISE = 0.02

# ============================================================
# 核心分析：每个形状 x 每个模型 的详细评估
# ============================================================
results_by_shape = {}
noisy_baselines_by_shape = {}

print('\n' + '='*70)
print(f'按形状评估 (训练噪声级别: {TRAIN_NOISE})')
print('='*70)

for shape in shapes:
    print(f'\n{"="*50}')
    print(f'形状: {shape}')
    print(f'{"="*50}')

    dataset = ShapeDataset(data['clean'], data['noisy'], shape=shape, noise_level=TRAIN_NOISE)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    gen = torch.Generator().manual_seed(42)
    _, _, test_subset = random_split(dataset, [train_size, val_size, test_size], generator=gen)
    loader = DataLoader(test_subset, batch_size=4, shuffle=False)

    print(f'  测试集样本数: {test_size}')

    # 计算噪声基线
    baseline = evaluate_noisy_baseline(loader)
    noisy_baselines_by_shape[shape] = baseline
    print(f'  输入噪声基线: CD={baseline["CD"]:.6f}, P2P={baseline["P2P"]:.6f}')

    results_by_shape[shape] = {'baseline': baseline, 'models': {}}

    for name, (model_obj, mtype) in models.items():
        result = evaluate_model(model_obj, loader)
        results_by_shape[shape]['models'][name] = result
        cd_imp = (baseline['CD'] - result['Chamfer Distance']) / baseline['CD'] * 100
        p2p_imp = (baseline['P2P'] - result['P2P Distance']) / baseline['P2P'] * 100
        print(f'  {name:16s}: CD={result["Chamfer Distance"]:.6f} ({cd_imp:+.1f}%), '
              f'P2P={result["P2P Distance"]:.6f} ({p2p_imp:+.1f}%), '
              f'GR={result["Geometric Recall"]:.4f}')


# ============================================================
# 跨噪声级别的形状分析
# ============================================================
print('\n\n' + '='*70)
print('跨噪声级别的形状 x 模型 完整矩阵')
print('='*70)

full_matrix = {}  # {shape: {noise_level: {model: metrics}}}

for shape in shapes:
    full_matrix[shape] = {}

    dataset_base = ShapeDataset(data['clean'], data['noisy'], shape=shape, noise_level=TRAIN_NOISE)
    total_size = len(dataset_base)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    gen = torch.Generator().manual_seed(42)
    _, _, test_subset = random_split(dataset_base, [train_size, val_size, test_size], generator=gen)

    for nl in noise_levels:
        # 对每个噪声级别使用相同的测试索引
        ds_nl = ShapeDataset(data['clean'], data['noisy'], shape=shape, noise_level=nl)
        # 用相同索引构造测试集
        test_indices = test_subset.indices
        from torch.utils.data import Subset
        test_ds = Subset(ds_nl, test_indices)
        loader = DataLoader(test_ds, batch_size=4, shuffle=False)

        full_matrix[shape][nl] = {}
        for name, (model_obj, mtype) in models.items():
            result = evaluate_model(model_obj, loader)
            full_matrix[shape][nl][name] = result


# ============================================================
# 打印详细对比表格
# ============================================================

def print_shape_comparison_table(matrix, metric_key, noise_level):
    """打印指定噪声级别下各形状的模型对比表"""
    print(f'\n--- {metric_key} @ Noise={noise_level} ---')
    header = f'{"Model":16s}'
    for shape in shapes:
        header += f' | {shape:>12s}'
    print(header)
    print('-' * len(header))

    for name in models:
        row = f'{name:16s}'
        for shape in shapes:
            val = matrix[shape][noise_level][name][metric_key]
            row += f' | {val:>12.6f}'
        print(row)

    # 找出每列最优
    row_best = f'{"** BEST **":16s}'
    for shape in shapes:
        best_name = min(models.keys(), key=lambda m: matrix[shape][noise_level][m][metric_key])
        best_val = matrix[shape][noise_level][best_name][metric_key]
        row_best += f' | {best_name:>12s}'
    print(row_best)


for nl in noise_levels:
    for metric in ['Chamfer Distance', 'P2P Distance']:
        print_shape_comparison_table(full_matrix, metric, nl)


# ============================================================
# 形状敏感性分析：各模型在不同形状上的波动程度
# ============================================================
print('\n\n' + '='*70)
print('形状敏感性分析 (Noise=0.02, 训练噪声)')
print('='*70)
print('(值越小表示对形状变化越稳定/不敏感)')

for name in models:
    cds = [results_by_shape[shape]['models'][name]['Chamfer Distance'] for shape in shapes]
    p2ps = [results_by_shape[shape]['models'][name]['P2P Distance'] for shape in shapes]

    cd_std = np.std(cds)
    cd_range = max(cds) - min(cds)
    p2p_std = np.std(p2ps)
    p2p_range = max(p2ps) - min(p2ps)

    print(f'\n{name}:')
    print(f'  CD: mean={np.mean(cds):.6f}, std={cd_std:.6f}, range=[{min(cds):.6f}, {max(cds):.6f}]')
    print(f'     各形状: ', end='')
    for s, v in zip(shapes, cds):
        marker = ' <-- 最优' if v == min(cds) else (' <-- 最差' if v == max(cds) else '')
        print(f'{s}={v:.6f}{marker}', end='  ')
    print()
    print(f'  P2P: mean={np.mean(p2ps):.6f}, std={p2p_std:.6f}, range=[{min(p2ps):.6f}, {max(p2ps):.6f}]')
    print(f'      各形状: ', end='')
    for s, v in zip(shapes, p2ps):
        marker = ' <-- 最优' if v == min(p2ps) else (' <-- 最差' if v == max(p2ps) else '')
        print(f'{s}={v:.6f}{marker}', end='  ')
    print()


# ============================================================
# 综合排名：每种形状上的模型排名
# ============================================================
print('\n\n' + '='*70)
print('各形状上模型综合排名 (Noise=0.02)')
print('='*70)

for shape in shapes:
    rankings = []
    for name in models:
        r = results_by_shape[shape]['models'][name]
        # 综合分数：归一化CD + 归一化P2P（越低越好）
        score = r['Chamfer Distance'] + r['P2P Distance']
        rankings.append((name, score, r))

    rankings.sort(key=lambda x: x[1])
    print(f'\n【{shape.upper()}】:')
    for rank, (name, score, r) in enumerate(rankings, 1):
        medal = ['🥇', '🥈', '🥉'][rank-1] if rank <= 3 else f'{rank}.'
        print(f'  {medal} {name}: 综合分={score:.6f} (CD={r["Chamfer Distance"]:.6f}, P2P={r["P2P Distance"]:.6f})')


# ============================================================
# 保存完整结果为JSON
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'analysis_type': 'per_shape_performance',
    'training_noise_level': TRAIN_NOISE,
    'shapes': shapes,
    'noise_levels_evaluated': noise_levels,
    'noisy_baselines_by_shape': noisy_baselines_by_shape,
    'results_by_shape': {
        shape: {
            'baseline': results_by_shape[shape]['baseline'],
            'models': {k: v for k, v in results_by_shape[shape]['models'].items()}
        } for shape in shapes
    },
    'full_matrix': full_matrix,
    'summary': {
        'best_model_per_shape': {},
        'shape_sensitivity': {}
    }
}

# 每种形状的最优模型
for shape in shapes:
    best = min(models.keys(),
               key=lambda m: results_by_shape[shape]['models'][m]['Chamfer Distance'] +
                            results_by_shape[shape]['models'][m]['P2P Distance'])
    output['summary']['best_model_per_shape'][shape] = best

# 形状敏感性
for name in models:
    cds = [results_by_shape[s]['models'][name]['Chamfer Distance'] for s in shapes]
    output['summary']['shape_sensitivity'][name] = {
        'cd_std': float(np.std(cds)),
        'cd_range': float(max(cds) - min(cds)),
        'best_shape': shapes[np.argmin(cds)],
        'worst_shape': shapes[np.argmax(cds)]
    }

output_path = os.path.join(
    r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising',
    'results',
    'shape_performance_analysis.json'
)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f'\n\n结果已保存到: {output_path}')

# 打印可视化用数据摘要
print('\n=== VIS_DATA ===')
vis_summary = {
    'shapes': shapes,
    'models': list(models.keys()),
    'training_noise': TRAIN_NOISE,
    'by_shape': {}
}
for shape in shapes:
    vis_summary['by_shape'][shape] = {}
    for name in models:
        r = results_by_shape[shape]['models'][name]
        b = results_by_shape[shape]['baseline']
        vis_summary['by_shape'][shape][name] = {
            'CD': r['Chamfer Distance'],
            'P2P': r['P2P Distance'],
            'GR': r['Geometric Recall'],
            'CD_improvement': round((b['CD'] - r['Chamfer Distance']) / b['CD'] * 100, 2),
            'P2P_improvement': round((b['P2P'] - r['P2P Distance']) / b['P2P'] * 100, 2),
            'baseline_CD': b['CD'],
            'baseline_P2P': b['P2P']
        }
print(json.dumps(vis_summary, indent=2))
