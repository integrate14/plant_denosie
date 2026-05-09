"""
混合形状训练脚本 - 同时使用 sphere / cube / cone 三种几何形状数据训练
解决之前模型只对 sphere 有效、cube/cone 严重退化的问题

核心改动：
1. 数据集：混合三种形状（每种 100 样本 x 3 形状 = 300 总样本）
2. 划分：70/15/15 训练/验证/测试
3. 每个批次随机混合不同形状，强迫模型学习通用去噪特征
4. 评估时按形状分别统计，验证泛化性
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
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import numpy as np
import json
from datetime import datetime

from data.synthetic_data_generator import load_dataset
from models.pointfilter import create_pointfilter_model
from models.iterative_pfn_improved import create_iterativepfn_improved_model
from models.straight_pcf_improved import create_straightpcf_improved_model

print('='*70)
print('Mixed-Shape Training: Sphere + Cube + Cone')
print('='*70)
print(f'PyTorch: {torch.__version__}\n')

# ============================================================
# 数据准备 - 混合三种形状
# ============================================================
DATA_PATH = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl'
CHECKPOINT_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\checkpoints'
RESULTS_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results'
NOISE_LEVEL = 0.02  # 训练噪声级别

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

data = load_dataset(DATA_PATH)
SHAPES = data['metadata']['shapes']  # ['sphere', 'cube', 'cone']
print(f'形状类型: {SHAPES}')
print(f'噪声级别: {NOISE_LEVEL}')
print(f'每种形状样本数: {len(data["clean"]["sphere"])}')


class SingleShapeDataset(Dataset):
    """单种形状的点云数据集"""
    def __init__(self, clean_shapes, noisy_shapes, shape, noise_level):
        self.clean = clean_shapes[shape]
        self.noisy = noisy_shapes[shape][noise_level]
        self.shape_name = shape

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.noisy[idx]).float(),
                torch.from_numpy(self.clean[idx]['points']).float())


class MixedShapeDataset(Dataset):
    """
    混合多种形状的数据集
    将所有形状的样本拼接在一起，每个样本带有 shape 标签
    """
    def __init__(self, clean_shapes, noisy_shapes, shapes, noise_level):
        self.datasets = []
        self.shape_indices = []  # 记录每个样本属于哪个形状
        for si, shape in enumerate(shapes):
            ds = SingleShapeDataset(clean_shapes, noisy_shapes, shape, noise_level)
            self.datasets.append(ds)
            self.shape_indices.extend([si] * len(ds))

        self.total_length = sum(len(ds) for ds in self.datasets)

        # 预加载数据到内存加速访问
        self.all_noisy = []
        self.all_clean = []
        for ds in self.datasets:
            for i in range(len(ds)):
                n, c = ds[i]
                self.all_noisy.append(n)
                self.all_clean.append(c)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        return (self.all_noisy[idx], self.all_clean[idx], self.shape_indices[idx])


def build_data_loaders(clean_shapes, noisy_shapes, shapes, noise_level, batch_size=4,
                       train_ratio=0.7, val_ratio=0.15, seed=42):
    """构建混合形状的训练/验证/测试 DataLoader"""
    # 创建混合数据集
    full_dataset = MixedShapeDataset(clean_shapes, noisy_shapes, shapes, noise_level)

    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    print(f'\n混合数据集统计:')
    print(f'  总样本数: {total_size} ({len(shapes)} 种形状 x 每种 ~{total_size//len(shapes)} 样本)')
    print(f'  划分: Train={train_size}, Val={val_size}, Test={test_size}')

    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=gen
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            drop_last=True)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = build_data_loaders(
    data['clean'], data['noisy'], SHAPES, NOISE_LEVEL, batch_size=8
)


device = 'cpu'


# ============================================================
# 训练函数
# ============================================================
def compute_loss(model, cleaned, clean, model_name):
    """根据模型类型计算损失函数"""
    if model_name == 'PointFilter':
        return model.get_loss(cleaned, clean)
    elif model_name == 'IterativePFN':
        return model.get_loss(cleaned, clean)  # 关键：使用 cleaned 输出计算 loss
    elif model_name == 'StraightPCF':
        cd = model.chamfer_distance(cleaned, clean)
        ptp = torch.mean(torch.sqrt(torch.sum((cleaned - clean)**2, dim=-1) + 1e-8))
        return cd + 0.1 * ptp
    else:
        raise ValueError(f'Unknown model: {model_name}')


def train_model(model, train_loader, val_loader, num_epochs, model_name, save_suffix=''):
    """
    训练单个模型
    
    Args:
        save_suffix: 检查点文件名后缀，如 '_mixed' 用于区分不同训练策略
    """
    print(f'\n{"="*60}')
    print(f'Training: {model_name}{" " + save_suffix if save_suffix else ""}')
    print(f'{"="*60}')

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    start_time = datetime.now()

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            # MixedShapeDataset 返回 (noisy, clean, shape_idx)
            if len(batch) == 3:
                noisy, clean, _ = batch
            else:
                noisy, clean = batch

            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()

            cleaned = model(noisy)
            if isinstance(cleaned, tuple):
                cleaned = cleaned[0]

            loss = compute_loss(model, cleaned, clean, model_name)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # ---- Validation ----
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

                cleaned = model(noisy)
                if isinstance(cleaned, tuple):
                    cleaned = cleaned[0]

                loss = compute_loss(model, cleaned, clean, model_name)
                val_loss += loss.item()
                n_val += 1

        val_loss /= max(n_val, 1)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            ckpt_name = f'{model_name}{save_suffix}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'training_type': 'mixed_shape',
                'shapes': SHAPES,
                'noise_level': NOISE_LEVEL,
            }, os.path.join(CHECKPOINT_DIR, ckpt_name))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f'  Epoch {epoch+1:3d}/{num_epochs}: '
                  f'Train={train_loss:.6f}, Val={val_loss:.6f} '
                  f'(best={best_val_loss:.6f}@ep{best_epoch}) [{elapsed:.0f}s]',
                  flush=True)

    total_elapsed = (datetime.now() - start_time).total_seconds()
    print(f'\n  Done! Best Val={best_val_loss:.6f} @ Epoch {best_epoch}, Total={total_elapsed:.0f}s',
          flush=True)

    # Save training history
    hist_path = os.path.join(CHECKPOINT_DIR, f'{model_name}{save_suffix}_history.json')
    with open(hist_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'training_type': 'mixed_shape',
            'shapes': SHAPES,
            'noise_level': NOISE_LEVEL,
            'num_epochs': num_epochs,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'total_time_seconds': total_elapsed,
            'history': history
        }, f, indent=2)

    return best_val_loss, history


# ============================================================
# 评估函数 - 按形状分别统计
# ============================================================
def evaluate_per_shape(model, data, shapes, noise_level, loader_fn=None, model_name='unknown'):
    """
    在每个形状上分别评估模型性能
    
    Returns:
        dict: {shape_name: {cd, p2p, gr}}
    """
    model.eval()
    results = {}

    for shape in shapes:
        # 为该形状创建独立的测试集
        shape_dataset = SingleShapeDataset(data['clean'], data['noisy'], shape, noise_level)
        total_size = len(shape_dataset)
        
        # 使用与训练时相同的划分比例和种子保持一致
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        gen = torch.Generator().manual_seed(42)
        _, _, test_subset = random_split(shape_dataset, [train_size, val_size, test_size], generator=gen)
        
        # 包装成不带 shape_idx 的格式
        class WrapperDS(Dataset):
            def __init__(self, subset): self.subset = subset
            def __len__(self): return len(self.subset)
            def __getitem__(self, idx):
                item = self.subset[idx]
                return (item[0], item[1])
        
        wrapped_test = WrapperDS(test_subset)
        test_loader = DataLoader(wrapped_test, batch_size=8, shuffle=False)

        total_cd = 0
        total_p2p = 0
        total_gr = 0
        n = 0

        with torch.no_grad():
            for noisy, clean in test_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                cleaned = model(noisy)
                if isinstance(cleaned, tuple):
                    cleaned = cleaned[0]

                cd = model.chamfer_distance(cleaned, clean)
                p2p = torch.mean(torch.sqrt(torch.sum((cleaned - clean)**2, dim=-1) + 1e-8))
                
                dists = torch.cdist(cleaned, clean)
                min_dists, _ = torch.min(dists, dim=2)
                gr = torch.mean((min_dists < 0.01).float())

                total_cd += cd.item()
                total_p2p += p2p.item()
                total_gr += gr.item()
                n += 1

        results[shape] = {
            'Chamfer Distance': total_cd / max(n, 1),
            'P2P Distance': total_p2p / max(n, 1),
            'Geometric Recall': total_gr / max(n, 1),
            'test_samples': n
        }

    return results


# ============================================================
# 训练三个模型
# ============================================================
NUM_EPOCHS = 30
SAVE_SUFFIX = '_mixed'

print('\n' + '='*70)
print('PHASE 1: TRAINING (Mixed Shapes)')
print('='*70)

models_to_train = {
    'PointFilter': lambda: create_pointfilter_model(num_points=2048),
    'IterativePFN': lambda: create_iterativepfn_improved_model(
        num_points=2048, num_iterations=3, feature_dim=256, hidden_dim=128
    ),
    'StraightPCF': lambda: create_straightpcf_improved_model(
        num_points=2048, feature_dim=256, hidden_dim=128
    ),
}

trained_models = {}
training_results = {}

for name, model_fn in models_to_train.items():
    model = model_fn()
    best_val, history = train_model(model, train_loader, val_loader, 
                                     num_epochs=NUM_EPOCHS, model_name=name,
                                     save_suffix=SAVE_SUFFIX)
    trained_models[name] = model
    training_results[name] = {'best_val_loss': best_val, 'history_len': len(history['train_loss'])}


# ============================================================
# Phase 2: 按形状详细评估
# ============================================================
print('\n\n' + '='*70)
print('PHASE 2: EVALUATION (Per-Shape Analysis)')
print('='*70)

all_eval_results = {}
comparison_summary = {}

for name, model in trained_models.items():
    # 加载最佳检查点
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{name}{SAVE_SUFFIX}_best.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'{name}: 已加载混合训练检查点')

    # 按形状评估
    per_shape_results = evaluate_per_shape(model, data, SHAPES, NOISE_LEVEL, model_name=name)
    all_eval_results[name] = per_shape_results

    # 打印该模型的各形状结果
    print(f'\n--- {name} 各形状表现 ---')
    print(f'  {"Shape":10s} | {"CD ↓":>12s} | {"P2P ↓":>12s} | {"GR ↑":>10s}')
    print(f'  {"-"*10}-+-{"-"*12}-+-{"-"*12}-+-{"-"*10}')

    for shape in SHAPES:
        r = per_shape_results[shape]
        print(f'  {shape:10s} | {r["Chamfer Distance"]:>12.6f} | {r["P2P Distance"]:>12.6f} | {r["Geometric Recall"]:>10.4f}')

    # 统计该模型的跨形状稳定性
    cds = [per_shape_results[s]['Chamfer Distance'] for s in SHAPES]
    p2ps = [per_shape_results[s]['P2P Distance'] for s in SHAPES]
    comparison_summary[name] = {
        'mean_cd': float(np.mean(cds)),
        'std_cd': float(np.std(cds)),
        'mean_p2p': float(np.mean(p2ps)),
        'std_p2p': float(np.std(p2ps)),
        'best_shape': SHAPES[np.argmin(cds)],
        'worst_shape': SHAPES[np.argmax(cds)],
        'per_shape': per_shape_results
    }


# ============================================================
# 打印对比表格
# ============================================================
print('\n\n' + '='*70)
print('MIXED TRAINING RESULTS SUMMARY')
print('='*70)

print(f'\n| Model           | Shape   | CD       | P2P      | GR     |')
print(f'|-----------------|---------|----------|----------|--------|')
for name in trained_models:
    for shape in SHAPES:
        r = all_eval_results[name][shape]
        marker = ''
        print(f'| {name:15s} | {shape:7s} | {r["Chamfer Distance"]:.6f} | {r["P2P Distance"]:.6f} | {r["Geometric Recall"]:.4f} |{marker}')


# ============================================================
# 保存完整结果
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'training_type': 'mixed_shape',
    'shapes': SHAPES,
    'noise_level': NOISE_LEVEL,
    'num_epochs': NUM_EPOCHS,
    'dataset_info': {
        'samples_per_shape': len(data['clean'][SHAPES[0]]),
        'total_samples_mixed': sum(len(data['clean'][s]) for s in SHAPES),
        'split': '70/15/15'
    },
    'training_summary': training_results,
    'evaluation': all_eval_results,
    'stability_analysis': comparison_summary
}

output_path = os.path.join(RESULTS_DIR, 'mixed_shape_training_results.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f'\n\n结果已保存到: {output_path}')

# 输出可视化数据
print('\n=== VIS_DATA ===')
vis = {'shapes': SHAPES, 'models': list(trained_models.keys()), 'by_shape': {}}
for name in trained_models:
    vis['by_shape'][name] = {}
    for shape in SHAPES:
        r = all_eval_results[name][shape]
        vis['by_shape'][name][shape] = {
            'CD': r['Chamfer Distance'],
            'P2P': r['P2P Distance'],
            'GR': r['Geometric Recall']
        }
print(json.dumps(vis, indent=2))

print('\n' + '='*70)
print('MIXED-SHAPE TRAINING COMPLETE!')
print('='*70)
