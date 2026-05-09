"""
仅训练 StraightPCF (Mixed Shape) - 其他模型已完成
修复: drop_last=True 避免 BatchNorm batch_size=1 崩溃
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
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
from datetime import datetime

from data.synthetic_data_generator import load_dataset
from models.pointfilter import create_pointfilter_model
from models.iterative_pfn_improved import create_iterativepfn_improved_model
from models.straight_pcf_improved import create_straightpcf_improved_model

print('='*70)
print('STRAIGHTPCF MIXED-SHAPE TRAINING (only)')
print('='*70)
print(f'PyTorch: {torch.__version__}\n')

# ============================================================
# 数据准备
# ============================================================
DATA_PATH = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl'
CHECKPOINT_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\checkpoints'
RESULTS_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results'
NOISE_LEVEL = 0.02

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

data = load_dataset(DATA_PATH)
SHAPES = data['metadata']['shapes']
print(f'形状类型: {SHAPES}, 噪声级别: {NOISE_LEVEL}')


class SingleShapeDataset(Dataset):
    def __init__(self, clean_shapes, noisy_shapes, shape, noise_level):
        self.clean = clean_shapes[shape]
        self.noisy = noisy_shapes[shape][noise_level]

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.noisy[idx]).float(),
                torch.from_numpy(self.clean[idx]['points']).float())


class MixedShapeDataset(Dataset):
    def __init__(self, clean_shapes, noisy_shapes, shapes, noise_level):
        self.datasets = []
        self.shape_indices = []
        for si, shape in enumerate(shapes):
            ds = SingleShapeDataset(clean_shapes, noisy_shapes, shape, noise_level)
            self.datasets.append(ds)
            self.shape_indices.extend([si] * len(ds))
        self.total_length = sum(len(ds) for ds in self.datasets)
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
    full_dataset = MixedShapeDataset(clean_shapes, noisy_shapes, shapes, noise_level)
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    print(f'混合数据集: Total={total_size}, Train={train_size}, Val={val_size}, Test={test_size}')

    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=gen
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              drop_last=True)   # 关键：drop_last 避免 BatchNorm batch=1 崩溃
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            drop_last=True)     # 同上
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = build_data_loaders(
    data['clean'], data['noisy'], SHAPES, NOISE_LEVEL, batch_size=8
)


device = 'cpu'


def compute_loss(model, cleaned, clean, model_name):
    if model_name == 'PointFilter':
        return model.get_loss(cleaned, clean)
    elif model_name == 'IterativePFN':
        return model.get_loss(cleaned, clean)
    elif model_name == 'StraightPCF':
        cd = model.chamfer_distance(cleaned, clean)
        ptp = torch.mean(torch.sqrt(torch.sum((cleaned - clean)**2, dim=-1) + 1e-8))
        return cd + 0.1 * ptp


def train_model(model, train_loader, val_loader, num_epochs, model_name, save_suffix=''):
    print(f'\n{"="*60}')
    print(f'Training: {model_name}{save_suffix}')
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
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(scheduler.get_last_lr()[0])

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


def evaluate_per_shape(model, data, shapes, noise_level, model_name='unknown'):
    """按形状分别评估"""
    model.eval()
    results = {}

    for shape in shapes:
        shape_dataset = SingleShapeDataset(data['clean'], data['noisy'], shape, noise_level)
        total_size = len(shape_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        gen = torch.Generator().manual_seed(42)
        _, _, test_subset = random_split(shape_dataset, [train_size, val_size, test_size], generator=gen)

        class WrapperDS(Dataset):
            def __init__(self, subset): self.subset = subset
            def __len__(self): return len(self.subset)
            def __getitem__(self, idx):
                item = self.subset[idx]
                return (item[0], item[1])

        wrapped_test = WrapperDS(test_subset)
        test_loader = DataLoader(wrapped_test, batch_size=8, shuffle=False)

        total_cd = 0; total_p2p = 0; total_gr = 0; n = 0
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
                total_cd += cd.item(); total_p2p += p2p.item(); total_gr += gr.item(); n += 1

        results[shape] = {
            'Chamfer Distance': total_cd / max(n, 1),
            'P2P Distance': total_p2p / max(n, 1),
            'Geometric Recall': total_gr / max(n, 1),
            'test_samples': n
        }
    return results


# ============================================================
# 只训练 StraightPCF
# ============================================================
NUM_EPOCHS = 30
SAVE_SUFFIX = '_mixed'

# 先加载已完成训练的模型（从 checkpoint）
trained_models = {}
training_results = {}

for name in ['PointFilter', 'IterativePFN']:
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{name}{SAVE_SUFFIX}_best.pth')
    if os.path.exists(ckpt_path):
        print(f'\n加载已有的 {name} mixed checkpoint...')
        if name == 'PointFilter':
            m = create_pointfilter_model(num_points=2048)
        elif name == 'IterativePFN':
            m = create_iterativepfn_improved_model(num_points=2048, num_iterations=3,
                                                    feature_dim=256, hidden_dim=128)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model_state_dict'])
        trained_models[name] = m
        training_results[name] = {'best_val_loss': ckpt['val_loss'], 'loaded': True}
        print(f'  {name}: Val Loss = {ckpt["val_loss"]:.6f}')
    else:
        print(f'  警告: 未找到 {name} checkpoint')

# 训练 StraightPCF
print('\n\n' + '='*70)
print('PHASE 1: TRAINING StraightPCF (Mixed Shapes)')
print('='*70)

spcf_model = create_straightpcf_improved_model(num_points=2048, feature_dim=256, hidden_dim=128)
best_val, history = train_model(spcf_model, train_loader, val_loader,
                                 num_epochs=NUM_EPOCHS, model_name='StraightPCF',
                                 save_suffix=SAVE_SUFFIX)
trained_models['StraightPCF'] = spcf_model
training_results['StraightPCF'] = {'best_val_loss': best_val, 'history_len': len(history['train_loss'])}


# ============================================================
# Phase 2: 按形状详细评估所有三个模型
# ============================================================
print('\n\n' + '='*70)
print('PHASE 2: EVALUATION (Per-Shape Analysis)')
print('='*70)

all_eval_results = {}
comparison_summary = {}

for name, model in trained_models.items():
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{name}{SAVE_SUFFIX}_best.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'{name}: 已加载混合训练检查点')

    per_shape_results = evaluate_per_shape(model, data, SHAPES, NOISE_LEVEL, model_name=name)
    all_eval_results[name] = per_shape_results

    print(f'\n--- {name} 各形状表现 ---')
    print(f'  {"Shape":10s} | {"CD ↓":>12s} | {"P2P ↓":>12s} | {"GR ↑":>10s}')
    print(f'  {"-"*10}-+-{"-"*12}-+-{"-"*12}-+-{"-"*10}')
    for shape in SHAPES:
        r = per_shape_results[shape]
        print(f'  {shape:10s} | {r["Chamfer Distance"]:>12.6f} | {r["P2P Distance"]:>12.6f} | {r["Geometric Recall"]:>10.4f}')

    cds = [per_shape_results[s]['Chamfer Distance'] for s in SHAPES]
    p2ps = [per_shape_results[s]['P2P Distance'] for s in SHAPES]
    comparison_summary[name] = {
        'mean_cd': float(np.mean(cds)), 'std_cd': float(np.std(cds)),
        'mean_p2p': float(np.mean(p2ps)), 'std_p2p': float(np.std(p2ps)),
        'best_shape': SHAPES[np.argmin(cds)], 'worst_shape': SHAPES[np.argmax(cds)],
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
        print(f'| {name:15s} | {shape:7s} | {r["Chamfer Distance"]:.6f} | {r["P2P Distance"]:.6f} | {r["Geometric Recall"]:.4f} |')


# ============================================================
# 保存完整结果
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'training_type': 'mixed_shape',
    'shapes': SHAPES, 'noise_level': NOISE_LEVEL, 'num_epochs': NUM_EPOCHS,
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
