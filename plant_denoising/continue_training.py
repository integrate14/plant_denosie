"""
继续训练脚本 - IterativePFN 和 StraightPCF
从已有的检查点继续训练，保存完整的训练历史
"""
import os
os.environ['TORCH_DISABLE_CUDA_INIT'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USERNAME'] = 'User'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\torch_cache'

import sys
sys.path.insert(0, r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising')

print('='*70)
print('Continue Training - IterativePFN & StraightPCF')
print('='*70)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from datetime import datetime
import json

from data.synthetic_data_generator import SyntheticPointCloudDataset, load_dataset
from models.iterative_pfn_improved import IterativePFNImproved, create_iterativepfn_improved_model
from models.straight_pcf_improved import StraightPCFImproved, create_straightpcf_improved_model

print(f'PyTorch: {torch.__version__}')
print()

# 配置
DATA_PATH = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl'
CHECKPOINT_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 加载数据
print('Loading data...')
data = load_dataset(DATA_PATH)
clean_shapes = data['clean']
noisy_shapes = data['noisy']

class PointCloudDataset(Dataset):
    def __init__(self, clean_shapes, noisy_shapes, shape='sphere', noise_level=0.02):
        self.clean = clean_shapes[shape]
        self.noisy = noisy_shapes[shape][noise_level]
    def __len__(self):
        return len(self.clean)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.noisy[idx]).float(),
                torch.from_numpy(self.clean[idx]['points']).float())

dataset = PointCloudDataset(clean_shapes, noisy_shapes, shape='sphere', noise_level=0.02)
train_size = int(0.8 * len(dataset))
train_subset, val_subset = random_split(dataset, [train_size, len(dataset)-train_size])
train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)
print(f'Data: Train {len(train_subset)}, Val {len(val_subset)}')

device = 'cpu'


def train_model_with_history(model, train_loader, val_loader, num_epochs, model_name,
                               start_epoch=0, resume_ckpt=None):
    """训练模型并保存完整历史记录"""
    print(f'\nTraining {model_name} (from epoch {start_epoch} to {start_epoch + num_epochs})')
    print('-' * 60)

    # 加载检查点（如果指定）
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f'Loading checkpoint: {resume_ckpt}')
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f'Resumed from epoch {start_epoch}')

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 加载已有历史（如果有）
    history_file = os.path.join(CHECKPOINT_DIR, f'{model_name}_history.json')
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            existing = json.load(f)
            # 只保留已有历史，不重复
            history = existing

    best_val_loss = float('inf')
    best_epoch = start_epoch
    start_time = datetime.now()

    # 手动设置初始学习率（跳过已完成的 epoch）
    initial_lr = 0.001 * (0.5 ** (start_epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr
    print(f'Starting with lr={initial_lr:.8f} (skipped {start_epoch} epochs of scheduler)')

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()

            cleaned = model(noisy)
            if isinstance(cleaned, tuple):
                cleaned = cleaned[0]

            # 根据模型类型计算损失
            if model_name == 'IterativePFN':
                # IterativePFN 的 get_loss 接受 (pred_points, clean_points)
                loss = model.get_loss(cleaned, clean)
            elif model_name == 'StraightPCF':
                # StraightPCF 的 get_loss 接受 (pred_points, clean_points)
                loss = model.get_loss(cleaned, clean)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # 验证
        model.eval()
        val_loss = 0.0
        val_cd = 0.0
        val_ptp = 0.0
        n_val = 0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                cleaned = model(noisy)
                if isinstance(cleaned, tuple):
                    cleaned = cleaned[0]

                # 计算损失（用于 val_loss）
                loss = model.get_loss(cleaned, clean)
                val_loss += loss.item()

                # 计算指标（用于记录历史）
                cd = model.chamfer_distance(cleaned, clean)
                ptp = torch.mean(torch.sqrt(torch.sum((cleaned - clean)**2, dim=-1) + 1e-8))
                val_cd += cd.item()
                val_ptp += ptp.item()
                n_val += 1

        val_loss /= max(n_val, 1)
        val_cd /= max(n_val, 1)
        val_ptp /= max(n_val, 1)
        scheduler.step()

        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'val_cd': float(val_cd),
            'val_ptp': float(val_ptp),
            'lr': optimizer.param_groups[0]['lr']
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(CHECKPOINT_DIR, f'{model_name}_best.pth'))

        # 保存最终检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(CHECKPOINT_DIR, f'{model_name}_final.pth'))

        # 保存历史
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        if (epoch + 1) % 10 == 0 or epoch == start_epoch:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f'  Epoch {epoch+1}/{start_epoch + num_epochs}: '
                  f'Train={train_loss:.6f}, Val={val_loss:.6f}, '
                  f'Best={best_val_loss:.6f} (ep{best_epoch}), '
                  f'Time={elapsed:.0f}s')

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f'\n{model_name} Training Complete!')
    print(f'  Best: epoch {best_epoch}, Val={best_val_loss:.6f}')
    print(f'  Total time: {elapsed:.0f}s')
    print(f'  History saved to: {history_file}')

    return best_val_loss, best_epoch


# ============================================================
# 训练 IterativePFN (从 epoch 100 继续到 epoch 200)
# ============================================================
print('\n' + '='*70)
print('CONTINUE TRAINING: IterativePFN (100 -> 200 epochs)')
print('='*70)

model_ipfn = create_iterativepfn_improved_model(
    num_points=2048, num_iterations=3, feature_dim=512, hidden_dim=256)

best_val, best_ep = train_model_with_history(
    model=model_ipfn,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,  # 继续训练 100 epochs
    model_name='IterativePFN',
    start_epoch=100,  # 从 epoch 100 开始
    resume_ckpt=os.path.join(CHECKPOINT_DIR, 'IterativePFN_best.pth')
)

print(f'\nIterativePFN final best: epoch {best_ep}, val_loss = {best_val:.6f}')


# ============================================================
# 训练 StraightPCF (从头开始训练 100 epochs，记录历史)
# ============================================================
print('\n' + '='*70)
print('TRAINING: StraightPCF (0 -> 100 epochs)')
print('='*70)

model_spcf = StraightPCFImproved(
    num_points=2048, feature_dim=256, hidden_dim=128, num_iterations=3, use_dgcnn=False)

best_val2, best_ep2 = train_model_with_history(
    model=model_spcf,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    model_name='StraightPCF',
    start_epoch=0,
    resume_ckpt=None  # StraightPCF 没有检查点，从头开始
)

print(f'\nStraightPCF final best: epoch {best_ep2}, val_loss = {best_val2:.6f}')


# ============================================================
# 保存实验结果摘要
# ============================================================
print('\n' + '='*70)
print('SUMMARY')
print('='*70)

results = {
    'timestamp': datetime.now().isoformat(),
    'training_type': 'continue_training',
    'models': {
        'IterativePFN': {
            'epochs': '100 -> 200',
            'final_best_epoch': best_ep,
            'final_best_val_loss': best_val
        },
        'StraightPCF': {
            'epochs': '0 -> 100',
            'final_best_epoch': best_ep2,
            'final_best_val_loss': best_val2
        }
    }
}

results_path = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results\continue_training_results.json'
os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'Results saved to: {results_path}')
print('\nDone!')
