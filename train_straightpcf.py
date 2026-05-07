"""
单独训练 StraightPCF - 记录完整训练历史
"""
import os
os.environ['TORCH_DISABLE_CUDA_INIT'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USERNAME'] = 'User'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\torch_cache'

import sys
sys.path.insert(0, r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising')

print('='*70)
print('Training StraightPCF (0 -> 100 epochs)')
print('='*70)

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
import json

from data.synthetic_data_generator import load_dataset
from models.straight_pcf_improved import StraightPCFImproved

print(f'PyTorch: {torch.__version__}')

# 配置
DATA_PATH = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl'
CHECKPOINT_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\checkpoints'

# 加载数据
print('\nLoading data...')
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

dataset = PointCloudDataset(data['clean'], data['noisy'], shape='sphere', noise_level=0.02)
train_size = int(0.8 * len(dataset))
train_subset, val_subset = random_split(dataset, [train_size, len(dataset)-train_size])
train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)
print(f'Data: Train {len(train_subset)}, Val {len(val_subset)}')

device = 'cpu'

# 创建模型
print('\nCreating StraightPCF model...')
model = StraightPCFImproved(
    num_points=2048, feature_dim=256, hidden_dim=128, num_iterations=3, use_dgcnn=False)
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# 初始化历史
history = []
best_val_loss = float('inf')
best_epoch = 0
start_time = datetime.now()

NUM_EPOCHS = 100

print(f'\nTraining from epoch 1 to {NUM_EPOCHS}...')
print('-'*60)

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    n_batches = 0

    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()

        cleaned = model(noisy)
        if isinstance(cleaned, tuple):
            cleaned = cleaned[0]

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

            cd = model.chamfer_distance(cleaned, clean)
            ptp = torch.mean(torch.sqrt(torch.sum((cleaned - clean)**2, dim=-1) + 1e-8))
            loss = model.get_loss(cleaned, clean)

            val_loss += loss.item()
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
        }, os.path.join(CHECKPOINT_DIR, 'StraightPCF_best.pth'))

    # 保存最终检查点
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
    }, os.path.join(CHECKPOINT_DIR, 'StraightPCF_final.pth'))

    if (epoch + 1) % 10 == 0 or epoch == 0:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}: Train={train_loss:.6f}, Val={val_loss:.6f}, '
              f'Best={best_val_loss:.6f} (ep{best_epoch}), Time={elapsed:.0f}s')

# 保存历史
history_file = os.path.join(CHECKPOINT_DIR, 'StraightPCF_history.json')
with open(history_file, 'w') as f:
    json.dump(history, f, indent=2)

elapsed = (datetime.now() - start_time).total_seconds()
print(f'\nStraightPCF Training Complete!')
print(f'Best: epoch {best_epoch}, val_loss={best_val_loss:.6f}')
print(f'Total time: {elapsed:.0f}s')
print(f'History saved to: {history_file}')
