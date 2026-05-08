"""
简单测试脚本 - 验证模型加载和训练
"""
import os
os.environ['TORCH_DISABLE_CUDA_INIT'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USERNAME'] = 'User'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\torch_cache'

import sys
sys.path.insert(0, r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising')

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from models.iterative_pfn_improved import create_iterativepfn_improved_model
from data.synthetic_data_generator import load_dataset

print('='*50)
print('Simple Training Test')
print('='*50)

# 加载数据
data = load_dataset(r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl')

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

# 创建模型
print('\nCreating model...')
model = create_iterativepfn_improved_model(
    num_points=2048, num_iterations=3, feature_dim=512, hidden_dim=256)
print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')

# 加载检查点
ckpt_path = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\checkpoints\IterativePFN_best.pth'
print(f'\nLoading checkpoint: {ckpt_path}')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f'Loaded from epoch {ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.6f}')

# 测试训练一步
print('\nTesting one training step...')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()

noisy, clean = next(iter(train_loader))
optimizer.zero_grad()
cleaned = model(noisy)
if isinstance(cleaned, tuple):
    cleaned = cleaned[0]
loss = model.get_loss(cleaned, clean)
loss.backward()
optimizer.step()
print(f'Training step successful! Loss = {loss.item():.6f}')

print('\nTest passed!')
