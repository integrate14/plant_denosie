"""
点云滤波模型训练脚本
支持PointFilter、IterativePFN和StraightPCF三种算法
"""

# Fix for Windows: set USERNAME before torch import
import os
os.environ['USERNAME'] = os.environ.get('USERNAME', 'User')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'torch_cache')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
from typing import Tuple, Dict, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_data_generator import SyntheticPointCloudDataset, load_dataset
from models.pointfilter import PointFilter, create_pointfilter_model
from models.iterative_pfn import IterativePFN, create_iterativepfn_model
from models.straight_pcf import StraightPCF, create_straightpcf_model


class PointCloudDataset(Dataset):
    """点云数据集类"""
    
    def __init__(self, 
                 clean_shapes: dict, 
                 noisy_shapes: dict,
                 shape: str = 'sphere',
                 noise_level: float = 0.02):
        """
        Args:
            clean_shapes: 干净点云字典
            noisy_shapes: 噪声点云字典
            shape: 形状类型
            noise_level: 噪声级别
        """
        self.clean = clean_shapes[shape]
        self.noisy = noisy_shapes[shape][noise_level]
        
    def __len__(self):
        return len(self.clean)
    
    def __getitem__(self, idx):
        clean = torch.from_numpy(self.clean[idx]['points']).float()
        noisy = torch.from_numpy(self.noisy[idx]).float()
        return noisy, clean


def train_model(model, 
                train_loader, 
                val_loader,
                num_epochs: int = 100,
                learning_rate: float = 0.001,
                device: str = 'cuda',
                model_name: str = 'model',
                save_dir: str = 'checkpoints'):
    """
    训练模型
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 计算设备
        model_name: 模型名称
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    
    # 选择优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # 损失函数
    if hasattr(model, 'get_loss'):
        criterion = model.get_loss
    else:
        criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\n开始训练 {model_name}")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    print(f"训练轮数: {num_epochs}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            if hasattr(model, 'forward'):
                cleaned = model(noisy)
                
                if isinstance(cleaned, tuple):
                    cleaned = cleaned[0]
            
            # 计算损失
            if hasattr(model, 'get_loss'):
                loss = model.get_loss(cleaned, clean)
            else:
                loss = criterion(cleaned, clean)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                cleaned = model(noisy)
                if isinstance(cleaned, tuple):
                    cleaned = cleaned[0]
                
                if hasattr(model, 'get_loss'):
                    loss = model.get_loss(cleaned, clean)
                else:
                    loss = criterion(cleaned, clean)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, f'{model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"Epoch {epoch+1}/{num_epochs} - 保存最佳模型")
        
        # 打印训练进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # 保存最终模型
    final_path = os.path.join(save_dir, f'{model_name}_final.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, final_path)
    
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")
    print(f"模型保存在: {save_dir}")
    
    return train_losses, val_losses


def load_model(model_name: str, model_path: str, device: str = 'cuda'):
    """
    加载预训练模型
    
    Args:
        model_name: 模型名称
        model_path: 模型路径
        device: 计算设备
    """
    # 创建模型
    if 'PointFilter' in model_name:
        model = create_pointfilter_model()
    elif 'IterativePFN' in model_name:
        model = create_iterativepfn_model()
    elif 'StraightPCF' in model_name:
        model = create_straightpcf_model()
    else:
        raise ValueError(f"未知模型: {model_name}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model


def quick_train_test():
    """
    快速训练测试 - 使用少量数据验证流程
    """
    print("=" * 60)
    print("快速训练测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 生成小规模测试数据
    print("\n生成测试数据...")
    dataset = SyntheticPointCloudDataset(
        num_samples=20,
        num_points=1024,
        noise_levels=[0.02, 0.05]
    )
    
    data_dir = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data'
    clean_shapes, noisy_shapes = dataset.generate_all(save_dir=data_dir)
    
    # 创建数据集
    train_dataset = PointCloudDataset(
        clean_shapes, noisy_shapes, 
        shape='sphere', noise_level=0.02
    )
    
    # 划分训练/验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    checkpoint_dir = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练PointFilter
    print("\n训练 PointFilter...")
    model_pf = create_pointfilter_model(num_points=1024)
    train_model(model_pf, train_loader, val_loader, 
                num_epochs=10, device=device,
                model_name='PointFilter',
                save_dir=checkpoint_dir)
    
    # 训练IterativePFN
    print("\n训练 IterativePFN...")
    model_ipfn = create_iterativepfn_model(num_points=1024, num_iterations=3)
    train_model(model_ipfn, train_loader, val_loader,
                num_epochs=10, device=device,
                model_name='IterativePFN',
                save_dir=checkpoint_dir)
    
    # 训练StraightPCF
    print("\n训练 StraightPCF...")
    model_spcf = create_straightpcf_model(num_points=1024)
    train_model(model_spcf, train_loader, val_loader,
                num_epochs=10, device=device,
                model_name='StraightPCF',
                save_dir=checkpoint_dir)
    
    print("\n快速训练测试完成！")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练点云滤波模型')
    parser.add_argument('--model', type=str, default='all',
                       choices=['PointFilter', 'IterativePFN', 'StraightPCF', 'all'],
                       help='要训练的模型')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--num_points', type=int, default=2048,
                       help='每样本点数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 设备选择
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # 加载数据
    print("加载数据集...")
    data_path = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl'
    
    if os.path.exists(data_path):
        data = load_dataset(data_path)
        clean_shapes = data['clean']
        noisy_shapes = data['noisy']
    else:
        print("数据集不存在，正在生成...")
        dataset = SyntheticPointCloudDataset(
            num_samples=100,
            num_points=args.num_points,
            noise_levels=[0.01, 0.02, 0.05, 0.1]
        )
        clean_shapes, noisy_shapes = dataset.generate_all(
            save_dir=r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data'
        )
    
    # 创建数据集
    train_dataset = PointCloudDataset(
        clean_shapes, noisy_shapes,
        shape='sphere', noise_level=0.02
    )
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    checkpoint_dir = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练选定的模型
    if args.model in ['PointFilter', 'all']:
        print("\n" + "=" * 60)
        print("训练 PointFilter")
        print("=" * 60)
        model = create_pointfilter_model(num_points=args.num_points)
        train_model(model, train_loader, val_loader,
                   num_epochs=args.epochs, learning_rate=args.lr,
                   device=device, model_name='PointFilter',
                   save_dir=checkpoint_dir)
    
    if args.model in ['IterativePFN', 'all']:
        print("\n" + "=" * 60)
        print("训练 IterativePFN")
        print("=" * 60)
        model = create_iterativepfn_model(num_points=args.num_points)
        train_model(model, train_loader, val_loader,
                   num_epochs=args.epochs, learning_rate=args.lr,
                   device=device, model_name='IterativePFN',
                   save_dir=checkpoint_dir)
    
    if args.model in ['StraightPCF', 'all']:
        print("\n" + "=" * 60)
        print("训练 StraightPCF")
        print("=" * 60)
        model = create_straightpcf_model(num_points=args.num_points)
        train_model(model, train_loader, val_loader,
                   num_epochs=args.epochs, learning_rate=args.lr,
                   device=device, model_name='StraightPCF',
                   save_dir=checkpoint_dir)
    
    print("\n所有训练完成！")
