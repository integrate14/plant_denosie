"""
IterativePFN改进版 - 添加注意力机制，减少迭代次数，优化损失函数
"""
import os
os.environ['TORCH_DISABLE_CUDA_INIT'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SelfAttention(nn.Module):
    """自注意力机制"""
    def __init__(self, feature_dim: int = 512):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5
        
    def forward(self, x):
        """
        Args:
            x: (B, N, feature_dim)
        Returns:
            attended: (B, N, feature_dim)
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, V)
        return out


class PointWiseAttention(nn.Module):
    """点级注意力机制"""
    def __init__(self, feature_dim: int = 512):
        super(PointWiseAttention, self).__init__()
        self.attention = SelfAttention(feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, N, feature_dim)
        Returns:
            out: (B, N, feature_dim)
        """
        attended = self.attention(x)
        out = self.norm(x + self.fc(attended))
        return out


class SimplePointNet(nn.Module):
    """简化的PointNet特征提取器 (LayerNorm版 - 替换BatchNorm)

    LayerNorm: 对每个样本的每个点，在 channel 维度做归一化。
    不依赖 batch 内其他样本的统计量，适合混合形状训练。
    """

    def __init__(self, feature_dim: int = 512):
        super(SimplePointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, feature_dim, 1)

        # LayerNorm: 归一化通道维度 (C)，对 Conv1d 输出 (B,C,N) 的每列归一化
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(128)
        self.ln3 = nn.LayerNorm(256)
        self.ln4 = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 3, N)
        Returns:
            features:    (B, feature_dim, N)
            global_feat: (B, feature_dim)
        """
        # conv -> transpose -> LN -> transpose back -> ReLU
        x = self.conv1(x)                       # (B, 64, N)
        x = self.ln1(x.transpose(2, 1))         # (B, N, 64)
        x = x.transpose(2, 1).relu()           # (B, 64, N)

        x = self.conv2(x)                       # (B, 128, N)
        x = self.ln2(x.transpose(2, 1))         # (B, N, 128)
        x = x.transpose(2, 1).relu()           # (B, 128, N)

        x = self.conv3(x)                       # (B, 256, N)
        x = self.ln3(x.transpose(2, 1))         # (B, N, 256)
        x = x.transpose(2, 1).relu()           # (B, 256, N)

        x = self.conv4(x)                       # (B, feature_dim, N)
        x = self.ln4(x.transpose(2, 1))         # (B, N, feature_dim)
        x = x.transpose(2, 1).relu()           # (B, feature_dim, N)

        global_feat = torch.max(x, dim=2, keepdim=False)[0]  # (B, feature_dim)
        return x, global_feat


class IterationModule(nn.Module):
    """迭代模块 (增强版 - 带注意力)"""
    
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super(IterationModule, self).__init__()
        
        self.pointnet = SimplePointNet(feature_dim=feature_dim)
        
        # 点级注意力
        self.attention = PointWiseAttention(feature_dim)
        
        # feature_dim + 3 + 1 (features + points + iter_embed)
        self.fc1 = nn.Linear(feature_dim + 3 + 1, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)
        
        
    def forward(self, points, iteration):
        """
        Args:
            points: (B, N, 3)
            iteration: int
        Returns:
            displacements: (B, N, 3)
        """
        batch_size = points.size(0)
        num_points = points.size(1)
        
        points_t = points.transpose(2, 1)
        features, global_feat = self.pointnet(points_t)
        features = features.transpose(2, 1)  # (B, N, feature_dim)
        
        # 应用注意力
        features = self.attention(features)
        
        # 迭代嵌入
        iter_embed = torch.full((batch_size, num_points, 1), iteration / 5.0, device=points.device)

        concat = torch.cat([features, points, iter_embed], dim=-1)  # (B, N, F)
        concat = concat.view(batch_size * num_points, -1)            # (B*N, F)

        x = self.fc1(concat)                                         # (B*N, hidden_dim)
        x = self.ln1(x.view(batch_size, num_points, -1))            # (B, N, hidden_dim)
        x = F.relu(x.view(batch_size * num_points, -1))            # (B*N, hidden_dim)

        x = self.fc2(x)                                             # (B*N, hidden_dim)
        x = self.ln2(x.view(batch_size, num_points, -1))            # (B, N, hidden_dim)
        x = F.relu(x.view(batch_size * num_points, -1))            # (B*N, hidden_dim)

        displacements = self.fc3(x)                                  # (B*N, 3)
        displacements = displacements.view(batch_size, num_points, 3) # (B, N, 3)
        
        return displacements


class AdaptiveGroundTruth(nn.Module):
    """自适应Ground Truth"""
    
    def __init__(self):
        super(AdaptiveGroundTruth, self).__init__()
        
    def compute_target(self, noisy_points, clean_points, iteration, total_iterations):
        """计算自适应目标"""
        progress = iteration / total_iterations
        weight = 1.0 - 0.5 * progress
        target = weight * noisy_points + (1 - weight) * clean_points
        return target


class IterativePFNImproved(nn.Module):
    """
    IterativePFN改进版
    - 减少迭代次数: 5 -> 3
    - 添加注意力机制
    - 使用PointFilter风格的损失函数
    """
    
    def __init__(self, 
                 num_points: int = 2048,
                 num_iterations: int = 3,  # 从5减少到3
                 feature_dim: int = 512,
                 hidden_dim: int = 256):
        super(IterativePFNImproved, self).__init__()
        
        self.num_points = num_points
        self.num_iterations = num_iterations
        
        self.iteration_modules = nn.ModuleList([
            IterationModule(feature_dim=feature_dim, hidden_dim=hidden_dim)
            for _ in range(num_iterations)
        ])
        
        self.adaptive_gt = AdaptiveGroundTruth()
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, noisy_points, return_all_iterations: bool = False):
        """
        Args:
            noisy_points: (B, N, 3)
            return_all_iterations: bool
        Returns:
            cleaned_points: (B, N, 3)
        """
        current_points = noisy_points
        all_points = [noisy_points]
        
        for i in range(self.num_iterations):
            displacements = self.iteration_modules[i](current_points, i)
            # 使用可学习的残差权重
            current_points = current_points + displacements * self.residual_weight
            all_points.append(current_points)
        
        if return_all_iterations:
            return all_points
        else:
            return current_points
    
    def get_loss(self, pred_points, clean_points):
        """
        使用PointFilter风格的损失函数: CD + 0.1 * P2P
        """
        cd_loss = self.chamfer_distance(pred_points, clean_points)
        p2p_loss = torch.mean(torch.sqrt(torch.sum((pred_points - clean_points) ** 2, dim=-1) + 1e-8))
        return cd_loss + 0.1 * p2p_loss
    
    def get_iterative_loss(self, noisy_points, clean_points):
        """迭代损失 - 累积每一步的损失"""
        current_points = noisy_points
        total_loss = 0.0
        
        for i in range(self.num_iterations):
            displacements = self.iteration_modules[i](current_points, i)
            current_points = current_points + displacements * self.residual_weight
            
            # 每一步都计算损失
            step_cd = self.chamfer_distance(current_points, clean_points)
            step_p2p = torch.mean(torch.sqrt(torch.sum((current_points - clean_points) ** 2, dim=-1) + 1e-8))
            step_loss = step_cd + 0.1 * step_p2p
            total_loss += step_loss
        
        return total_loss / self.num_iterations
    
    @staticmethod
    def chamfer_distance(points1, points2):
        """Chamfer距离"""
        points1_expand = points1.unsqueeze(2)
        points2_expand = points2.unsqueeze(1)
        dist1 = torch.sum((points1_expand - points2_expand) ** 2, dim=-1)
        min_dist1 = torch.min(dist1, dim=2)[0]
        cd1 = torch.mean(min_dist1, dim=1)
        
        dist2 = torch.sum((points2_expand - points1_expand) ** 2, dim=-1)
        min_dist2 = torch.min(dist2, dim=2)[0]
        cd2 = torch.mean(min_dist2, dim=1)
        
        return torch.mean(cd1 + cd2)


def create_iterativepfn_improved_model(num_points: int = 2048, num_iterations: int = 3, **kwargs) -> IterativePFNImproved:
    """创建改进版IterativePFN模型"""
    return IterativePFNImproved(num_points=num_points, num_iterations=num_iterations, **kwargs)


if __name__ == '__main__':
    print("=" * 60)
    print("IterativePFN Improved 模型测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = IterativePFNImproved(num_points=2048, num_iterations=3, feature_dim=512, hidden_dim=256).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    batch_size = 4
    noisy_points = torch.randn(batch_size, 2048, 3).to(device)
    clean_points = torch.randn(batch_size, 2048, 3).to(device)
    
    print("运行前向传播...")
    cleaned = model(noisy_points)
    print(f"输入形状: {noisy_points.shape}")
    print(f"输出形状: {cleaned.shape}")
    
    all_iterations = model(noisy_points, return_all_iterations=True)
    print(f"迭代次数: {len(all_iterations) - 1}")
    
    # 测试PointFilter风格的损失
    loss_pf = model.get_loss(cleaned, clean_points)
    print(f"PointFilter风格损失: {loss_pf.item():.6f}")
    
    # 测试迭代损失
    loss_iter = model.get_iterative_loss(noisy_points, clean_points)
    print(f"迭代损失: {loss_iter.item():.6f}")
    
    print("\n模型测试通过！")
