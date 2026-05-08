"""
PointFilter: Point Cloud Filtering via Encoder-Decoder Modeling
基于编码器-解码器的点云滤波网络

简化版实现 - 修复维度问题
"""

# Disable CUDA initialization to avoid hanging
import os
os.environ['TORCH_DISABLE_CUDA_INIT'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class PointNetEncoder(nn.Module):
    """简化的PointNet编码器"""
    
    def __init__(self, input_dim: int = 3, feature_dim: int = 1024):
        super(PointNetEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        
    def forward(self, points):
        """
        Args:
            points: (B, N, 3) 点云坐标
        Returns:
            global_features: (B, feature_dim) 全局特征
            local_features: (B, N, feature_dim) 局部特征
        """
        x = points.transpose(2, 1)  # (B, 3, N)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # (B, feature_dim, N)
        
        # Global feature
        global_features = torch.max(x, 2, keepdim=False)[0]  # (B, feature_dim)
        
        # Local features
        local_features = x.transpose(2, 1)  # (B, N, feature_dim)
        
        return global_features, local_features


class PointFeatureExtractor(nn.Module):
    """单点特征提取器"""
    
    def __init__(self, k: int = 16, feature_dim: int = 1024):
        super(PointFeatureExtractor, self).__init__()
        self.k = k
        
        # 局部特征MLP
        self.local_mlp = nn.Sequential(
            nn.Linear(k * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # 融合特征MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(256 + feature_dim + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )
        
    def knn(self, points, k):
        """找到每个点的k个最近邻"""
        batch_size = points.size(0)
        num_points = points.size(1)
        
        # 计算距离矩阵 (B, N, N)
        # ||x_i - x_j||^2 = ||x_i||^2 - 2*x_i·x_j + ||x_j||^2
        xx = torch.sum(points ** 2, dim=2, keepdim=True)  # (B, N, 1)
        xy = torch.bmm(points, points.transpose(2, 1))  # (B, N, N)
        pairwise_distance = xx - 2 * xy + xx.transpose(2, 1)  # (B, N, N)
        
        # 找到k近邻（距离最小的k个）
        _, idx = torch.topk(pairwise_distance, k=k, dim=-1, largest=False)
        return idx
    
    def get_local_neighborhood(self, points, idx):
        """获取局部邻域"""
        batch_size = points.size(0)
        num_points = points.size(1)
        k = idx.size(-1)
        
        idx_base = torch.arange(0, batch_size, device=points.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        points_flat = points.view(batch_size * num_points, -1)
        neighbors = points_flat[idx].view(batch_size, num_points, k, 3)
        
        # 相对坐标
        points_expanded = points.unsqueeze(2).expand(batch_size, num_points, k, 3)
        local_coords = neighbors - points_expanded
        
        return local_coords
    
    def forward(self, points, global_features, local_features):
        """
        Args:
            points: (B, N, 3)
            global_features: (B, feature_dim)
            local_features: (B, N, feature_dim)
        Returns:
            point_features: (B, N, 1024)
        """
        batch_size = points.size(0)
        num_points = points.size(1)
        
        # 找到k近邻
        idx = self.knn(points, self.k)
        local_coords = self.get_local_neighborhood(points, idx)  # (B, N, k, 3)
        
        # 处理局部坐标
        local_coords = local_coords.view(batch_size * num_points, self.k * 3)
        local_feat = self.local_mlp(local_coords)  # (B*N, 256)
        local_feat = local_feat.view(batch_size, num_points, 256)
        
        # 融合特征
        global_feat_expanded = global_features.unsqueeze(1).expand(batch_size, num_points, -1)
        concat = torch.cat([points, local_feat, global_feat_expanded], dim=-1)
        
        point_features = self.fusion_mlp(concat)  # (B, N, 1024)
        
        return point_features


class PointNetDecoder(nn.Module):
    """解码器"""
    
    def __init__(self, feature_dim: int = 1024):
        super(PointNetDecoder, self).__init__()
        
        self.displacement_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
    def forward(self, point_features, points):
        """
        Args:
            point_features: (B, N, feature_dim)
            points: (B, N, 3)
        Returns:
            displacements: (B, N, 3)
            cleaned_points: (B, N, 3)
        """
        batch_size = point_features.size(0)
        num_points = point_features.size(1)
        
        # 预测位移
        point_features_flat = point_features.view(batch_size * num_points, -1)
        displacements_flat = self.displacement_mlp(point_features_flat)
        displacements = displacements_flat.view(batch_size, num_points, 3)
        
        # 计算滤波后的点
        cleaned_points = points + displacements
        
        return displacements, cleaned_points


class PointFilter(nn.Module):
    """PointFilter: 完整的点云滤波网络"""
    
    def __init__(self, 
                 num_points: int = 2048,
                 local_k: int = 16,
                 feature_dim: int = 1024):
        super(PointFilter, self).__init__()
        
        self.num_points = num_points
        self.feature_dim = feature_dim
        
        # 编码器
        self.encoder = PointNetEncoder(input_dim=3, feature_dim=feature_dim)
        
        # 点特征提取器
        self.feature_extractor = PointFeatureExtractor(
            k=local_k,
            feature_dim=feature_dim
        )
        
        # 解码器
        self.decoder = PointNetDecoder(feature_dim=1024)
        
    def forward(self, points):
        """
        Args:
            points: (B, N, 3)
        Returns:
            cleaned_points: (B, N, 3)
            displacements: (B, N, 3)
        """
        # 编码
        global_features, local_features = self.encoder(points)
        
        # 提取点特征
        point_features = self.feature_extractor(points, global_features, local_features)
        
        # 解码
        displacements, cleaned_points = self.decoder(point_features, points)
        
        return cleaned_points, displacements
    
    def get_loss(self, pred_points, clean_points):
        """计算损失"""
        cd_loss = self.chamfer_distance(pred_points, clean_points)
        emd_loss = torch.mean(torch.sqrt(torch.sum((pred_points - clean_points) ** 2, dim=-1) + 1e-8))
        return cd_loss + 0.1 * emd_loss
    
    @staticmethod
    def chamfer_distance(points1, points2):
        """Chamfer距离"""
        batch_size = points1.size(0)
        
        points1_expand = points1.unsqueeze(2)
        points2_expand = points2.unsqueeze(1)
        dist1 = torch.sum((points1_expand - points2_expand) ** 2, dim=-1)
        min_dist1 = torch.min(dist1, dim=2)[0]
        cd1 = torch.mean(min_dist1, dim=1)
        
        dist2 = torch.sum((points2_expand - points1_expand) ** 2, dim=-1)
        min_dist2 = torch.min(dist2, dim=2)[0]
        cd2 = torch.mean(min_dist2, dim=1)
        
        return torch.mean(cd1 + cd2)


def create_pointfilter_model(num_points: int = 2048, **kwargs) -> PointFilter:
    """创建PointFilter模型"""
    return PointFilter(num_points=num_points, **kwargs)


if __name__ == '__main__':
    print("=" * 60)
    print("PointFilter 模型测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = PointFilter(num_points=2048).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    batch_size = 4
    noisy_points = torch.randn(batch_size, 2048, 3).to(device)
    clean_points = torch.randn(batch_size, 2048, 3).to(device)
    
    print("运行前向传播...")
    cleaned, displacements = model(noisy_points)
    print(f"输入形状: {noisy_points.shape}")
    print(f"输出形状: {cleaned.shape}")
    print(f"位移形状: {displacements.shape}")
    
    loss = model.get_loss(cleaned, clean_points)
    print(f"损失值: {loss.item():.6f}")
    
    print("\n模型测试通过！")
