"""
StraightPCF改进版 - 增加网络深度，引入残差连接，使用DGCNN特征提取器

修复记录 (2026-04-30):
- 修复 DGCNNFeatureExtractor: edge_conv中BatchNorm1d维度错误，
  重构为正确的EdgeConv结构，对(B*N, feat_in*2)做Linear而非BatchNorm1d(feat)
- 修复 EnhancedPointNetExtractor: 构造参数名由output_dim统一为feature_dim
- 修复 VelocityModuleImproved / DistanceModuleImproved: 错误传参output_dim改为feature_dim
- 修复 DGCNN forward中feature_dim变量作用域缺失问题
"""
import os
os.environ['TORCH_DISABLE_CUDA_INIT'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# 基础构件
# ─────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """残差块（用于EnhancedPointNetExtractor）"""
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2   = nn.BatchNorm1d(out_channels)

        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, 1),
                          nn.BatchNorm1d(out_channels))
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


# ─────────────────────────────────────────────
# 特征提取器
# ─────────────────────────────────────────────

class EnhancedPointNetExtractor(nn.Module):
    """增强版PointNet特征提取器 - 更深的网络 + 残差连接

    Args:
        input_dim:   输入通道数（默认3 = xyz）
        feature_dim: 输出特征维度
    """

    def __init__(self, input_dim: int = 3, feature_dim: int = 512):
        super(EnhancedPointNetExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.bn1   = nn.BatchNorm1d(64)

        self.res1 = ResidualBlock(64, 128)
        self.res2 = ResidualBlock(128, 256)
        self.res3 = ResidualBlock(256, feature_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim, N)
        Returns:
            features:    (B, feature_dim, N)
            global_feat: (B, feature_dim)
        """
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x = self.res1(x)   # (B, 128, N)
        x = self.res2(x)   # (B, 256, N)
        x = self.res3(x)   # (B, feature_dim, N)
        global_feat = torch.max(x, dim=2)[0]  # (B, feature_dim)
        return x, global_feat


class DGCNNFeatureExtractor(nn.Module):
    """DGCNN风格的图卷积特征提取器
    
    正确实现：
    - 为每条边 (center, neighbor-center) 拼接成 2*feat_in 维输入
    - 用 Linear(2*feat_in, feat_out) + BN + ReLU 做边卷积
    - 对 k 个邻居做 max-pooling 得到点特征
    
    Args:
        k:           近邻数量
        feature_dim: 最终输出维度
    """

    def __init__(self, k: int = 16, feature_dim: int = 512):
        super(DGCNNFeatureExtractor, self).__init__()
        self.k = k

        # edge conv 层：输入 = (center_feat || edge_feat), 输出 = next_feat
        self.econv1 = self._make_econv(3  * 2, 64)
        self.econv2 = self._make_econv(64 * 2, 128)
        self.econv3 = self._make_econv(128* 2, 256)
        self.econv4 = self._make_econv(256* 2, feature_dim)

    @staticmethod
    def _make_econv(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    # --------------------------------------------------
    def knn(self, x: torch.Tensor) -> torch.Tensor:
        """返回 k 近邻索引
        Args:  x: (B, N, C)
        Returns: idx: (B, N, k)
        """
        xx  = torch.sum(x ** 2, dim=2, keepdim=True)          # (B, N, 1)
        dist = xx - 2 * torch.bmm(x, x.transpose(2, 1)) + xx.transpose(2, 1)  # (B, N, N)
        _, idx = dist.topk(k=self.k, dim=-1, largest=False)    # (B, N, k)
        return idx

    def get_edge_feature(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """构建边特征 (center || neighbor-center)
        Args:
            x:   (B, N, C)
            idx: (B, N, k)
        Returns:
            edge_feat: (B*N*k, 2C)
        """
        B, N, C = x.shape
        k = idx.shape[-1]

        # 取邻居
        base = torch.arange(B, device=x.device).view(B, 1, 1) * N
        idx_flat = (idx + base).view(-1)              # (B*N*k,)
        x_flat   = x.view(B * N, C)                  # (B*N, C)
        neighbors = x_flat[idx_flat].view(B, N, k, C) # (B, N, k, C)

        center = x.unsqueeze(2).expand(B, N, k, C)    # (B, N, k, C)
        edge   = neighbors - center                    # (B, N, k, C)

        # 拼接：(center || edge) -> (B*N*k, 2C)
        feat = torch.cat([center, edge], dim=-1).view(B * N * k, 2 * C)
        return feat

    # --------------------------------------------------
    def _econv_forward(self, feat_bnk_2c: torch.Tensor, econv: nn.Sequential,
                       B: int, N: int, k: int) -> torch.Tensor:
        """对 (B*N*k, 2C) 做边卷积并聚合到 (B, N, out_dim)"""
        out_dim = econv[0].out_features
        out = econv(feat_bnk_2c)                     # (B*N*k, out_dim)
        out = out.view(B, N, k, out_dim)
        out = torch.max(out, dim=2)[0]               # (B, N, out_dim)
        return out

    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, 3)
        Returns:
            features:    (B, N, feature_dim)
            global_feat: (B, feature_dim)
        """
        B, N, _ = x.shape
        k = self.k

        # Layer 1: 输入 xyz
        idx = self.knn(x)                                      # (B, N, k)
        e1  = self.get_edge_feature(x, idx)                   # (B*N*k, 6)
        x1  = self._econv_forward(e1, self.econv1, B, N, k)  # (B, N, 64)

        # Layer 2
        idx = self.knn(x1)
        e2  = self.get_edge_feature(x1, idx)                  # (B*N*k, 128)
        x2  = self._econv_forward(e2, self.econv2, B, N, k)  # (B, N, 128)

        # Layer 3
        idx = self.knn(x2)
        e3  = self.get_edge_feature(x2, idx)                  # (B*N*k, 256)
        x3  = self._econv_forward(e3, self.econv3, B, N, k)  # (B, N, 256)

        # Layer 4
        idx = self.knn(x3)
        e4  = self.get_edge_feature(x3, idx)                  # (B*N*k, 512)
        x4  = self._econv_forward(e4, self.econv4, B, N, k)  # (B, N, feature_dim)

        global_feat = torch.max(x4, dim=1)[0]                 # (B, feature_dim)
        return x4, global_feat


# ─────────────────────────────────────────────
# 速度模块 & 距离模块
# ─────────────────────────────────────────────

class VelocityModuleImproved(nn.Module):
    """改进版VelocityModule - 更深的网络"""

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256,
                 use_dgcnn: bool = False):
        super(VelocityModuleImproved, self).__init__()

        if use_dgcnn:
            self.feature_extractor = DGCNNFeatureExtractor(feature_dim=feature_dim)
            self._use_dgcnn = True
        else:
            self.feature_extractor = EnhancedPointNetExtractor(
                input_dim=3, feature_dim=feature_dim)
            self._use_dgcnn = False

        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
        )

        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def _extract(self, points: torch.Tensor) -> torch.Tensor:
        """提取全局特征 -> (B, feature_dim)"""
        if self._use_dgcnn:
            _, global_feat = self.feature_extractor(points)         # points: (B, N, 3)
        else:
            _, global_feat = self.feature_extractor(
                points.transpose(2, 1))                              # (B, 3, N)
        return global_feat

    def forward(self, noisy_patch: torch.Tensor,
                high_noise_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_patch:      (B, N, 3)
            high_noise_patch: (B, N, 3)
        Returns:
            velocity: (B, 3)
        """
        noisy_feat = self._extract(noisy_patch)
        high_feat  = self._extract(high_noise_patch)
        fused    = self.fusion_mlp(torch.cat([noisy_feat, high_feat], dim=1))
        velocity = self.velocity_head(fused)
        return velocity


class DistanceModuleImproved(nn.Module):
    """改进版DistanceModule - 更深的网络"""

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 128,
                 use_dgcnn: bool = False):
        super(DistanceModuleImproved, self).__init__()

        if use_dgcnn:
            self.feature_extractor = DGCNNFeatureExtractor(feature_dim=feature_dim)
            self._use_dgcnn = True
        else:
            self.feature_extractor = EnhancedPointNetExtractor(
                input_dim=3, feature_dim=feature_dim)
            self._use_dgcnn = False

        self.global_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def _extract(self, points: torch.Tensor) -> torch.Tensor:
        if self._use_dgcnn:
            _, global_feat = self.feature_extractor(points)
        else:
            _, global_feat = self.feature_extractor(points.transpose(2, 1))
        return global_feat

    def forward(self, points: torch.Tensor,
                high_noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points:     (B, N, 3)
            high_noise: (B, N, 3)
        Returns:
            distance_scalar: (B, 1)
        """
        pts_feat  = self._extract(points)
        high_feat = self._extract(high_noise)
        dist = self.global_mlp(torch.cat([pts_feat, high_feat], dim=1))
        return dist


# ─────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────

class StraightPCFImproved(nn.Module):
    """
    StraightPCF改进版
    - 增加网络深度（更深的MLP）
    - 引入残差连接（ResidualBlock）
    - 可选DGCNN风格特征提取器（use_dgcnn=True/False）
    """

    def __init__(self,
                 num_points: int = 2048,
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 num_iterations: int = 3,
                 use_dgcnn: bool = False):
        super(StraightPCFImproved, self).__init__()

        self.num_points    = num_points
        self.num_iterations = num_iterations

        self.velocity_module = VelocityModuleImproved(
            feature_dim=feature_dim, hidden_dim=hidden_dim, use_dgcnn=use_dgcnn)
        self.distance_module = DistanceModuleImproved(
            feature_dim=feature_dim, hidden_dim=hidden_dim // 2, use_dgcnn=use_dgcnn)

        # 可学习步长
        self.step_scale = nn.Parameter(torch.tensor(0.5))

    def generate_high_noise(self, points: torch.Tensor,
                            noise_level: float = 0.1) -> torch.Tensor:
        return points + torch.randn_like(points) * noise_level

    def forward(self, noisy_points: torch.Tensor,
                num_iterations: int = None) -> torch.Tensor:
        """
        Args:
            noisy_points: (B, N, 3)
        Returns:
            cleaned_points: (B, N, 3)
        """
        if num_iterations is None:
            num_iterations = self.num_iterations

        current = noisy_points
        for _ in range(num_iterations):
            high_noise = self.generate_high_noise(current)

            velocity = self.velocity_module(current, high_noise)   # (B, 3)
            dist_s   = self.distance_module(current, high_noise)   # (B, 1)

            # displacement: (B, N, 3)
            displacement = (velocity.unsqueeze(1) * dist_s.unsqueeze(1)
                            ).expand(-1, self.num_points, -1)

            current = current + displacement * self.step_scale

        return current

    def get_loss(self, pred_points: torch.Tensor,
                 clean_points: torch.Tensor) -> torch.Tensor:
        cd_loss  = self.chamfer_distance(pred_points, clean_points)
        ptp_loss = torch.mean(
            torch.sqrt(torch.sum((pred_points - clean_points) ** 2, dim=-1) + 1e-8))
        return cd_loss + 0.1 * ptp_loss

    @staticmethod
    def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        p1e = p1.unsqueeze(2)   # (B, N, 1, 3)
        p2e = p2.unsqueeze(1)   # (B, 1, M, 3)
        dist = torch.sum((p1e - p2e) ** 2, dim=-1)  # (B, N, M)
        cd1 = torch.mean(torch.min(dist, dim=2)[0], dim=1)
        cd2 = torch.mean(torch.min(dist, dim=1)[0], dim=1)
        return torch.mean(cd1 + cd2)


# ─────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────

def create_straightpcf_improved_model(num_points: int = 2048,
                                      use_dgcnn: bool = False,
                                      **kwargs) -> StraightPCFImproved:
    """创建改进版StraightPCF模型（默认使用EnhancedPointNet，更稳定）"""
    return StraightPCFImproved(num_points=num_points, use_dgcnn=use_dgcnn, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 自测
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.environ['TORCH_DISABLE_CUDA_INIT'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    print("=" * 60)
    print("StraightPCF Improved 模型测试")
    print("=" * 60)

    device = torch.device('cpu')
    B, N = 2, 256   # 小批次加速测试

    print("\n--- EnhancedPointNet版本 ---")
    model_pn = StraightPCFImproved(
        num_points=N, feature_dim=256, hidden_dim=128,
        num_iterations=2, use_dgcnn=False).to(device)
    print(f"参数量: {count_parameters(model_pn):,}")

    noisy = torch.randn(B, N, 3)
    clean = torch.randn(B, N, 3)
    out   = model_pn(noisy)
    loss  = model_pn.get_loss(out, clean)
    print(f"输入: {noisy.shape}  输出: {out.shape}  损失: {loss.item():.6f}")
    print("EnhancedPointNet版本 ✓")

    print("\n--- DGCNN版本 ---")
    model_dgcnn = StraightPCFImproved(
        num_points=N, feature_dim=64, hidden_dim=64,
        num_iterations=2, use_dgcnn=True).to(device)
    print(f"参数量: {count_parameters(model_dgcnn):,}")

    out2  = model_dgcnn(noisy)
    loss2 = model_dgcnn.get_loss(out2, clean)
    print(f"输入: {noisy.shape}  输出: {out2.shape}  损失: {loss2.item():.6f}")
    print("DGCNN版本 ✓")

    print("\n所有测试通过！")
