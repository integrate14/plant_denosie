"""
Plant Point Cloud Denoising Project
虚拟点云噪声数据生成模块
"""

import numpy as np
import os
from typing import Tuple, List
import pickle

def set_random_seed(seed: int = 42):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)

set_random_seed(42)


def generate_sphere_point_cloud(num_points: int = 2048, radius: float = 1.0) -> np.ndarray:
    """
    生成标准球体点云
    
    Args:
        num_points: 点数量
        radius: 球体半径
    
    Returns:
        形状为 (num_points, 3) 的点云坐标
    """
    # 使用均匀采样生成球面点
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(costheta)
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    points = np.stack([x, y, z], axis=1)
    return points


def generate_cube_point_cloud(num_points: int = 2048, size: float = 2.0) -> np.ndarray:
    """
    生成标准立方体点云
    
    Args:
        num_points: 点数量
        size: 立方体边长
    
    Returns:
        形状为 (num_points, 3) 的点云坐标
    """
    points = []
    half_size = size / 2
    
    # 每个面分配相同数量的点
    points_per_face = num_points // 6
    remainder = num_points % 6
    
    faces = [
        (0, 1, 2, half_size, 0, 0),   # +X面
        (0, 1, 2, -half_size, 0, 0),  # -X面
        (1, 2, 0, 0, half_size, 0),    # +Y面
        (1, 2, 0, 0, -half_size, 0),   # -Y面
        (0, 2, 1, 0, 0, half_size),   # +Z面
        (0, 2, 1, 0, 0, -half_size),  # -Z面
    ]
    
    for i, (ax, ay, az, x, y, z) in enumerate(faces):
        count = points_per_face + (1 if i < remainder else 0)
        face_points = np.random.uniform(-half_size, half_size, (count, 2))
        
        pt = np.zeros((count, 3))
        pt[:, ax] = x
        pt[:, ay] = face_points[:, 0]
        pt[:, az] = face_points[:, 1]
        points.append(pt)
    
    points = np.concatenate(points, axis=0)
    return points


def generate_cone_point_cloud(num_points: int = 2048, height: float = 2.0, radius: float = 1.0) -> np.ndarray:
    """
    生成圆锥体点云
    
    Args:
        num_points: 点数量
        height: 圆锥高度
        radius: 圆锥底面半径
    
    Returns:
        形状为 (num_points, 3) 的点云坐标
    """
    points = []
    
    # 圆锥侧面
    num_side = int(num_points * 0.7)
    num_base = num_points - num_side
    
    # 侧面：从顶点到底面的渐变
    heights = np.random.uniform(0, 1, num_side)
    radii = height - heights * height
    radii = radii * (radius / height)
    
    theta = np.random.uniform(0, 2 * np.pi, num_side)
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    z = heights * height
    
    side_points = np.stack([x, y, z], axis=1)
    points.append(side_points)
    
    # 底面圆
    r_base = np.random.uniform(0, radius, num_base)
    t_base = np.random.uniform(0, 2 * np.pi, num_base)
    x_base = r_base * np.cos(t_base)
    y_base = r_base * np.sin(t_base)
    
    base_points = np.stack([x_base, y_base, np.zeros(num_base)], axis=1)
    points.append(base_points)
    
    points = np.concatenate(points, axis=0)
    return points


def add_gaussian_noise(points: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
    """
    添加高斯噪声
    
    Args:
        points: 原始点云，形状 (N, 3)
        noise_level: 噪声标准差，相对于点云尺度的比例
    
    Returns:
        添加噪声后的点云
    """
    noise = np.random.normal(0, noise_level, points.shape)
    noisy_points = points + noise
    return noisy_points


def normalize_point_cloud(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    归一化点云到单位球内
    
    Args:
        points: 输入点云
    
    Returns:
        归一化后的点云, 中心点, 缩放因子
    """
    center = np.mean(points, axis=0)
    centered = points - center
    scale = np.max(np.linalg.norm(centered, axis=1))
    normalized = centered / scale
    return normalized, center, scale


def denormalize_point_cloud(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """
    反归一化点云
    
    Args:
        points: 归一化点云
        center: 中心点
        scale: 缩放因子
    
    Returns:
        原始尺度的点云
    """
    return points * scale + center


class SyntheticPointCloudDataset:
    """虚拟点云数据集类"""
    
    def __init__(self, 
                 num_samples: int = 100,
                 num_points: int = 2048,
                 noise_levels: List[float] = [0.01, 0.02, 0.05, 0.1]):
        """
        初始化数据集生成器
        
        Args:
            num_samples: 每种形状的样本数量
            num_points: 每个样本的点数
            noise_levels: 噪声级别列表
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.noise_levels = noise_levels
        self.shapes = ['sphere', 'cube', 'cone']
    
    def generate_clean_shapes(self) -> dict:
        """生成所有干净形状的点云"""
        clean_shapes = {}
        
        for shape in self.shapes:
            clean_shapes[shape] = []
            for _ in range(self.num_samples):
                if shape == 'sphere':
                    points = generate_sphere_point_cloud(self.num_points)
                elif shape == 'cube':
                    points = generate_cube_point_cloud(self.num_points)
                else:  # cone
                    points = generate_cone_point_cloud(self.num_points)
                
                points, center, scale = normalize_point_cloud(points)
                clean_shapes[shape].append({
                    'points': points,
                    'center': center,
                    'scale': scale
                })
        
        return clean_shapes
    
    def generate_noisy_shapes(self, clean_shapes: dict) -> dict:
        """基于干净形状生成带噪声的点云"""
        noisy_shapes = {}
        
        for shape in self.shapes:
            noisy_shapes[shape] = {}
            for noise_level in self.noise_levels:
                noisy_shapes[shape][noise_level] = []
                
                for clean_data in clean_shapes[shape]:
                    clean_points = clean_data['points']
                    # 反归一化后添加噪声，再归一化
                    original = denormalize_point_cloud(clean_points, 
                                                      clean_data['center'], 
                                                      clean_data['scale'])
                    noisy_original = add_gaussian_noise(original, noise_level)
                    noisy_points, _, _ = normalize_point_cloud(noisy_original)
                    
                    noisy_shapes[shape][noise_level].append(noisy_points)
        
        return noisy_shapes
    
    def generate_all(self, save_dir: str = None) -> Tuple[dict, dict]:
        """
        生成完整的训练和测试数据集
        
        Args:
            save_dir: 保存路径
        
        Returns:
            (clean_shapes, noisy_shapes)
        """
        print("生成干净形状点云...")
        clean_shapes = self.generate_clean_shapes()
        
        print("生成带噪声点云...")
        noisy_shapes = self.generate_noisy_shapes(clean_shapes)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            data = {
                'clean': clean_shapes,
                'noisy': noisy_shapes,
                'metadata': {
                    'num_samples': self.num_samples,
                    'num_points': self.num_points,
                    'noise_levels': self.noise_levels,
                    'shapes': self.shapes
                }
            }
            save_path = os.path.join(save_dir, 'synthetic_dataset.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"数据集已保存到: {save_path}")
        
        return clean_shapes, noisy_shapes


def load_dataset(data_path: str) -> dict:
    """加载预生成的数据集"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    # 测试数据生成
    print("=" * 60)
    print("虚拟点云数据生成测试")
    print("=" * 60)
    
    # 创建小规模测试数据集
    dataset = SyntheticPointCloudDataset(
        num_samples=5,
        num_points=2048,
        noise_levels=[0.01, 0.02, 0.05, 0.1]
    )
    
    # 生成数据
    data_dir = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data'
    clean_shapes, noisy_shapes = dataset.generate_all(save_dir=data_dir)
    
    # 打印统计信息
    print("\n数据集统计:")
    print(f"  - 形状类型: {dataset.shapes}")
    print(f"  - 每种形状样本数: {dataset.num_samples}")
    print(f"  - 每样本点数: {dataset.num_points}")
    print(f"  - 噪声级别: {dataset.noise_levels}")
    
    # 验证数据
    print("\n数据验证:")
    for shape in dataset.shapes:
        for noise_level in dataset.noise_levels:
            clean_sample = clean_shapes[shape][0]['points']
            noisy_sample = noisy_shapes[shape][noise_level][0]
            
            clean_mean = np.mean(clean_sample, axis=0)
            noisy_mean = np.mean(noisy_sample, axis=0)
            
            print(f"  {shape} (noise={noise_level}):")
            print(f"    干净点云均值: {clean_mean}")
            print(f"    噪声点云均值: {noisy_mean}")
            print(f"    点数匹配: {clean_sample.shape[0] == noisy_sample.shape[0]}")
    
    print("\n数据生成完成！")
