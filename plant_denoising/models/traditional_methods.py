"""传统点云去噪方法 (已修复 MLS 收缩 bug)"""
import numpy as np
from typing import List


# =============================================================================
# 工具函数
# =============================================================================

def _get_k_neighbors(points: np.ndarray, k: int) -> np.ndarray:
    try:
        from sklearn.neighbors import KDTree
        tree = KDTree(points)
        _, idx = tree.query(points, k=k + 1)
        return idx[:, 1:]
    except ImportError:
        N = points.shape[0]
        dist = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2) + 1e-12)
        idx = np.argsort(dist, axis=1)[:, 1:k + 1]
        return idx


def _get_radius_neighbors(points: np.ndarray, radius: float) -> List[np.ndarray]:
    try:
        from sklearn.neighbors import KDTree
        tree = KDTree(points)
        result = tree.query_radius(points, r=radius)
    except ImportError:
        N = points.shape[0]
        dist = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2) + 1e-12)
        result = [np.where(dist[i] <= radius)[0] for i in range(N)]
    final = []
    for i, nb in enumerate(result):
        if isinstance(nb, np.ndarray):
            mask = nb != i
            final.append(nb[mask])
        else:
            final.append(np.array([n for n in nb if n != i], dtype=np.int64))
    return final


# =============================================================================
# 1. 高斯滤波
# =============================================================================

def gaussian_filter(points: np.ndarray, k: int = 16, sigma: float = 0.02) -> np.ndarray:
    N = points.shape[0]
    idx = _get_k_neighbors(points, k)
    nb = points[idx]
    dist = np.sqrt(np.sum((nb - points[:, None, :]) ** 2, axis=2) + 1e-8)
    w = np.exp(-0.5 * (dist / (sigma + 1e-8)) ** 2)
    w /= (np.sum(w, axis=1, keepdims=True) + 1e-8)
    return np.sum(nb * w[:, :, None], axis=1)


# =============================================================================
# 2. 双边滤波
# =============================================================================

def bilateral_filter(points: np.ndarray, k: int = 16,
                     sigma_s: float = 0.02, sigma_r: float = 0.02) -> np.ndarray:
    N = points.shape[0]
    idx = _get_k_neighbors(points, k)
    nb = points[idx]
    dist = np.sqrt(np.sum((nb - points[:, None, :]) ** 2, axis=2) + 1e-8)
    w_s = np.exp(-0.5 * (dist / (sigma_s + 1e-8)) ** 2)
    w_r = np.exp(-0.5 * (dist / (sigma_r + 1e-8)) ** 2)
    w = w_s * w_r
    w /= (np.sum(w, axis=1, keepdims=True) + 1e-8)
    return np.sum(nb * w[:, :, None], axis=1)


# =============================================================================
# 3. 统计离群点去除 (SOR) + 修复
# =============================================================================

def sor_denoise(points: np.ndarray, k: int = 16,
                std_ratio: float = 2.0) -> np.ndarray:
    N = points.shape[0]
    idx = _get_k_neighbors(points, k)
    nb = points[idx]
    dists = np.sqrt(np.sum((nb - points[:, None, :]) ** 2, axis=2) + 1e-8)
    mean_d = np.mean(dists, axis=1)

    mu = np.mean(mean_d)
    sigma = np.std(mean_d)
    outlier_mask = mean_d > (mu + std_ratio * sigma)

    corrected = points.copy()
    for i in np.where(outlier_mask)[0]:
        nb_pts = points[idx[i]]
        nb_d = np.sqrt(np.sum((nb_pts - points[i]) ** 2, axis=1) + 1e-8)
        w = np.exp(-0.5 * (nb_d / (mu + 1e-8)) ** 2)
        w /= (np.sum(w) + 1e-8)
        corrected[i] = np.sum(nb_pts * w[:, None], axis=0)
    return corrected


# =============================================================================
# 4. 半径离群点去除 (ROR) + 修复
# =============================================================================

def ror_denoise(points: np.ndarray, radius: float = 0.05,
                min_neighbors: int = 4) -> np.ndarray:
    N = points.shape[0]
    r_nb = _get_radius_neighbors(points, radius)
    outlier_mask = np.array([len(nb) < min_neighbors for nb in r_nb])

    corrected = points.copy()
    outlier_idx = np.where(outlier_mask)[0]
    if len(outlier_idx) == 0:
        return corrected

    try:
        from sklearn.neighbors import KDTree
        inlier_pts = points[~outlier_mask]
        if len(inlier_pts) > 0:
            tree = KDTree(inlier_pts)
            for i in outlier_idx:
                _, ind = tree.query([points[i]], k=1)
                corrected[i] = inlier_pts[ind[0][0]]
    except ImportError:
        for i in outlier_idx:
            d = np.sqrt(np.sum((points - points[i]) ** 2, axis=1) + 1e-12)
            d[outlier_mask] = np.inf
            corrected[i] = points[np.argmin(d)]
    return corrected


# =============================================================================
# 5. 中值滤波
# =============================================================================

def median_filter(points: np.ndarray, k: int = 16) -> np.ndarray:
    idx = _get_k_neighbors(points, k)
    nb = points[idx]
    return np.median(nb, axis=1)


# =============================================================================
# 6. 移动最小二乘 (MLS) — 修复版：不收缩
# =============================================================================

def mls_denoise(points: np.ndarray, k: int = 16,
                 sigma: float = 0.02) -> np.ndarray:
    """
    移动最小二乘 (修复版):
      - 对每个点, 用 k 近邻拟合局部平面
      - 将点投影到平面上 (投影点 = 去噪结果)
      - 注意: 不要加加权平均, 否则球面会收缩
    """
    N = points.shape[0]
    idx = _get_k_neighbors(points, k)
    projected = np.zeros_like(points)

    for i in range(N):
        nb = points[idx[i]]   # (k, 3)
        centroid = np.mean(nb, axis=0)
        nb_centered = nb - centroid

        try:
            cov = (nb_centered.T @ nb_centered) / (len(nb) - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]  # 最小特征值 → 法向
        except np.linalg.LinAlgError:
            normal = np.array([0., 0., 1.])

        # 将点投影到局部平面 (不额外加权平均, 避免收缩)
        v = points[i] - centroid
        dist_to_plane = np.dot(v, normal)
        proj = points[i] - dist_to_plane * normal
        projected[i] = proj

    return projected


# =============================================================================
# 统一接口
# =============================================================================

def apply_method(points: np.ndarray, method: str, **kwargs) -> np.ndarray:
    method_map = {
        'gaussian':  gaussian_filter,
        'bilateral': bilateral_filter,
        'sor':       sor_denoise,
        'ror':       ror_denoise,
        'median':    median_filter,
        'mls':       mls_denoise,
    }
    if method not in method_map:
        raise ValueError(f'未知方法: {method}')
    return method_map[method](points, **kwargs)
