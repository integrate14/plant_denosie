"""
评估指标模块
用于评估点云滤波算法的精度和时间效率
"""

import numpy as np
import torch
import time
from typing import Tuple, Dict, List
import json


def chamfer_distance_np(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    计算两组点云之间的Chamfer距离（NumPy版本）
    
    Args:
        points1: (N, 3) 点云1
        points2: (M, 3) 点云2
    
    Returns:
        Chamfer距离
    """
    # points1到points2的距离
    dist1 = np.sum((points1[:, np.newaxis, :] - points2[np.newaxis, :, :]) ** 2, axis=-1)
    min_dist1 = np.min(dist1, axis=1)
    cd1 = np.mean(min_dist1)
    
    # points2到points1的距离
    dist2 = np.sum((points2[:, np.newaxis, :] - points1[np.newaxis, :, :]) ** 2, axis=-1)
    min_dist2 = np.min(dist2, axis=1)
    cd2 = np.mean(min_dist2)
    
    return (cd1 + cd2) / 2


def hausdorff_distance_np(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    计算Hausdorff距离（NumPy版本）
    
    Args:
        points1: (N, 3) 点云1
        points2: (M, 3) 点云2
    
    Returns:
        Hausdorff距离
    """
    # points1到points2
    dist1 = np.sum((points1[:, np.newaxis, :] - points2[np.newaxis, :, :]) ** 2, axis=-1)
    min_dist1 = np.min(dist1, axis=1)
    max_dist1 = np.max(min_dist1)
    
    # points2到points1
    dist2 = np.sum((points2[:, np.newaxis, :] - points1[np.newaxis, :, :]) ** 2, axis=-1)
    min_dist2 = np.min(dist2, axis=1)
    max_dist2 = np.max(min_dist2)
    
    return max(max_dist1, max_dist2)


def point_to_point_distance_np(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    计算点到点距离（假设点对应关系）
    
    Args:
        points1: (N, 3) 点云1
        points2: (N, 3) 点云2（相同点数）
    
    Returns:
        平均欧氏距离
    """
    return np.mean(np.sqrt(np.sum((points1 - points2) ** 2, axis=-1)))


def earth_movers_distance_np(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    简化版的Earth Mover's Distance
    
    Args:
        points1: (N, 3) 点云1
        points2: (N, 3) 点云2
    
    Returns:
        EMD近似值
    """
    # 使用平均距离作为简化估计
    return np.mean(np.sqrt(np.sum((points1 - points2) ** 2, axis=-1)))


def geometric_recall_np(pred_points: np.ndarray, clean_points: np.ndarray, threshold: float = 0.01) -> float:
    """
    计算几何召回率 (Geometric Recall)
    
    定义：对于预测点云中的每个点，在干净点云中找到最近邻，
    若最近邻距离 <= threshold，则认为该点被"成功召回"。
    几何召回率 = 被召回点数 / 总点数
    
    Args:
        pred_points:  (N, 3) 预测（去噪后）点云
        clean_points: (M, 3) 干净目标点云
        threshold:    判定召回的距离阈值（默认 0.01，归一化坐标空间）
    
    Returns:
        几何召回率，取值 [0, 1]，越高越好
    """
    # 计算 pred_points 中每个点到 clean_points 的最近邻距离
    diff = pred_points[:, np.newaxis, :] - clean_points[np.newaxis, :, :]   # (N, M, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))                               # (N, M)
    min_dist = np.min(dist, axis=1)                                          # (N,)
    recalled = np.sum(min_dist <= threshold)
    return float(recalled) / float(len(pred_points))


class PointCloudMetrics:
    """点云评估指标计算器"""
    
    def __init__(self, recall_threshold: float = 0.01):
        """
        Args:
            recall_threshold: 几何召回率判定阈值（默认 0.01）
        """
        self.recall_threshold = recall_threshold
        self.metrics = {
            'chamfer_distance': [],
            'hausdorff_distance': [],
            'point_to_point_distance': [],
            'earth_movers_distance': [],
            'geometric_recall': []
        }
        
    def compute(self, pred_points: np.ndarray, clean_points: np.ndarray) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            pred_points: (N, 3) 预测的点云
            clean_points: (N, 3) 干净的目标点云
        
        Returns:
            各项指标的值
        """
        results = {}
        
        results['chamfer_distance'] = chamfer_distance_np(pred_points, clean_points)
        results['hausdorff_distance'] = np.sqrt(hausdorff_distance_np(pred_points, clean_points))
        results['point_to_point_distance'] = point_to_point_distance_np(pred_points, clean_points)
        results['earth_movers_distance'] = earth_movers_distance_np(pred_points, clean_points)
        results['geometric_recall'] = geometric_recall_np(pred_points, clean_points,
                                                          threshold=self.recall_threshold)
        
        return results
    
    def add(self, results: Dict[str, float]):
        """添加一组结果"""
        for key, value in results.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取统计摘要"""
        summary = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        return summary
    
    def reset(self):
        """重置所有指标"""
        for key in self.metrics:
            self.metrics[key] = []


class Timer:
    """时间测量器"""
    
    def __init__(self):
        self.times = {}
        self.current_key = None
        self.start_time = None
        
    def start(self, key: str):
        """开始计时"""
        self.current_key = key
        self.start_time = time.time()
        
    def stop(self) -> float:
        """停止计时并返回经过的时间"""
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        if self.current_key not in self.times:
            self.times[self.current_key] = []
        self.times[self.current_key].append(elapsed)
        self.start_time = None
        return elapsed
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取时间统计"""
        summary = {}
        for key, values in self.times.items():
            if len(values) > 0:
                summary[key] = {
                    'total': float(np.sum(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        return summary
    
    def reset(self):
        """重置计时器"""
        self.times = {}
        self.current_key = None
        self.start_time = None


class DenoisingEvaluator:
    """
    点云去噪评估器
    
    综合评估模型的精度和时间效率
    """
    
    def __init__(self):
        self.metrics = PointCloudMetrics()
        self.timer = Timer()
        self.results = {}
        
    def evaluate_model(self, 
                      model, 
                      noisy_points: np.ndarray, 
                      clean_points: np.ndarray,
                      model_name: str,
                      device: str = 'cpu') -> Dict:
        """
        评估单个模型
        
        Args:
            model: 点云滤波模型
            noisy_points: 噪声点云 (N, 3)
            clean_points: 干净点云 (N, 3)
            model_name: 模型名称
            device: 计算设备
        
        Returns:
            评估结果
        """
        # 转换为torch tensor
        noisy_tensor = torch.from_numpy(noisy_points).float().unsqueeze(0).to(device)
        clean_tensor = torch.from_numpy(clean_points).float().unsqueeze(0).to(device)
        
        # 推理时间测量
        self.timer.start('inference')
        with torch.no_grad():
            cleaned_tensor = model(noisy_tensor)
        inference_time = self.timer.stop()
        
        # 转换回numpy
        cleaned_points = cleaned_tensor.squeeze(0).cpu().numpy()
        
        # 计算精度指标
        metrics_results = self.metrics.compute(cleaned_points, clean_points)
        
        result = {
            'model_name': model_name,
            'inference_time': inference_time,
            'num_points': noisy_points.shape[0],
            'metrics': metrics_results
        }
        
        return result
    
    def evaluate_models(self,
                       models: Dict,
                       test_data: List[Tuple[np.ndarray, np.ndarray]],
                       model_names: List[str],
                       device: str = 'cpu') -> List[Dict]:
        """
        评估多个模型
        
        Args:
            models: 模型字典 {name: model}
            test_data: 测试数据列表 [(noisy, clean), ...]
            model_names: 要评估的模型名称列表
            device: 计算设备
        
        Returns:
            所有模型的评估结果
        """
        all_results = []
        
        for model_name in model_names:
            if model_name not in models:
                print(f"警告: 模型 {model_name} 不存在，跳过")
                continue
            
            print(f"\n评估模型: {model_name}")
            print("-" * 40)
            
            model = models[model_name].to(device)
            model.eval()
            
            # 重置指标
            self.metrics.reset()
            
            for i, (noisy, clean) in enumerate(test_data):
                result = self.evaluate_model(model, noisy, clean, model_name, device)
                self.metrics.add(result['metrics'])
                print(f"  样本 {i+1}/{len(test_data)}: CD={result['metrics']['chamfer_distance']:.6f}, "
                      f"时间={result['inference_time']*1000:.2f}ms")
            
            # 获取统计摘要
            metrics_summary = self.metrics.get_summary()
            time_summary = self.timer.get_summary()
            
            summary_result = {
                'model_name': model_name,
                'num_samples': len(test_data),
                'metrics': metrics_summary,
                'timing': time_summary
            }
            
            all_results.append(summary_result)
            self.timer.reset()
        
        return all_results
    
    def print_comparison(self, results: List[Dict]):
        """打印模型对比结果"""
        print("\n" + "=" * 80)
        print("模型性能对比")
        print("=" * 80)
        
        for result in results:
            print(f"\n【{result['model_name']}】")
            print(f"  样本数量: {result['num_samples']}")
            
            print("  精度指标:")
            for metric_name, metric_values in result['metrics'].items():
                print(f"    {metric_name}:")
                print(f"      均值: {metric_values['mean']:.6f} ± {metric_values['std']:.6f}")
                print(f"      范围: [{metric_values['min']:.6f}, {metric_values['max']:.6f}]")
            
            print("  时间效率:")
            if 'inference' in result['timing']:
                t = result['timing']['inference']
                print(f"    平均推理时间: {t['mean']*1000:.2f}ms ± {t['std']*1000:.2f}ms")
                print(f"    总时间: {t['total']:.4f}s")
    
    def save_results(self, results: List[Dict], save_path: str):
        """保存评估结果到JSON文件"""
        # 转换numpy类型为Python原生类型
        serializable_results = []
        for result in results:
            serializable_result = {
                'model_name': result['model_name'],
                'num_samples': result['num_samples'],
                'metrics': {},
                'timing': {}
            }
            
            for metric_name, metric_values in result['metrics'].items():
                serializable_result['metrics'][metric_name] = {
                    k: float(v) for k, v in metric_values.items()
                }
            
            for time_name, time_values in result['timing'].items():
                serializable_result['timing'][time_name] = {
                    k: float(v) if not isinstance(v, int) else v 
                    for k, v in time_values.items()
                }
            
            serializable_results.append(serializable_result)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已保存到: {save_path}")


def create_comparison_table(results: List[Dict]) -> str:
    """
    创建模型对比表格
    
    Args:
        results: 评估结果列表
    
    Returns:
        Markdown格式的对比表格
    """
    table = "| 模型 | Chamfer Distance ↓ | P2P Distance ↓ | 平均时间(ms) |\n"
    table += "|:----:|:----:|:----:|:----:|\n"
    
    for result in results:
        model_name = result['model_name']
        cd = result['metrics']['chamfer_distance']['mean']
        p2p = result['metrics']['point_to_point_distance']['mean']
        
        time_mean = result['timing']['inference']['mean'] * 1000 if 'inference' in result['timing'] else 0
        
        table += f"| {model_name} | {cd:.6f} | {p2p:.6f} | {time_mean:.2f} |\n"
    
    return table


if __name__ == '__main__':
    # 测试评估模块
    print("=" * 60)
    print("评估指标模块测试")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    clean_points = np.random.randn(2048, 3)
    noisy_points = clean_points + np.random.randn(2048, 3) * 0.05
    
    # 计算指标
    metrics = PointCloudMetrics()
    results = metrics.compute(noisy_points, clean_points)
    
    print("\n噪声点云指标:")
    for name, value in results.items():
        print(f"  {name}: {value:.6f}")
    
    # 测试评估器
    evaluator = DenoisingEvaluator()
    print("\n评估器初始化完成")
    
    print("\n评估模块测试完成！")
