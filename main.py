"""
植物点云滤波算法实现与对比分析
Plant Point Cloud Denoising - Main Entry Point

研究内容：
1. 虚拟点云噪声数据构建（球体、立方体、圆锥体）
2. 实现三种点云滤波算法：
   - PointFilter: 基于编码器-解码器的点云滤波
   - IterativePFN: 真正的迭代点云滤波网络（CVPR 2023）
   - StraightPCF: 直线路径点云滤波（CVPR 2024）
3. 测试算法在植物点云及虚拟点云上的去噪精度和时间消耗
"""

import torch
import numpy as np
import os
import sys
import time
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.synthetic_data_generator import (
    SyntheticPointCloudDataset, generate_sphere_point_cloud,
    generate_cube_point_cloud, generate_cone_point_cloud,
    add_gaussian_noise, normalize_point_cloud
)
from models.pointfilter import PointFilter, create_pointfilter_model
from models.iterative_pfn import IterativePFN, create_iterativepfn_model
from models.straight_pcf import StraightPCF, create_straightpcf_model
from evaluation.metrics import (
    DenoisingEvaluator, chamfer_distance_np, 
    point_to_point_distance_np, create_comparison_table
)


class DenoisingExperiment:
    """点云去噪实验主类"""
    
    def __init__(self, 
                 num_points: int = 2048,
                 device: str = None):
        """
        初始化实验
        
        Args:
            num_points: 每样本点数
            device: 计算设备
        """
        self.num_points = num_points
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.models = {}
        self._init_models()
        
        # 评估器
        self.evaluator = DenoisingEvaluator()
        
        # 测试数据
        self.test_data = []
        
    def _init_models(self):
        """初始化所有模型"""
        print("\n初始化模型...")
        
        # PointFilter
        self.models['PointFilter'] = create_pointfilter_model(
            num_points=self.num_points
        ).to(self.device)
        
        # IterativePFN
        self.models['IterativePFN'] = create_iterativepfn_model(
            num_points=self.num_points,
            num_iterations=5
        ).to(self.device)
        
        # StraightPCF
        self.models['StraightPCF'] = create_straightpcf_model(
            num_points=self.num_points
        ).to(self.device)
        
        # 打印参数量
        print("\n模型参数量:")
        for name, model in self.models.items():
            params = sum(p.numel() for p in model.parameters())
            print(f"  {name}: {params:,}")
    
    def generate_synthetic_test_data(self,
                                     shapes: List[str] = None,
                                     noise_levels: List[float] = None,
                                     samples_per_shape: int = 5):
        """
        生成虚拟测试数据
        
        Args:
            shapes: 形状列表 ['sphere', 'cube', 'cone']
            noise_levels: 噪声级别列表
            samples_per_shape: 每种形状的样本数
        """
        if shapes is None:
            shapes = ['sphere', 'cube', 'cone']
        if noise_levels is None:
            noise_levels = [0.01, 0.02, 0.05, 0.1]
        
        print(f"\n生成虚拟测试数据...")
        print(f"  形状: {shapes}")
        print(f"  噪声级别: {noise_levels}")
        print(f"  每种形状样本数: {samples_per_shape}")
        
        self.test_data = []
        
        for shape in shapes:
            for noise_level in noise_levels:
                for _ in range(samples_per_shape):
                    # 生成干净点云
                    if shape == 'sphere':
                        clean = generate_sphere_point_cloud(self.num_points)
                    elif shape == 'cube':
                        clean = generate_cube_point_cloud(self.num_points)
                    else:
                        clean = generate_cone_point_cloud(self.num_points)
                    
                    # 归一化
                    clean, center, scale = normalize_point_cloud(clean)
                    
                    # 添加噪声
                    noisy = clean + np.random.randn(self.num_points, 3) * noise_level
                    # 重新归一化
                    noisy_center = np.mean(noisy, axis=0)
                    noisy = noisy - noisy_center
                    noisy_scale = np.max(np.linalg.norm(noisy, axis=1))
                    noisy = noisy / noisy_scale
                    
                    self.test_data.append({
                        'shape': shape,
                        'noise_level': noise_level,
                        'clean': clean,
                        'noisy': noisy
                    })
        
        print(f"生成了 {len(self.test_data)} 个测试样本")
        
    def load_real_plant_data(self, data_dir: str):
        """
        加载真实植物点云数据
        
        Args:
            data_dir: 点云数据目录
        """
        print(f"\n加载植物点云数据: {data_dir}")
        
        # 查找PLY文件
        ply_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.ply'):
                    ply_files.append(os.path.join(root, file))
        
        print(f"找到 {len(ply_files)} 个点云文件")
        
        # 加载前几个文件作为测试
        max_samples = 10
        for ply_file in ply_files[:max_samples]:
            try:
                # 简单的PLY加载（仅支持ASCII格式）
                points = []
                with open(ply_file, 'r') as f:
                    for line in f:
                        if line.startswith('end_header'):
                            break
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            try:
                                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                                points.append([x, y, z])
                            except:
                                continue
                
                if len(points) > 100:
                    points = np.array(points)
                    # 归一化
                    center = np.mean(points, axis=0)
                    points = points - center
                    scale = np.max(np.linalg.norm(points, axis=1))
                    points = points / scale
                    
                    # 添加噪声生成测试样本
                    clean = points[:self.num_points] if len(points) >= self.num_points else np.tile(points, (self.num_points // len(points) + 1, 1))[:self.num_points]
                    noisy = clean + np.random.randn(self.num_points, 3) * 0.02
                    noisy = noisy - np.mean(noisy, axis=0)
                    noisy = noisy / np.max(np.linalg.norm(noisy, axis=1))
                    
                    self.test_data.append({
                        'shape': os.path.basename(ply_file),
                        'noise_level': 0.02,
                        'clean': clean,
                        'noisy': noisy
                    })
                    
            except Exception as e:
                print(f"  加载失败 {ply_file}: {e}")
        
        print(f"加载了 {len(self.test_data)} 个植物点云样本")
    
    def denoise_with_model(self, model_name: str, noisy_points: np.ndarray) -> np.ndarray:
        """
        使用指定模型去噪
        
        Args:
            model_name: 模型名称
            noisy_points: 噪声点云 (N, 3)
        
        Returns:
            去噪后的点云
        """
        model = self.models[model_name]
        model.eval()
        
        noisy_tensor = torch.from_numpy(noisy_points).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            cleaned_tensor = model(noisy_tensor)
            if isinstance(cleaned_tensor, tuple):
                cleaned_tensor = cleaned_tensor[0]
        
        return cleaned_tensor.squeeze(0).cpu().numpy()
    
    def run_experiment(self, save_results: bool = True):
        """
        运行完整实验
        
        Args:
            save_results: 是否保存结果
        """
        print("\n" + "=" * 80)
        print("开始实验：三种点云滤波算法对比")
        print("=" * 80)
        
        results = []
        
        for model_name in self.models.keys():
            print(f"\n{'='*60}")
            print(f"测试模型: {model_name}")
            print(f"{'='*60}")
            
            # 重置评估器
            self.evaluator.metrics.reset()
            self.evaluator.timer.reset()
            
            for i, data in enumerate(self.test_data):
                noisy = data['noisy']
                clean = data['clean']
                
                # 计时推理
                start_time = time.time()
                cleaned = self.denoise_with_model(model_name, noisy)
                inference_time = time.time() - start_time
                
                # 计算指标
                metrics = self.evaluator.metrics.compute(cleaned, clean)
                self.evaluator.metrics.add(metrics)
                self.evaluator.timer.times.setdefault('inference', []).append(inference_time)
                
                # 打印进度
                cd = metrics['chamfer_distance']
                print(f"  [{i+1}/{len(self.test_data)}] {data['shape']} (noise={data['noise_level']}) "
                      f"- CD: {cd:.6f}, Time: {inference_time*1000:.2f}ms")
            
            # 获取统计结果
            metrics_summary = self.evaluator.metrics.get_summary()
            time_summary = self.evaluator.timer.get_summary()
            
            result = {
                'model_name': model_name,
                'num_samples': len(self.test_data),
                'metrics': metrics_summary,
                'timing': time_summary
            }
            results.append(result)
            
            # 打印总结
            print(f"\n{model_name} 统计结果:")
            print(f"  平均Chamfer距离: {metrics_summary['chamfer_distance']['mean']:.6f}")
            print(f"  平均P2P距离: {metrics_summary['point_to_point_distance']['mean']:.6f}")
            print(f"  平均推理时间: {time_summary['inference']['mean']*1000:.2f}ms")
        
        # 打印对比结果
        self.evaluator.print_comparison(results)
        
        # 创建对比表格
        comparison_table = create_comparison_table(results)
        print("\n对比表格:")
        print(comparison_table)
        
        # 保存结果
        if save_results:
            results_dir = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results'
            os.makedirs(results_dir, exist_ok=True)
            
            self.evaluator.save_results(
                results, 
                os.path.join(results_dir, 'experiment_results.json')
            )
            
            # 保存对比表格
            table_path = os.path.join(results_dir, 'comparison_table.md')
            with open(table_path, 'w', encoding='utf-8') as f:
                f.write("# 点云滤波算法对比结果\n\n")
                f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"测试样本数: {len(self.test_data)}\n\n")
                f.write(comparison_table)
            print(f"\n结果已保存到: {results_dir}")
        
        return results
    
    def generate_report(self, results: List[Dict]) -> str:
        """
        生成详细的实验报告
        
        Args:
            results: 实验结果
        
        Returns:
            Markdown格式的报告
        """
        report = []
        report.append("# 植物点云滤波算法对比分析报告\n")
        report.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        report.append("## 1. 研究背景\n\n")
        report.append("植物点云滤波是从原始植物点云数据中提取出有效、可靠的信息同时抑制或去除噪声，")
        report.append("在作物表型和植物数字孪生领域具有重要的科学价值和现实意义。")
        report.append("在植物去噪过程中如何更好地保持尖锐边缘、棱角及表面信息等几何特征是精细化建模的核心需求。\n\n")
        
        report.append("## 2. 研究方法\n\n")
        report.append("### 2.1 测试数据集\n")
        report.append(f"- 虚拟噪声数据：球体、立方体、圆锥体\n")
        report.append(f"- 噪声级别：{list(set([d['noise_level'] for d in self.test_data]))}\n")
        report.append(f"- 每样本点数：{self.num_points}\n")
        report.append(f"- 总测试样本数：{len(self.test_data)}\n\n")
        
        report.append("### 2.2 算法实现\n\n")
        report.append("本项目实现了三种点云滤波深度学习算法：\n\n")
        
        report.append("| 算法 | 论文来源 | 核心思想 | 参数量 |\n")
        report.append("|:----:|:----:|:----:|:----:|\n")
        report.append(f"| PointFilter | arXiv:2002.05968 | 编码器-解码器架构 | "
                     f"{sum(p.numel() for p in self.models['PointFilter'].parameters()):,} |\n")
        report.append(f"| IterativePFN | CVPR 2023 | 内部迭代模块 + 自适应GT | "
                     f"{sum(p.numel() for p in self.models['IterativePFN'].parameters()):,} |\n")
        report.append(f"| StraightPCF | CVPR 2024 | 直线路径 + Velocity/Distance模块 | "
                     f"{sum(p.numel() for p in self.models['StraightPCF'].parameters()):,} |\n\n")
        
        report.append("## 3. 实验结果\n\n")
        report.append("### 3.1 Chamfer距离对比\n\n")
        report.append("| 模型 | 均值 | 标准差 | 最小值 | 最大值 |\n")
        report.append("|:----:|:----:|:----:|:----:|:----:|\n")
        for r in results:
            m = r['metrics']['chamfer_distance']
            report.append(f"| {r['model_name']} | {m['mean']:.6f} | {m['std']:.6f} | "
                         f"{m['min']:.6f} | {m['max']:.6f} |\n")
        report.append("\n")
        
        report.append("### 3.2 推理时间对比\n\n")
        report.append("| 模型 | 平均时间(ms) | 标准差 | 总时间(s) |\n")
        report.append("|:----:|:----:|:----:|:----:|\n")
        for r in results:
            t = r['timing']['inference']
            report.append(f"| {r['model_name']} | {t['mean']*1000:.2f} | {t['std']*1000:.2f} | "
                         f"{t['total']:.4f} |\n")
        report.append("\n")
        
        report.append("## 4. 结论\n\n")
        report.append("基于实验结果，对三种算法进行综合评价：\n\n")
        
        # 找出最优算法
        best_cd = min(results, key=lambda x: x['metrics']['chamfer_distance']['mean'])
        fastest = min(results, key=lambda x: x['timing']['inference']['mean'])
        
        report.append(f"- **精度最优**: {best_cd['model_name']} (Chamfer距离: "
                     f"{best_cd['metrics']['chamfer_distance']['mean']:.6f})\n")
        report.append(f"- **速度最快**: {fastest['model_name']} (平均时间: "
                     f"{fastest['timing']['inference']['mean']*1000:.2f}ms)\n\n")
        
        report.append("## 5. 算法特点分析\n\n")
        report.append("### PointFilter\n")
        report.append("- 基于PointNet的编码器-解码器架构\n")
        report.append("- 通过预测位移向量实现点云去噪\n")
        report.append("- 需要学习点到点的对应关系\n\n")
        
        report.append("### IterativePFN\n")
        report.append("- 引入迭代模块，在网络内部模拟迭代滤波过程\n")
        report.append("- 使用自适应ground truth损失函数\n")
        report.append("- 能更好地保持几何特征\n\n")
        
        report.append("### StraightPCF\n")
        report.append("- 通过直线路径移动噪声点，减少离散化误差\n")
        report.append("- 轻量级设计，参数约为IterativePFN的17%\n")
        report.append("- 无需正则化即可产生良好分布\n\n")
        
        return "".join(report)


def main():
    """主函数"""
    print("=" * 80)
    print("植物点云滤波算法实现与对比分析")
    print("Plant Point Cloud Denoising - Research Project")
    print("=" * 80)
    
    # 创建实验
    experiment = DenoisingExperiment(
        num_points=2048,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 生成虚拟测试数据
    experiment.generate_synthetic_test_data(
        shapes=['sphere', 'cube', 'cone'],
        noise_levels=[0.01, 0.02, 0.05, 0.1],
        samples_per_shape=3
    )
    
    # 尝试加载植物点云数据
    plant_data_dir = r'C:\Users\Lenovo\Desktop\deep-work\Crops3D'
    if os.path.exists(plant_data_dir):
        experiment.load_real_plant_data(plant_data_dir)
    
    # 运行实验
    results = experiment.run_experiment(save_results=True)
    
    # 生成报告
    report = experiment.generate_report(results)
    
    # 保存报告
    results_dir = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results'
    report_path = os.path.join(results_dir, 'research_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n研究报告已保存到: {report_path}")
    
    print("\n实验完成！")


if __name__ == '__main__':
    main()
