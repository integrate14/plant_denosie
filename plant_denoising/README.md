# 植物点云去噪 - 最优模型

本项目包含三个深度学习模型用于植物点云去噪，经过训练和评估后，各模型保留最优版本。

## 最优模型性能

| 模型 | Chamfer Distance | Point-to-Point Distance | 训练轮次 | 配置 |
|:---|:---:|:---:|:---:|:---|
| **IterativePFN** | **0.001515** | 0.0296 | 100 | Improved版本，带Self-Attention |
| **PointFilter** | 0.001790 | **0.0314** | 100 | High-P2P配置 (CD:P2P = 1:0.5) |
| **StraightPCF** | 0.007412 | 0.0598 | 50 | 基础版本 |

> **结论**: IterativePFN-Improved 综合性能最优，CD指标领先其他模型约16%。

## 项目结构

```
plant_denoising/
├── checkpoints/              # 模型检查点
│   ├── IterativePFN_best.pth      # 最优IterativePFN模型
│   ├── IterativePFN_history.json  # 训练历史
│   ├── PointFilter_best.pth       # 最优PointFilter模型
│   ├── PointFilter_history.json   # 训练历史
│   ├── StraightPCF_best.pth       # StraightPCF模型
│   └── StraightPCF_final.pth
├── data/                     # 数据
│   ├── synthetic_dataset.pkl
│   └── synthetic_data_generator.py
├── evaluation/               # 评估模块
│   ├── metrics.py
│   └── train.py
├── models/                   # 模型实现
│   ├── iterative_pfn_improved.py   # IterativePFN (最优)
│   ├── pointfilter.py              # PointFilter
│   └── straight_pcf_improved.py    # StraightPCF
├── results/                  # 结果和可视化
│   ├── experiment_results.json
│   ├── research_report.md
│   └── visualizations/       # 可视化图表
├── main.py                   # 主程序入口
├── train_complete.py         # 完整训练脚本
└── README.md                 # 本文件
```

## 快速开始

### 环境要求
- Python 3.13+
- PyTorch 2.13.0+ (CPU版本)
- NumPy, Matplotlib

### 运行去噪
```bash
python main.py
```

### 重新训练
```bash
python train_complete.py
```

## 模型说明

### IterativePFN (最优)
- **特点**: 迭代优化 + Self-Attention机制
- **优势**: Chamfer Distance表现最佳
- **适用**: 对几何精度要求高的场景

### PointFilter
- **特点**: 端到端滤波网络
- **优势**: 训练稳定，P2P指标优秀
- **适用**: 需要平衡速度和精度的场景

### StraightPCF
- **特点**: 直接预测位移向量
- **优势**: 模型简单，推理速度快
- **适用**: 实时性要求高的场景

## 可视化结果

所有性能对比图表保存在 `results/visualizations/`:
- `dashboard.png` - 综合仪表板
- `model_comparison.png` - 模型对比
- `denoising_effect_comparison.png` - 去噪效果对比
- `deep_learning_comparison.png` - 深度学习模型对比
- `final_performance_report.md` - 详细性能报告

## 引用

- PointFilter: [PointFilter paper]
- IterativePFN: [IterativePFN paper]
- StraightPCF: [StraightPCF paper]
