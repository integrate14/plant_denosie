# 植物点云去噪 - 最优模型

本项目包含三个深度学习模型用于植物点云去噪，经过训练和评估后，各模型保留最优版本。

## 最优模型性能（2026-05-11 更新）

### 综合评估结果（混合形状：sphere + cube + cone）

| 模型 | Mean CD ↓ | Mean P2P ↓ | Mean GR ↑ | 备注 |
|:---|:---------:|:----------:|:---------:|:-----|
| **IterativePFN (LayerNorm)** | **0.001293** | **0.026658** | **0.9343** | 🏆 **最优**，LayerNorm优化版 |
| PointFilter | 0.002604 | 0.034620 | 0.7482 | 综合第二 |
| StraightPCF | 0.002931 | 0.036667 | 0.7189 | 基础版本 |
| IterativePFN (BatchNorm) | 0.004430 | 0.072046 | 0.5406 | 旧版本，不推荐 |

> **核心发现**: IterativePFN + LayerNorm 综合性能最优，CD 比噪声基数低 **55%**，GR（几何保留率）提升 **30%**

### 分形状评估

| 形状 | IPFN LN CD | IPFN LN GR | 最佳方法 |
|:----:|:----------:|:----------:|:--------:|
| Sphere | 0.00198 | 0.84 | IPFN LN |
| Cube | 0.00082 | 0.99 | IPFN LN |
| Cone | 0.00108 | 0.97 | IPFN LN |

### 传统方法对比

| 方法 | Mean CD ↓ | Mean GR ↑ | 备注 |
|:----:|:---------:|:---------:|:-----|
| SOR | 0.002992 | 0.7045 | 统计异常值去除 |
| MLS | 0.003573 | **0.9523** | 移动最小二乘法（不适用球面） |
| Gaussian | 0.003451 | 0.6171 | 高斯滤波 |
| ROR | 0.044945 | 0.3240 |半径异常值去除（效果较差） |

### LayerNorm 优化结论

| 模型 | LayerNorm 适用 | 效果 |
|:----|:-------------:|:-----|
| IterativePFN | ✅ **是** | GR +73%, CD -71% |
| StraightPCF | ❌ 否 | GR -99%，不适用 |

> ⚠️ LayerNorm 替换 BatchNorm 对 IterativePFN 有显著改善，但对 StraightPCF 导致 GR 崩溃。

## 项目结构

```
plant_denoising/
├── checkpoints/              # 模型检查点（不包含在仓库中）
├── data/                     # 数据
│   ├── synthetic_dataset.pkl
│   └── synthetic_data_generator.py
├── evaluation/               # 评估模块
│   └── metrics.py
├── models/                   # 模型实现
│   ├── iterative_pfn_improved.py   # IterativePFN（LayerNorm优化版）
│   ├── pointfilter.py              # PointFilter
│   └── straight_pcf_improved.py    # StraightPCF
├── results/                  # 结果和可视化
│   ├── comprehensive_evaluation.json
│   ├── evaluation_report_updated.html
│   └── *.png                   # 可视化图表
├── train_ipfn_ln.py          # IPFN LayerNorm训练脚本
├── train_spcf_ln.py          # StraightPCF训练脚本
├── evaluate_comprehensive.py  # 综合评估脚本
└── README.md
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
# 训练 IterativePFN (LayerNorm优化版)
python train_ipfn_ln.py

# 训练 PointFilter
python train_complete.py

# 训练 StraightPCF
python train_spcf_mixed.py
```

### 综合评估
```bash
python evaluate_comprehensive.py
```

## 模型说明

### IterativePFN (LayerNorm) ⭐ 最优
- **特点**: 迭代优化 + LayerNorm 归一化
- **优势**: Chamfer Distance 表现最佳，GR 最高
- **适用**: 对几何精度和保留率要求高的场景

### PointFilter
- **特点**: 端到端滤波网络
- **优势**: 训练稳定，综合表现良好
- **适用**: 需要平衡速度和精度的场景

### StraightPCF
- **特点**: 直接预测位移向量
- **优势**: 模型简单，推理速度快
- **适用**: 实时性要求高的场景

## 技术细节

### LayerNorm 替换 BatchNorm

在 Conv1d 输出 `(B,C,N)` 上应用 LayerNorm：

```python
# LayerNorm 在 (B,N,C) 上归一化通道 C
x = self.conv1(x)
x = self.ln1(x.transpose(2, 1)).transpose(2, 1)  # LN
x = F.relu(x)
```

**适用性**：仅对 IterativePFN 有效，StraightPCF 不适用。

## 引用

- PointFilter: [PointFilter paper]
- IterativePFN: [IterativePFN paper]
- StraightPCF: [StraightPCF paper]
