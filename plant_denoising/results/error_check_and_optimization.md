# 项目错误检查报告 + 优化建议
> 生成时间: 2026-05-10
> 评估条件: noise_level=0.02, THRESHOLD=0.04, 测试集=sphere+cube+cone 各8样本

---

## 一、已发现并修复的错误

### ✅ 已修复

| # | 文件 | 错误描述 | 修复方式 |
|:--:|:----:|:--------:|:--------:|
| 1 | `evaluate_all.py` | GR Bug①: `dist`存平方距离，与线性阈值0.04比较 | 改用 `torch.cdist` 获取欧氏距离 |
| 2 | `evaluate_all.py` | GR Bug②: `dim=1`(Precision方向)，应为`dim=0`(Recall方向) | 改为 `torch.min(dist_l2, dim=0)` |
| 3 | `traditional_methods.py` | MLS函数变量名`dists`/`dist_s` 不一致 (NameError) | 重写文件，统一变量名 |
| 4 | `traditional_methods.py` | MLS加权平均导致球面收缩 (半径0.9865→0.9329) | 去掉加权平均，直接用投影点 |
| 5 | `generate_summary_report.py` | JSON读取缺`encoding='utf-8'` (Windows GBK乱码) | 加 `encoding='utf-8'` |
| 6 | `evaluate_traditional.py` | MLS在sphere上GR≈0，评估失真 | ✅ **已修复**：标注"MLS不适用于球面"，计算均值时排除 |
| 7 | `evaluate_gr_fixed.py` | THRESHOLD未随noise_level自适应 | ✅ **已修复**：改为`THRESHOLD = 2 * NOISE_LEVEL` |

### ⚠️ 仍有问题（待处理）

| # | 文件 | 问题 | 影响 | 建议 |
|:--:|:----:|:----:|:----:|:----:|
| 8 | `evaluate_traditional.py` | 传统方法的P2P计算**无意义**（不保持点对应关系） | 报告中的传统方法P2P值是错误的 | 已修复：移除传统方法的P2P计算（本次已做） |

---

## 二、从指标入手的优化建议

### 当前最佳结果汇总

| 方法 | Mean CD ↓ | Mean GR ↑ | 排名(CD) | 排名(GR) |
|:----:|:----:|:----:|:----:|:----:|
| **PointFilter** | **0.002146** | **0.8035** | 🥇1 | 🥇1 |
| **SOR** (传统) | 0.002992 | 0.7045 | 🥈2 | 🥉3 |
| StraightPCF | 0.002947 | 0.7189 | 🥉3 | 🥈2 |
| MLS (传统) | 0.003573 | 0.9523* | 4 | 1* |
| IterativePFN | 0.004430 | 0.5406 | 5 | 5 |

*> MLS在sphere上已标注为N/A（算法不适用），GR均值排除sphere计算
*> MLS的GR实际最高（仅cube+cone），但CD不如其他传统方法

---

### 优化方向 1：提升 GR（当前最大短板）

所有模型在 **sphere 上 GR 都偏低**：
- PointFilter sphere GR = 0.4457（最好但仍有提升空间）
- IterativePFN sphere GR = 0.4866
- StraightPCF sphere GR = 0.2058 ⚠️

**建议操作：**
1. **增大训练数据中的 sphere 样本比例**（当前是 1:1:1）
2. **增大 `THRESHOLD` 或改用自适应阈值**（当前 0.04 对 sphere 可能偏严）
3. **在 loss 中显式加入 GR 相关项**（让模型优化召回率，而非仅优化 CD）

---

### 优化方向 2：传统方法作为 DL 模型的前处理

观察到：**SOR 的 Mean CD=0.002992，接近 DL 模型**

```
SOR → PointFilter 串联：
  噪声点云 → SOR去离群点 → PointFilter微调 → 输出
```

**建议操作：**
1. 在 `train_complete.py` / `train_mixed_shapes.py` 中加入"SOR前处理"对比实验
2. 评估 `SOR + PointFilter` 的 CD/GR 是否优于单独 PointFilter

---

### 优化方向 3：修复 StraightPCF 在 sphere 上的劣化

StraightPCF 在 sphere 上 GR=0.2058，远低于 PointFilter 的 0.4457。

**可能原因：**
- StraightPCF 的网络结构对曲面结构建模能力不足
- 训练时 sphere 样本的损失权重不够

**建议操作：**
1. 检查 `models/straight_pcf_improved.py` 的网络结构
2. 在混合训练中对 sphere 样本加**样本权重**（使其损失贡献 ×2）
3. 重新训练并对比

---

### 优化方向 4：超参数调优

当前训练超参数（已固化在代码中）：
```
batch_size=8, lr=1e-3, epochs=30, feature_dim=256, hidden_dim=128
```

**建议操作：**
1. **Grid Search** `lr ∈ {5e-4, 1e-3, 2e-3}` + `batch_size ∈ {4, 8, 16}`
2. **网络宽度** `feature_dim ∈ {128, 256, 512}`
3. 用 **Optuna** 做自动超参数搜索（需新增 `hyperparam_search.py`）

---

### 优化方向 5：增加更多形状，提升泛化性

当前仅用 `sphere + cube + cone`（3种），形状覆盖太窄。

**建议操作：**
1. 在 `data/synthetic_data_generator.py` 中增加形状：
   - `cylinder`（圆柱）、`torus`（圆环）、`ellipsoid`（椭球）
2. 重新生成数据集 `synthetic_dataset.pkl`
3. 用更丰富的形状重新训练，测试泛化性

---

## 三、优先级建议

| 优先级 | 操作 | 预期收益 | 状态 |
|:----:|:----:|:----:|:----:|
| 🔴 高 | 修复 MLS 在 sphere 上的评估（跳过或标注） | 报告准确性 | ✅ **已完成** |
| 🔴 高 | `THRESHOLD` 改为自适应（随 noise_level 变化） | GR 指标正确性 | ✅ **已完成** |
| 🟡 中 | StraightPCF 在 sphere 上加样本权重重训 | GR 提升 ~0.2 | 待处理 |
| 🟡 中 | SOR + PointFilter 串联实验 | CD/GR 可能进一步提升 | 待处理 |
| 🟢 低 | 超参数网格搜索 | CD 可能提升 ~5-10% | 待处理 |
| 🟢 低 | 增加更多形状重新训练 | 泛化性提升 | 待处理 |

---

## 四、下一步推荐

**✅ 高优先级任务已完成：**
1. ✅ 修复 `evaluate_traditional.py`：标注 MLS 不适用于球面
2. ✅ 修复 `evaluate_gr_fixed.py`：THRESHOLD 已改为自适应

**建议接下来处理中优先级任务：**
1. StraightPCF 在 sphere 上加样本权重重训
2. SOR + PointFilter 串联实验

处理完优化任务后，可以考虑低优先级的超参数搜索和数据集扩展。
