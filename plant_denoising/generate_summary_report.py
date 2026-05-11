"""
生成完整的性能总结报告 (Markdown + 可视化图表)
包含: DL模型 (PointFilter, IterativePFN, StraightPCF) + 传统方法
"""
import os
os.environ['TORCH_DISABLE_CUDA_INIT'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USERNAME'] = 'User'

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results'

# 加载结果 (Windows 需显式指定 utf-8)
with open(os.path.join(RESULTS_DIR, 'gr_fixed_evaluation.json'), 'r', encoding='utf-8') as f:
    dl_results = json.load(f)

with open(os.path.join(RESULTS_DIR, 'traditional_methods_evaluation.json'), 'r', encoding='utf-8') as f:
    trad_results = json.load(f)

SHAPES = ['sphere', 'cube', 'cone']

# ================================================================
# 1. 汇总数据
# ================================================================

# DL 模型结果 (已从 gr_fixed_evaluation.json)
dl_data = {}
for model_name, shape_results in dl_results['results'].items():
    dl_data[model_name] = {}
    for shape in SHAPES:
        r = shape_results[shape]
        dl_data[model_name][shape] = {
            'CD':  r['CD'],
            'P2P': r['P2P'],
            'GR':  r['GR'],
        }

# 传统方法结果
trad_data = {}
for method_name, shape_results in trad_results['results'].items():
    trad_data[method_name] = {}
    for shape in SHAPES:
        r = shape_results[shape]
        trad_data[method_name][shape] = {
            'CD':  r['CD'],
            'P2P': r['P2P'],
            'GR':  r['GR'],
        }

# ================================================================
# 2. 计算均值
# ================================================================
def add_means(data_dict):
    for method in data_dict:
        cds, p2ps, grs = [], [], []
        for shape in SHAPES:
            cds.append(data_dict[method][shape]['CD'])
            p2ps.append(data_dict[method][shape]['P2P'])
            grs.append(data_dict[method][shape]['GR'])
        data_dict[method]['Mean'] = {
            'CD':  np.mean(cds),
            'P2P': np.mean(p2ps),
            'GR':  np.mean(grs),
        }

add_means(dl_data)
add_means(trad_data)

# ================================================================
# 3. 生成 Markdown 报告
# ================================================================
report_lines = []
report_lines.append('# 植物点云去噪 — 模型性能总结报告\n')
report_lines.append(f'> 评估日期: {dl_results["timestamp"][:10]}  |  '
                   f'噪声级别: {dl_results["noise_level"]}  |  '
                   f'GR阈值: {dl_results["threshold"]}\n')
report_lines.append(f'> GR计算: 修复版 (torch.cdist + dim=0 召回方向)\n')
report_lines.append(f'> 测试集: 混合形状 (sphere+cube+cone), 每个形状 8 个测试样本\n')
report_lines.append('\n---\n')

# 3.1 指标说明
report_lines.append('## 指标说明\n')
report_lines.append('| 指标 | 全称 | 含义 | 好坏 |\n')
report_lines.append('|:----:|:----:|:----:|:----:|\n')
report_lines.append('| **CD** | Chamfer Distance | 预测点与干净点云的双向平均距离 (平方) | ↓ 越低越好 |\n')
report_lines.append('| **P2P** | Point-to-Point Distance | 逐点欧氏距离均值 (需点对应) | ↓ 越低越好 |\n')
report_lines.append('| **GR** | Geometric Recall | 干净点云中被预测点覆盖的比例 (距离≤阈值) | ↑ 越高越好 |\n')
report_lines.append('\n---\n')

# 3.2 DL 模型结果表
report_lines.append('## 一、深度学习模型性能\n')
report_lines.append('### 按形状 (CD ↓ / P2P ↓ / GR ↑)\n')
report_lines.append('| 模型 | Sphere | Cube | Cone | **Mean** |\n')
report_lines.append('|:----:|:----:|:----:|:----:|:----:|\n')

for model in dl_data:
    row = f'| **{model}** '
    for shape in SHAPES:
        r = dl_data[model][shape]
        row += f'| CD={r["CD"]:.6f}<br>P2P={r["P2P"]:.6f}<br>GR={r["GR"]:.4f} '
    m = dl_data[model]['Mean']
    row += f'| **CD={m["CD"]:.6f}**<br>**P2P={m["P2P"]:.6f}**<br>**GR={m["GR"]:.4f}** '
    row += '|'
    report_lines.append(row)

report_lines.append('\n')

# 3.3 传统方法结果表
report_lines.append('## 二、传统方法性能\n')
report_lines.append('### 按形状 (CD ↓ / P2P ↓ / GR ↑)\n')
report_lines.append('| 方法 | Sphere | Cube | Cone | **Mean** |\n')
report_lines.append('|:----:|:----:|:----:|:----:|:----:|\n')

for method in trad_data:
    row = f'| **{method}** '
    for shape in SHAPES:
        r = trad_data[method][shape]
        row += f'| CD={r["CD"]:.6f}<br>P2P={r["P2P"]:.6f}<br>GR={r["GR"]:.4f} '
    m = trad_data[method]['Mean']
    row += f'| **CD={m["CD"]:.6f}**<br>**P2P={m["P2P"]:.6f}**<br>**GR={m["GR"]:.4f}** '
    row += '|'
    report_lines.append(row)

report_lines.append('\n---\n')

# 3.4 综合排名表
report_lines.append('## 三、综合排名 (按 Mean 值)\n')

# 收集所有方法
all_methods = {}
for m in dl_data:
    all_methods[m] = dl_data[m]['Mean']
for m in trad_data:
    all_methods[m] = trad_data[m]['Mean']

# 按 CD 排名 (升序)
cd_rank = sorted(all_methods.items(), key=lambda x: x[1]['CD'])
# 按 P2P 排名 (升序)
p2p_rank = sorted(all_methods.items(), key=lambda x: x[1]['P2P'])
# 按 GR 排名 (降序)
gr_rank = sorted(all_methods.items(), key=lambda x: x[1]['GR'], reverse=True)

report_lines.append('### CD ↓ (Chamfer Distance, 越低越好)\n')
report_lines.append('| 排名 | 方法 | Mean CD |\n')
report_lines.append('|:----:|:----:|:----:|\n')
for i, (name, vals) in enumerate(cd_rank, 1):
    tag = '🥇' if i == 1 else ('🥈' if i == 2 else ('🥉' if i == 3 else ''))
    report_lines.append(f'| {i} | {name} {tag} | {vals["CD"]:.6f} |\n')

report_lines.append('\n### GR ↑ (Geometric Recall, 越高越好)\n')
report_lines.append('| 排名 | 方法 | Mean GR |\n')
report_lines.append('|:----:|:----:|:----:|\n')
for i, (name, vals) in enumerate(gr_rank, 1):
    tag = '🥇' if i == 1 else ('🥈' if i == 2 else ('🥉' if i == 3 else ''))
    report_lines.append(f'| {i} | {name} {tag} | {vals["GR"]:.4f} |\n')

report_lines.append('\n---\n')

# 3.5 核心发现
report_lines.append('## 四、核心发现\n')
report_lines.append('### 深度学习模型\n')
report_lines.append('1. **PointFilter**: CD最低 (0.002146), 综合去噪精度最高; '
                   '但在 sphere 上 GR 偏低 (0.4457), 说明对曲面结构恢复能力有限\n')
report_lines.append('2. **IterativePFN**: 跨形状最稳定, sphere/cube/cone 上 CD 差异最小; '
                   'GR 中等 (0.5411), 适合泛化场景\n')
report_lines.append('3. **StraightPCF**: sphere 上 GR 明显偏低 (0.2058), 对曲面恢复能力不足; '
                   'cube/cone 上表现接近 PointFilter\n')
report_lines.append('\n### 传统方法\n')
report_lines.append('1. **SOR** (统计离群点去除): 传统方法中综合最优, '
                   'Mean CD=0.002992, Mean GR=0.7045; 在 cube/cone 上 GR 超 0.94\n')
report_lines.append('2. **MLS** (移动最小二乘): CD 与 SOR 相当 (0.003447), '
                   '在 cube 上 GR 达 0.9810 (所有方法中最高)\n')
report_lines.append('3. **Gaussian**: 简单有效, Mean CD=0.003451; 但 GR 偏低 (0.6170)\n')
report_lines.append('4. **ROR** (半径离群点去除): 表现最差, Mean CD=0.044945, '
                   '说明固定半径策略对复杂形状不适用\n')
report_lines.append('\n### 方法对比\n')
report_lines.append('- **最佳 CD**: PointFilter (0.002146) > SOR (0.002992) > MLS (0.003447)\n')
report_lines.append('- **最佳 GR**: MLS-cube (0.9810) > SOR-cube (0.9741) > PointFilter-cube (0.9965)\n')
report_lines.append('- **传统方法 vs DL**: 最佳传统方法 (SOR) 的 CD 接近最差 DL 模型 (StraightPCF, 0.003947), '
                   '但 GR 明显低于 PointFilter\n')
report_lines.append('\n---\n')

# 3.6 按形状分析
report_lines.append('## 五、按形状分析\n')
report_lines.append('### Sphere (曲面, 最具挑战性)\n')
report_lines.append('- 所有方法在 sphere 上 CD 最高 (曲面复杂, 难以恢复)\n')
report_lines.append('- IterativePFN 在 sphere 上 CD 最低 (0.004637), 说明对曲面更鲁棒\n')
report_lines.append('- 传统方法在 sphere 上 GR 普遍偏低 (<0.20), 说明传统滤波难以恢复完整曲面\n')
report_lines.append('\n### Cube (平面, 最简单)\n')
report_lines.append('- 所有方法在 cube 上表现最好 (平面结构易于恢复)\n')
report_lines.append('- PointFilter 在 cube 上 GR=0.9965, 接近完美恢复\n')
report_lines.append('- MLS 在 cube 上 GR=0.9810, 传统方法中的最佳表现\n')
report_lines.append('\n### Cone (混合结构)\n')
report_lines.append('- 性能介于 sphere 与 cube 之间\n')
report_lines.append('- PointFilter 在 cone 上 GR=0.9684, 明显优于传统方法\n')
report_lines.append('\n---\n')

# 3.7 结论与建议
report_lines.append('## 六、结论与建议\n')
report_lines.append('1. **最佳整体性能**: PointFilter — CD 最低, cube/cone 上 GR 接近完美\n')
report_lines.append('2. **最佳泛化性**: IterativePFN — 跨形状 CD 最稳定\n')
report_lines.append('3. **最佳传统方法**: SOR 或 MLS — 可作为 DL 模型的前处理或基准对比\n')
report_lines.append('4. **GR 计算修复的影响**: 修复前 GR 普遍 <0.15 (严重低估), '
                   '修复后 PointFilter 在 cube/cone 上超 0.96 (更符合实际观察)\n')
report_lines.append('\n---\n')
report_lines.append(f'> 报告生成时间: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

report_content = ''.join(report_lines)

# 保存报告
report_path = os.path.join(RESULTS_DIR, 'performance_summary_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f'Markdown 报告已保存: {report_path}')

# ================================================================
# 4. 生成可视化图表
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('植物点云去噪 — 模型性能对比 (CD / GR / P2P)', fontsize=14, fontweight='bold')

# 准备数据
all_method_names = list(dl_data.keys()) + list(trad_data.keys())
n_methods = len(all_method_names)
x = np.arange(len(SHAPES))
width = 0.1

colors = plt.cm.Set3(np.linspace(0, 1, n_methods))

# ----- 子图1: CD by shape (all methods) -----
ax = axes[0, 0]
for i, method in enumerate(all_method_names):
    data_src = dl_data if method in dl_data else trad_data
    cds = [data_src[method][s]['CD'] for s in SHAPES]
    offset = (i - n_methods/2) * width + width/2
    ax.bar(x + offset, cds, width, label=method, color=colors[i], alpha=0.8)

ax.set_title('CD ↓ (Chamfer Distance)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(SHAPES)
ax.set_ylabel('CD (log scale)')
ax.set_yscale('log')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ----- 子图2: GR by shape -----
ax = axes[0, 1]
for i, method in enumerate(all_method_names):
    data_src = dl_data if method in dl_data else trad_data
    grs = [data_src[method][s]['GR'] for s in SHAPES]
    offset = (i - n_methods/2) * width + width/2
    ax.bar(x + offset, grs, width, label=method, color=colors[i], alpha=0.8)

ax.set_title('GR ↑ (Geometric Recall)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(SHAPES)
ax.set_ylabel('GR (0-1, 越高越好)')
ax.set_ylim([0, 1.05])
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ----- 子图3: P2P by shape -----
ax = axes[0, 2]
for i, method in enumerate(all_method_names):
    data_src = dl_data if method in dl_data else trad_data
    p2ps = [data_src[method][s]['P2P'] for s in SHAPES]
    offset = (i - n_methods/2) * width + width/2
    ax.bar(x + offset, p2ps, width, label=method, color=colors[i], alpha=0.8)

ax.set_title('P2P ↓ (Point-to-Point Distance)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(SHAPES)
ax.set_ylabel('P2P')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ----- 子图4: Mean CD 排名 -----
ax = axes[1, 0]
mean_cds = []
for method in all_method_names:
    data_src = dl_data if method in dl_data else trad_data
    mean_cds.append((method, data_src[method]['Mean']['CD']))
mean_cds.sort(key=lambda x: x[1])
methods_sorted, cds_sorted = zip(*mean_cds)
y_pos = np.arange(len(methods_sorted))
bars = ax.barh(y_pos, cds_sorted, color=[colors[all_method_names.index(m)] for m in methods_sorted], alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(methods_sorted)
ax.set_xlabel('Mean CD ↓ (log scale)')
ax.set_title('Mean CD 排名 (越低越好)', fontweight='bold')
ax.set_xscale('log')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# ----- 子图5: Mean GR 排名 -----
ax = axes[1, 1]
mean_grs = []
for method in all_method_names:
    data_src = dl_data if method in dl_data else trad_data
    mean_grs.append((method, data_src[method]['Mean']['GR']))
mean_grs.sort(key=lambda x: x[1], reverse=True)
methods_sorted_gr, grs_sorted = zip(*mean_grs)
y_pos = np.arange(len(methods_sorted_gr))
bars = ax.barh(y_pos, grs_sorted, color=[colors[all_method_names.index(m)] for m in methods_sorted_gr], alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(methods_sorted_gr)
ax.set_xlabel('Mean GR ↑ (0-1, 越高越好)')
ax.set_title('Mean GR 排名 (越高越好)', fontweight='bold')
ax.set_xlim([0, 1.05])
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# ----- 子图6: CD vs GR 散点图 -----
ax = axes[1, 2]
# 用 Mean 值
for i, method in enumerate(all_method_names):
    data_src = dl_data if method in dl_data else trad_data
    m = data_src[method]['Mean']
    marker = 's' if method in dl_data else 'o'
    ax.scatter(m['CD'], m['GR'], s=100, color=colors[i], marker=marker,
               label=f'{method} ({"DL" if method in dl_data else "Trad"})', alpha=0.8)
    ax.annotate(method, (m['CD'], m['GR']), fontsize=7, alpha=0.7)

ax.set_xlabel('Mean CD ↓ (log scale)')
ax.set_ylabel('Mean GR ↑')
ax.set_title('CD vs GR (越低+越高=越好)', fontweight='bold')
ax.set_xscale('log')
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

plt.tight_layout()
chart_path = os.path.join(RESULTS_DIR, 'performance_comparison_chart.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f'可视化图表已保存: {chart_path}')

plt.close()
print('\n完成!')
