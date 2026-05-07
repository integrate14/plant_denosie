"""
更新可视化 - 匹配新的JSON格式
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')
import os

JSON_PATH = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results\experiment_results.json'
VIS_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results\visualizations'
os.makedirs(VIS_DIR, exist_ok=True)

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

models = data['models']

COLOR = {
    'PointFilter':  '#CE93D8',
    'IterativePFN': '#F48FB1',
    'StraightPCF':  '#80DEEA',
    'Bilateral':    '#4FC3F7',
    'Laplacian':    '#81C784',
}

BG   = '#1a1a2e'
CARD = '#16213e'
AXES = '#0f3460'
TXT  = '#e0e0e0'

all_methods = ['PointFilter', 'IterativePFN', 'StraightPCF', 'Bilateral', 'Laplacian']
colors = [COLOR.get(m, '#888888') for m in all_methods]

# 过滤掉 NaN 的方法
valid_methods = [m for m in all_methods if not np.isnan(models[m]['chamfer'])]
valid_colors = [COLOR.get(m, '#888888') for m in valid_methods]

def styled_ax(ax, title=''):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(AXES)
    if title:
        ax.set_title(title, color=TXT, fontsize=11, pad=8, fontweight='bold')

# 准备数据
cd_vals = [models[m]['chamfer'] for m in all_methods]
p2p_vals = [models[m]['p2p'] for m in all_methods]
gr_vals = [models[m]['geometric_recall'] for m in all_methods]

# 过滤 NaN
valid_cd = [models[m]['chamfer'] for m in valid_methods]
valid_p2p = [models[m]['p2p'] for m in valid_methods]
valid_gr = [models[m]['geometric_recall'] for m in valid_methods]

# ------------------------------------------------------------------
# 1. dashboard.png
# ------------------------------------------------------------------
fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

# (0,0) CD
ax0 = fig.add_subplot(gs[0, 0])
styled_ax(ax0, 'Chamfer Distance (lower is better)')
bars0 = ax0.bar(range(len(valid_methods)), valid_cd, color=valid_colors, edgecolor='#333', linewidth=0.5)
ax0.set_xticks(range(len(valid_methods)))
ax0.set_xticklabels(valid_methods, rotation=30, ha='right', fontsize=9)
ax0.set_ylabel('CD', color=TXT)
for b, v in zip(bars0, valid_cd):
    ax0.text(b.get_x() + b.get_width() / 2, v + 0.00005,
             f'{v:.5f}', ha='center', va='bottom', color=TXT, fontsize=8)

# (0,1) P2P
ax1 = fig.add_subplot(gs[0, 1])
styled_ax(ax1, 'Point-to-Point Distance (lower is better)')
bars1 = ax1.bar(range(len(valid_methods)), valid_p2p, color=valid_colors, edgecolor='#333', linewidth=0.5)
ax1.set_xticks(range(len(valid_methods)))
ax1.set_xticklabels(valid_methods, rotation=30, ha='right', fontsize=9)
ax1.set_ylabel('P2P Dist', color=TXT)
for b, v in zip(bars1, valid_p2p):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.001,
             f'{v:.4f}', ha='center', va='bottom', color=TXT, fontsize=8)

# (0,2) Geometric Recall
ax2 = fig.add_subplot(gs[0, 2])
styled_ax(ax2, 'Geometric Recall (higher is better)')
bars2 = ax2.bar(range(len(valid_methods)), valid_gr, color=valid_colors, edgecolor='#333', linewidth=0.5)
ax2.set_xticks(range(len(valid_methods)))
ax2.set_xticklabels(valid_methods, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel('Recall', color=TXT)
ax2.set_ylim(0.98, 1.002)
for b, v in zip(bars2, valid_gr):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.0002,
             f'{v:.4f}', ha='center', va='bottom', color=TXT, fontsize=8)

# (1,0) DL vs Traditional CD
ax3 = fig.add_subplot(gs[1, 0])
styled_ax(ax3, 'Deep Learning vs Traditional (CD)')
dl_methods = ['PointFilter', 'IterativePFN', 'StraightPCF']
trad_methods = ['Bilateral', 'Laplacian']
dl_cd = [models[m]['chamfer'] for m in dl_methods]
trad_cd = [models[m]['chamfer'] for m in trad_methods if not np.isnan(models[m]['chamfer'])]
x = np.arange(len(dl_methods))
width = 0.35
bars_dl = ax3.bar(x - width/2, dl_cd, width, label='Deep Learning', color=[COLOR[m] for m in dl_methods])
bars_trad = ax3.bar(x + width/2, trad_cd, width, label='Traditional', color=[COLOR.get(m, '#888') for m in trad_methods if not np.isnan(models[m]['chamfer'])])
ax3.set_xticks(x)
ax3.set_xticklabels(dl_methods, fontsize=9)
ax3.set_ylabel('CD', color=TXT)
ax3.legend(fontsize=8, facecolor=CARD, edgecolor=AXES, labelcolor=TXT)

# (1,1) Radar Chart (DL only)
ax4 = fig.add_subplot(gs[1, 1], projection='polar')
ax4.set_facecolor(CARD)
ax4.tick_params(colors=TXT, labelsize=8)
ax4.spines['polar'].set_color(AXES)
categories = ['CD (inv)', 'P2P (inv)', 'Geo Recall']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

cd_dl = [models[m]['chamfer'] for m in dl_methods]
p2p_dl = [models[m]['p2p'] for m in dl_methods]
gr_dl = [models[m]['geometric_recall'] for m in dl_methods]
inv_cd = [1 / c for c in cd_dl]; max_icd = max(inv_cd)
inv_p2p = [1 / p for p in p2p_dl]; max_ip = max(inv_p2p)
max_gr = max(gr_dl)

for j, m in enumerate(dl_methods):
    vals = [inv_cd[j] / max_icd, inv_p2p[j] / max_ip, gr_dl[j] / max_gr]
    vals += vals[:1]
    ax4.plot(angles, vals, color=COLOR[m], linewidth=2, label=m)
    ax4.fill(angles, vals, color=COLOR[m], alpha=0.15)

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, color=TXT, fontsize=9)
ax4.set_yticks([0.25, 0.5, 0.75, 1.0])
ax4.set_yticklabels(['25%', '50%', '75%', '100%'], color=TXT, fontsize=7)
ax4.set_title('DL Models Radar', color=TXT, fontsize=11, pad=15, fontweight='bold')
ax4.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
           fontsize=7, facecolor=CARD, edgecolor=AXES, labelcolor=TXT)

# (1,2) Summary Table
ax5 = fig.add_subplot(gs[1, 2])
styled_ax(ax5, 'Summary Table')
ax5.axis('off')
rows = [['Model', 'CD', 'P2P', 'GR']]
for m in valid_methods:
    rows.append([m, f"{models[m]['chamfer']:.6f}", f"{models[m]['p2p']:.6f}", f"{models[m]['geometric_recall']:.4f}"])
tbl = ax5.table(cellText=rows[1:], colLabels=rows[0],
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for key, cell in tbl.get_celld().items():
    cell.set_facecolor(CARD if key[0] % 2 == 0 else AXES)
    cell.set_text_props(color=TXT)
    cell.set_edgecolor(AXES)

fig.text(0.5, 0.97,
         'Plant Point Cloud Denoising — Updated Results',
         ha='center', va='top', color='#80DEEA', fontsize=15, fontweight='bold')
plt.savefig(VIS_DIR + '\\dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('dashboard.png saved')

# ------------------------------------------------------------------
# 2. model_comparison.png
# ------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
metric_info = [
    ('Chamfer Distance (lower is better)', valid_cd, False),
    ('P2P Distance (lower is better)', valid_p2p, False),
    ('Geometric Recall (higher is better)', valid_gr, True),
]

for ax, (title, vals, higher) in zip(axes3, metric_info):
    styled_ax(ax, title)
    bars = ax.bar(range(len(valid_methods)), vals, color=valid_colors, edgecolor='#222', linewidth=0.5)
    ax.set_xticks(range(len(valid_methods)))
    ax.set_xticklabels(valid_methods, rotation=30, ha='right', fontsize=9)
    best_idx = int(np.argmax(vals)) if higher else int(np.argmin(vals))
    for k, (b, v) in enumerate(zip(bars, vals)):
        fc_c = '#FFD700' if k == best_idx else TXT
        fw = 'bold' if k == best_idx else 'normal'
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02 if higher else v + max(vals) * 0.005,
                f'{v:.4f}', ha='center', va='bottom', color=fc_c, fontsize=8, fontweight=fw)
    if higher:
        ax.set_ylabel('Recall Rate', color=TXT)
        ax.set_ylim(0.98, 1.005)
    else:
        ax.set_ylabel('Distance', color=TXT)

fig3.text(0.5, 0.98,
          'Model Comparison — Three Metrics (Gold = Best)',
          ha='center', va='top', color='#80DEEA', fontsize=13, fontweight='bold')
fig3.patch.set_facecolor(BG)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(VIS_DIR + '\\model_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('model_comparison.png saved')

print('\nAll visualizations updated successfully!')
