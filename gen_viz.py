"""
重新生成可视化（含几何召回率）
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

JSON_PATH = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results\experiment_results.json'
VIS_DIR   = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results\visualizations' + '\\'

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

avg     = data['average_results']
fc      = data['final_comparison']['models']
records = data['individual_results']

COLOR = {
    'BilateralFilter':  '#4FC3F7',
    'LaplacianSmooth':  '#81C784',
    'IterativeDenoise': '#FFB74D',
    'PointFilter':      '#CE93D8',
    'IterativePFN':     '#F48FB1',
    'StraightPCF':      '#80DEEA',
}
BG   = '#1a1a2e'
CARD = '#16213e'
AXES = '#0f3460'
TXT  = '#e0e0e0'

methods_trad = ['BilateralFilter', 'LaplacianSmooth', 'IterativeDenoise']
dl_methods   = ['PointFilter', 'IterativePFN', 'StraightPCF']
all_methods  = methods_trad + dl_methods
colors       = [COLOR[m] for m in all_methods]


def styled_ax(ax, title=''):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(AXES)
    if title:
        ax.set_title(title, color=TXT, fontsize=11, pad=8, fontweight='bold')
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)


def norm_max(v):
    m = max(v)
    return [x / m for x in v]


# ------------------------------------------------------------------
# 准备数据
# ------------------------------------------------------------------
cd_vals  = [avg[m]['cd'] for m in methods_trad] + [fc[m]['chamfer'] for m in dl_methods]
p2p_vals = [np.mean([r[m + '_p2p'] for r in records if m + '_p2p' in r])
             for m in methods_trad] + [fc[m]['ptp'] for m in dl_methods]
gr_vals  = [avg[m]['gr'] for m in methods_trad] + \
           [fc[m]['geometric_recall'] for m in dl_methods]

# ------------------------------------------------------------------
# 1. dashboard.png
# ------------------------------------------------------------------
fig = plt.figure(figsize=(18, 12), facecolor=BG)
gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

# (0,0) CD
ax0 = fig.add_subplot(gs[0, 0])
styled_ax(ax0, 'Chamfer Distance (lower is better)')
bars0 = ax0.bar(range(len(all_methods)), cd_vals, color=colors, edgecolor='#333', linewidth=0.5)
ax0.set_xticks(range(len(all_methods)))
ax0.set_xticklabels(['Bilateral', 'Laplacian', 'Iterative',
                     'PointFilter', 'IterPFN', 'StraightPCF'],
                    rotation=30, ha='right', fontsize=8)
ax0.set_ylabel('CD', color=TXT)
for b, v in zip(bars0, cd_vals):
    ax0.text(b.get_x() + b.get_width() / 2, v + 0.0001,
             f'{v:.4f}', ha='center', va='bottom', color=TXT, fontsize=7)

# (0,1) P2P
ax1 = fig.add_subplot(gs[0, 1])
styled_ax(ax1, 'Point-to-Point Distance (lower is better)')
bars1 = ax1.bar(range(len(all_methods)), p2p_vals, color=colors, edgecolor='#333', linewidth=0.5)
ax1.set_xticks(range(len(all_methods)))
ax1.set_xticklabels(['Bilateral', 'Laplacian', 'Iterative',
                     'PointFilter', 'IterPFN', 'StraightPCF'],
                    rotation=30, ha='right', fontsize=8)
ax1.set_ylabel('P2P Dist', color=TXT)
for b, v in zip(bars1, p2p_vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.001,
             f'{v:.4f}', ha='center', va='bottom', color=TXT, fontsize=7)

# (0,2) Geometric Recall ★ NEW ★
ax2 = fig.add_subplot(gs[0, 2])
styled_ax(ax2, 'Geometric Recall (higher is better)')
bars2 = ax2.bar(range(len(all_methods)), gr_vals, color=colors, edgecolor='#333', linewidth=0.5)
ax2.set_xticks(range(len(all_methods)))
ax2.set_xticklabels(['Bilateral', 'Laplacian', 'Iterative',
                     'PointFilter', 'IterPFN', 'StraightPCF'],
                    rotation=30, ha='right', fontsize=8)
ax2.set_ylabel('Recall', color=TXT)
ax2.set_ylim(0, min(1.0, max(gr_vals) * 1.35 + 0.02))
for b, v in zip(bars2, gr_vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.003,
             f'{v:.3f}', ha='center', va='bottom', color=TXT, fontsize=7)

# (1,0) CD vs Noise Level
ax3 = fig.add_subplot(gs[1, 0])
styled_ax(ax3, 'CD vs Noise Level (Traditional)')
nl_bins = [0.01, 0.02, 0.05, 0.1]
for method in methods_trad:
    cd_by_nl = {}
    for rec in records:
        nl_match = min(nl_bins, key=lambda nl: abs(rec['noise_level'] - nl))
        cd_by_nl.setdefault(nl_match, []).append(rec[method + '_cd'])
    sorted_nl = sorted(cd_by_nl.keys())
    ax3.plot(sorted_nl, [np.mean(cd_by_nl[nl]) for nl in sorted_nl],
             marker='o', color=COLOR[method], label=method, linewidth=1.5)
ax3.set_xlabel('Noise Level', color=TXT)
ax3.set_ylabel('Mean CD', color=TXT)
ax3.legend(fontsize=7, facecolor=CARD, edgecolor=AXES, labelcolor=TXT)

# (1,1) Radar (DL models)
ax4 = fig.add_subplot(gs[1, 1], projection='polar')
ax4.set_facecolor(CARD)
ax4.tick_params(colors=TXT, labelsize=8)
ax4.spines['polar'].set_color(AXES)
categories = ['CD (inv)', 'P2P (inv)', 'Geo Recall']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

cd_dl  = [fc[m]['chamfer'] for m in dl_methods]
p2p_dl = [fc[m]['ptp'] for m in dl_methods]
gr_dl  = [fc[m]['geometric_recall'] for m in dl_methods]
inv_cd  = [1 / c for c in cd_dl];  max_icd = max(inv_cd)
inv_p2p = [1 / p for p in p2p_dl]; max_ip  = max(inv_p2p)
max_gr  = max(gr_dl)

for j, m in enumerate(dl_methods):
    vals = [inv_cd[j] / max_icd, inv_p2p[j] / max_ip, gr_dl[j] / max_gr]
    vals += vals[:1]
    ax4.plot(angles, vals, color=COLOR[m], linewidth=2)
    ax4.fill(angles, vals, color=COLOR[m], alpha=0.15)

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, color=TXT, fontsize=9)
ax4.set_yticks([0.25, 0.5, 0.75, 1.0])
ax4.set_yticklabels(['25%', '50%', '75%', '100%'], color=TXT, fontsize=7)
ax4.set_title('DL Model Radar', color=TXT, fontsize=11, pad=15, fontweight='bold')
patches4 = [mpatches.Patch(color=COLOR[m], label=m) for m in dl_methods]
ax4.legend(handles=patches4, loc='upper right', bbox_to_anchor=(1.4, 1.1),
           fontsize=7, facecolor=CARD, edgecolor=AXES, labelcolor=TXT)

# (1,2) Summary Table
ax5 = fig.add_subplot(gs[1, 2])
styled_ax(ax5, 'Summary Table')
ax5.axis('off')
rows = [['Model', 'CD', 'P2P', 'GR']]
for m in methods_trad:
    p2p_m = np.mean([r[m + '_p2p'] for r in records if m + '_p2p' in r])
    rows.append([m[:12], f"{avg[m]['cd']:.5f}", f"{p2p_m:.5f}", f"{avg[m]['gr']:.3f}"])
rows.append(['---', '---', '---', '---'])
for m in dl_methods:
    rows.append([m[:12], f"{fc[m]['chamfer']:.5f}",
                 f"{fc[m]['ptp']:.5f}", f"{fc[m]['geometric_recall']:.3f}"])
tbl = ax5.table(cellText=rows[1:], colLabels=rows[0],
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for key, cell in tbl.get_celld().items():
    cell.set_facecolor(CARD if key[0] % 2 == 0 else AXES)
    cell.set_text_props(color=TXT)
    cell.set_edgecolor(AXES)

fig.text(0.5, 0.97,
         'Plant Point Cloud Denoising — Performance Dashboard',
         ha='center', va='top', color='#80DEEA', fontsize=15, fontweight='bold')
plt.savefig(VIS_DIR + 'dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('dashboard.png saved')

# ------------------------------------------------------------------
# 2. radar_chart.png
# ------------------------------------------------------------------
fig2, axes_r = plt.subplots(1, 2, subplot_kw={'projection': 'polar'},
                             figsize=(14, 6), facecolor=BG)
cats4 = ['CD (inv)', 'P2P (inv)', 'Geo Recall', 'Speed (inv)']
angles4 = np.linspace(0, 2 * np.pi, len(cats4), endpoint=False).tolist()
angles4 += angles4[:1]

# Traditional
ax_t = axes_r[0]
ax_t.set_facecolor(CARD)
ax_t.spines['polar'].set_color(AXES)
ax_t.tick_params(colors=TXT)
time_t = [avg[m]['time'] for m in methods_trad]
cd_tv  = [avg[m]['cd'] for m in methods_trad]
p2p_tv = [np.mean([r[m + '_p2p'] for r in records if m + '_p2p' in r]) for m in methods_trad]
gr_tv  = [avg[m]['gr'] for m in methods_trad]
sp_tv  = [1 / t for t in time_t]
cd_tn  = norm_max([1 / x for x in cd_tv])
p2p_tn = norm_max([1 / x for x in p2p_tv])
gr_tn  = norm_max(gr_tv)
sp_tn  = norm_max(sp_tv)
for j, m in enumerate(methods_trad):
    v = [cd_tn[j], p2p_tn[j], gr_tn[j], sp_tn[j]]
    v += v[:1]
    ax_t.plot(angles4, v, color=COLOR[m], linewidth=2, label=m)
    ax_t.fill(angles4, v, color=COLOR[m], alpha=0.1)
ax_t.set_xticks(angles4[:-1])
ax_t.set_xticklabels(cats4, color=TXT, fontsize=9)
ax_t.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_t.set_yticklabels(['25%', '50%', '75%', '100%'], color=TXT, fontsize=7)
ax_t.set_title('Traditional Methods', color=TXT, fontsize=12, pad=15, fontweight='bold')
ax_t.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15),
            fontsize=8, facecolor=CARD, edgecolor=AXES, labelcolor=TXT)

# DL
ax_d = axes_r[1]
ax_d.set_facecolor(CARD)
ax_d.spines['polar'].set_color(AXES)
ax_d.tick_params(colors=TXT)
cd_dv  = [fc[m]['chamfer'] for m in dl_methods]
p2p_dv = [fc[m]['ptp'] for m in dl_methods]
gr_dv  = [fc[m]['geometric_recall'] for m in dl_methods]
cd_dn  = norm_max([1 / x for x in cd_dv])
p2p_dn = norm_max([1 / x for x in p2p_dv])
gr_dn  = norm_max(gr_dv)
for j, m in enumerate(dl_methods):
    v = [cd_dn[j], p2p_dn[j], gr_dn[j], 1.0]
    v += v[:1]
    ax_d.plot(angles4, v, color=COLOR[m], linewidth=2, label=m)
    ax_d.fill(angles4, v, color=COLOR[m], alpha=0.1)
ax_d.set_xticks(angles4[:-1])
ax_d.set_xticklabels(cats4, color=TXT, fontsize=9)
ax_d.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_d.set_yticklabels(['25%', '50%', '75%', '100%'], color=TXT, fontsize=7)
ax_d.set_title('Deep Learning Models', color=TXT, fontsize=12, pad=15, fontweight='bold')
ax_d.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15),
            fontsize=8, facecolor=CARD, edgecolor=AXES, labelcolor=TXT)

fig2.text(0.5, 0.98,
          'Radar Chart — Multi-Dimensional Evaluation (with Geometric Recall)',
          ha='center', va='top', color='#80DEEA', fontsize=13, fontweight='bold')
fig2.patch.set_facecolor(BG)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(VIS_DIR + 'radar_chart.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('radar_chart.png saved')

# ------------------------------------------------------------------
# 3. model_comparison.png
# ------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
metric_info = [
    ('Chamfer Distance (lower is better)', cd_vals,  False),
    ('P2P Distance (lower is better)',     p2p_vals, False),
    ('Geometric Recall (higher is better)', gr_vals,  True),
]
label_short = ['Bilateral', 'Laplacian', 'Iterative', 'PointFilter', 'IterPFN', 'StraightPCF']

for ax, (title, vals, higher) in zip(axes3, metric_info):
    styled_ax(ax, title)
    bars = ax.bar(range(len(all_methods)), vals, color=colors, edgecolor='#222', linewidth=0.5)
    ax.set_xticks(range(len(all_methods)))
    ax.set_xticklabels(label_short, rotation=30, ha='right', fontsize=8)
    best_idx = int(np.argmax(vals)) if higher else int(np.argmin(vals))
    for k, (b, v) in enumerate(zip(bars, vals)):
        fc_c = '#FFD700' if k == best_idx else TXT
        fw   = 'bold' if k == best_idx else 'normal'
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02 if higher else v + max(vals) * 0.005,
                f'{v:.4f}', ha='center', va='bottom', color=fc_c, fontsize=7, fontweight=fw)
    if higher:
        ax.set_ylabel('Recall Rate', color=TXT)
        ax.set_ylim(0, max(vals) * 1.4)
    else:
        ax.set_ylabel('Distance', color=TXT)

fig3.text(0.5, 0.98,
          'Model Comparison — Three Metrics (Gold = Best)',
          ha='center', va='top', color='#80DEEA', fontsize=13, fontweight='bold')
fig3.patch.set_facecolor(BG)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(VIS_DIR + 'model_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('model_comparison.png saved')

print('\nAll visualizations updated successfully!')
