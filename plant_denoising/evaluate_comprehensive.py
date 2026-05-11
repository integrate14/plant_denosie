"""
综合评估脚本
评估范围：
1. 噪声基数 (Noisy Baseline) - 直接使用带噪声点云
2. 传统方法：Gaussian, Bilateral, SOR, ROR, Median, MLS
3. 深度学习模型：PointFilter, IterativePFN, StraightPCF

指标：CD (Chamfer Distance), P2P (Point-to-Point), GR (Geometric Recall)
"""
import os
os.environ['TORCH_DISABLE_CUDA_INIT'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USERNAME'] = 'User'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\torch_cache'

import sys
sys.path.insert(0, r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising')

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
import json
import time

from data.synthetic_data_generator import load_dataset
from models.pointfilter import create_pointfilter_model
from models.iterative_pfn_improved import create_iterativepfn_improved_model
from models.straight_pcf_improved import StraightPCFImproved
from models.traditional_methods import (
    gaussian_filter, bilateral_filter, sor_denoise,
    ror_denoise, median_filter, mls_denoise,
)

# ================================================================
# 配置
# ================================================================
DATA_PATH     = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\data\synthetic_dataset.pkl'
CHECKPOINT_DIR = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\checkpoints'
RESULTS_DIR   = r'C:\Users\Lenovo\Desktop\deep-work\plant_denoising\results'
NOISE_LEVEL   = 0.02
THRESHOLD     = 2 * NOISE_LEVEL  # 0.04
SHAPES        = ['sphere', 'cube', 'cone']

print('=' * 80)
print('综合评估：噪声基数 vs 传统方法 vs 深度学习模型')
print('=' * 80)
print(f'噪声水平: {NOISE_LEVEL}')
print(f'GR阈值: {THRESHOLD} (2 x noise_level)')
print(f'形状: {SHAPES}')
print()

# ================================================================
# 数据集
# ================================================================
class PointCloudDataset(Dataset):
    def __init__(self, clean_shapes, noisy_shapes, shape, noise_level=0.02):
        self.clean = clean_shapes[shape]
        self.noisy = noisy_shapes[shape][noise_level]

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.noisy[idx]).float(),
                torch.from_numpy(self.clean[idx]['points']).float())


# ================================================================
# 指标计算 (修复版 GR)
# ================================================================
def compute_all_metrics(pred: torch.Tensor, clean: torch.Tensor) -> dict:
    """
    计算 CD, P2P, GR
    - CD: 平方 Chamfer Distance
    - P2P: 逐点欧氏距离均值
    - GR: torch.cdist 线性距离, dim=0 (Recall方向)
    """
    if pred.dim() == 3:
        pred  = pred.squeeze(0)
    if clean.dim() == 3:
        clean = clean.squeeze(0)

    # Chamfer Distance (平方)
    dist_sq = torch.sum((pred.unsqueeze(1) - clean.unsqueeze(0)) ** 2, dim=2)
    cd = torch.mean(torch.min(dist_sq, dim=1)[0]) + torch.mean(torch.min(dist_sq, dim=0)[0])

    # P2P
    p2p = torch.mean(torch.sqrt(torch.sum((pred - clean) ** 2, dim=-1) + 1e-8))

    # GR - 修复版
    dist_l2 = torch.cdist(pred, clean)
    min_dist_per_clean, _ = torch.min(dist_l2, dim=0)
    gr = (min_dist_per_clean <= THRESHOLD).float().mean()

    return {
        'CD':  cd.item(),
        'P2P': p2p.item(),
        'GR':  gr.item()
    }


# ================================================================
# 评估函数
# ================================================================
def evaluate_noisy_baseline(loader) -> dict:
    """噪声基数评估"""
    total_cd = total_p2p = total_gr = 0.0
    cnt = 0
    for noisy, clean in loader:
        metrics = compute_all_metrics(noisy.squeeze(0), clean.squeeze(0))
        total_cd  += metrics['CD']
        total_p2p += metrics['P2P']
        total_gr  += metrics['GR']
        cnt += 1
    return {'CD': total_cd/cnt, 'P2P': total_p2p/cnt, 'GR': total_gr/cnt, 'n': cnt}


def evaluate_traditional_method(method_fn, loader) -> dict:
    """传统方法评估"""
    total_cd = total_gr = 0.0
    cnt = 0
    for noisy, clean in loader:
        noisy_np = noisy.squeeze(0).cpu().numpy()
        cleaned_np = method_fn(noisy_np)
        cleaned_tensor = torch.from_numpy(cleaned_np).float()
        clean_tensor = clean.squeeze(0)

        metrics = compute_all_metrics(cleaned_tensor, clean_tensor)
        total_cd  += metrics['CD']
        total_gr  += metrics['GR']
        cnt += 1
    return {'CD': total_cd/cnt, 'P2P': None, 'GR': total_gr/cnt, 'n': cnt}


def evaluate_dl_model(model, loader) -> dict:
    """深度学习模型评估"""
    model.eval()
    total_cd = total_p2p = total_gr = 0.0
    cnt = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to('cpu')
            clean = clean.to('cpu')
            cleaned = model(noisy)
            if isinstance(cleaned, tuple):
                cleaned = cleaned[0]
            if cleaned.dim() == 3:
                cleaned = cleaned.squeeze(0)
            if clean.dim() == 3:
                clean = clean.squeeze(0)

            metrics = compute_all_metrics(cleaned, clean)
            total_cd  += metrics['CD']
            total_p2p += metrics['P2P']
            total_gr  += metrics['GR']
            cnt += 1
    return {'CD': total_cd/cnt, 'P2P': total_p2p/cnt, 'GR': total_gr/cnt, 'n': cnt}


# ================================================================
# 主程序
# ================================================================
def main():
    data = load_dataset(DATA_PATH)
    torch.manual_seed(42)

    # 所有方法定义
    methods = {
        'Noisy Baseline': {'type': 'noisy'},
        'Gaussian':      {'type': 'traditional', 'fn': lambda p: gaussian_filter(p, k=16, sigma=0.02)},
        'Bilateral':     {'type': 'traditional', 'fn': lambda p: bilateral_filter(p, k=16, sigma_s=0.02, sigma_r=0.02)},
        'SOR':           {'type': 'traditional', 'fn': lambda p: sor_denoise(p, k=16, std_ratio=2.0)},
        'ROR':           {'type': 'traditional', 'fn': lambda p: ror_denoise(p, radius=0.05, min_neighbors=4)},
        'Median':        {'type': 'traditional', 'fn': lambda p: median_filter(p, k=16)},
        'MLS':           {'type': 'traditional', 'fn': lambda p: mls_denoise(p, k=16, sigma=0.02)},
    }

    # 加载深度学习模型
    print('加载深度学习模型...')
    device = 'cpu'

    model_pf = create_pointfilter_model(num_points=2048)
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'PointFilter_mixed_best.pth'),
                      map_location=device, weights_only=False)
    model_pf.load_state_dict(ckpt['model_state_dict'])
    print(f'  PointFilter   (val_loss={ckpt.get("val_loss", "?"):.6f})')

    model_ipfn = create_iterativepfn_improved_model(
        num_points=2048, num_iterations=3, feature_dim=256, hidden_dim=128)
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'IterativePFN_mixed_best.pth'),
                      map_location=device, weights_only=False)
    # strict=False: residual_weight 在不同层级的兼容加载
    missing, unexpected = model_ipfn.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f'    [WARN] Missing keys: {missing}')
    if unexpected:
        # 把 iteration_modules.X.residual_weight 手动填入顶层 residual_weight
        sd = ckpt['model_state_dict']
        rw_vals = [sd[k].item() for k in sd if 'residual_weight' in k]
        model_ipfn.residual_weight.data.fill_(float(sum(rw_vals) / len(rw_vals)))
    print(f'  IterativePFN  (val_loss={ckpt.get("val_loss", "?"):.6f})')

    model_spcf = StraightPCFImproved(num_points=2048, feature_dim=256, hidden_dim=128,
                                     num_iterations=3, use_dgcnn=False)
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'StraightPCF_mixed_best.pth'),
                      map_location=device, weights_only=False)
    model_spcf.load_state_dict(ckpt['model_state_dict'])
    print(f'  StraightPCF   (val_loss={ckpt.get("val_loss", "?"):.6f})')

    dl_models = {
        'PointFilter':  model_pf,
        'IterativePFN': model_ipfn,
        'StraightPCF':  model_spcf,
    }

    # 评估所有方法
    all_results = {}

    for method_name, method_info in methods.items():
        print(f'\n评估 {method_name}...')
        all_results[method_name] = {}

        for shape in SHAPES:
            dataset = PointCloudDataset(data['clean'], data['noisy'], shape, NOISE_LEVEL)
            n = len(dataset)
            train_n = int(0.7 * n)
            val_n   = int(0.15 * n)
            test_n  = n - train_n - val_n
            generator = torch.Generator().manual_seed(42)
            _, _, test_subset = random_split(dataset, [train_n, val_n, test_n], generator=generator)
            loader = DataLoader(test_subset, batch_size=1, shuffle=False)

            if method_info['type'] == 'noisy':
                r = evaluate_noisy_baseline(loader)
            else:
                r = evaluate_traditional_method(method_info['fn'], loader)

            # MLS 在 sphere 上不适用
            if method_name == 'MLS' and shape == 'sphere':
                r['GR'] = None
                r['note'] = 'MLS不适用于球面'

            all_results[method_name][shape] = r
            p2p_str = f'{r["P2P"]:.6f}' if r.get('P2P') is not None else '   N/A  '
            gr_str  = f'{r["GR"]:.4f}' if r.get('GR') is not None else '  N/A  '
            print(f'  {shape:6s}: CD={r["CD"]:.6f}  P2P={p2p_str}  GR={gr_str}')

    for model_name, model in dl_models.items():
        print(f'\n评估 {model_name} (DL)...')
        all_results[model_name] = {}

        for shape in SHAPES:
            dataset = PointCloudDataset(data['clean'], data['noisy'], shape, NOISE_LEVEL)
            n = len(dataset)
            train_n = int(0.7 * n)
            val_n   = int(0.15 * n)
            test_n  = n - train_n - val_n
            generator = torch.Generator().manual_seed(42)
            _, _, test_subset = random_split(dataset, [train_n, val_n, test_n], generator=generator)
            loader = DataLoader(test_subset, batch_size=1, shuffle=False)

            r = evaluate_dl_model(model, loader)
            all_results[model_name][shape] = r
            print(f'  {shape:6s}: CD={r["CD"]:.6f}  P2P={r["P2P"]:.6f}  GR={r["GR"]:.4f}')

    # ================================================================
    # 汇总表格
    # ================================================================
    print('\n' + '=' * 100)
    print('综合评估结果汇总')
    print('=' * 100)

    # 按类型分组
    print('\n【噪声基数】')
    print(f"{'Shape':<8} | {'CD ↓':>12} | {'GR ↑':>10}")
    print('-' * 40)
    r = all_results['Noisy Baseline']
    for shape in SHAPES:
        print(f'{shape:<8} | {r[shape]["CD"]:>12.6f} | {r[shape]["GR"]:>10.4f}')

    print('\n【传统方法】')
    print(f"{'Method':<12} | {'Shape':<8} | {'CD ↓':>12} | {'GR ↑':>10}")
    print('-' * 50)
    for method in ['Gaussian', 'Bilateral', 'SOR', 'ROR', 'Median', 'MLS']:
        for shape in SHAPES:
            r = all_results[method][shape]
            gr_str = f'{r["GR"]:>10.4f}' if r.get('GR') is not None else '       N/A'
            print(f'{method:<12} | {shape:<8} | {r["CD"]:>12.6f} | {gr_str}')

    print('\n【深度学习模型】')
    print(f"{'Model':<14} | {'Shape':<8} | {'CD ↓':>12} | {'P2P ↓':>12} | {'GR ↑':>10}")
    print('-' * 65)
    for model_name in ['PointFilter', 'IterativePFN', 'StraightPCF']:
        for shape in SHAPES:
            r = all_results[model_name][shape]
            print(f'{model_name:<14} | {shape:<8} | {r["CD"]:>12.6f} | {r["P2P"]:>12.6f} | {r["GR"]:>10.4f}')

    # ================================================================
    # 均值对比表
    # ================================================================
    print('\n' + '=' * 100)
    print('均值对比 (所有形状)')
    print('=' * 100)

    # 计算各类别均值
    categories = {
        '噪声基数': ['Noisy Baseline'],
        '传统方法': ['Gaussian', 'Bilateral', 'SOR', 'ROR', 'Median', 'MLS'],
        '深度学习': ['PointFilter', 'IterativePFN', 'StraightPCF'],
    }

    summary_rows = []

    # 噪声基数
    for name in categories['噪声基数']:
        cds = [all_results[name][s]['CD'] for s in SHAPES]
        grs = [all_results[name][s]['GR'] for s in SHAPES]
        p2ps = [all_results[name][s]['P2P'] for s in SHAPES if all_results[name][s].get('P2P') is not None]
        summary_rows.append({
            'Category': '噪声基数',
            'Method': name,
            'Mean_CD': np.mean(cds),
            'Mean_P2P': np.mean(p2ps) if p2ps else None,
            'Mean_GR': np.mean(grs),
        })

    # 传统方法
    for name in categories['传统方法']:
        cds = [all_results[name][s]['CD'] for s in SHAPES]
        # 排除MLS+sphere的无效GR
        grs_valid = [all_results[name][s]['GR'] for s in SHAPES if all_results[name][s].get('GR') is not None]
        summary_rows.append({
            'Category': '传统方法',
            'Method': name,
            'Mean_CD': np.mean(cds),
            'Mean_P2P': None,
            'Mean_GR': np.mean(grs_valid) if grs_valid else None,
        })

    # 深度学习
    for name in categories['深度学习']:
        cds = [all_results[name][s]['CD'] for s in SHAPES]
        p2ps = [all_results[name][s]['P2P'] for s in SHAPES]
        grs = [all_results[name][s]['GR'] for s in SHAPES]
        summary_rows.append({
            'Category': '深度学习',
            'Method': name,
            'Mean_CD': np.mean(cds),
            'Mean_P2P': np.mean(p2ps),
            'Mean_GR': np.mean(grs),
        })

    print(f"\n{'Category':<12} | {'Method':<14} | {'Mean CD ↓':>12} | {'Mean P2P ↓':>12} | {'Mean GR ↑':>10}")
    print('-' * 75)
    for row in summary_rows:
        p2p_str = f'{row["Mean_P2P"]:>12.6f}' if row['Mean_P2P'] is not None else '         N/A'
        gr_str  = f'{row["Mean_GR"]:>10.4f}' if row['Mean_GR'] is not None else '       N/A'
        print(f"{row['Category']:<12} | {row['Method']:<14} | {row['Mean_CD']:>12.6f} | {p2p_str} | {gr_str}")

    # ================================================================
    # 保存结果
    # ================================================================
    output = {
        'timestamp':    datetime.now().isoformat(),
        'noise_level':  NOISE_LEVEL,
        'threshold':    THRESHOLD,
        'shapes':       SHAPES,
        'per_shape_results': all_results,
        'summary':      summary_rows,
    }

    out_path = os.path.join(RESULTS_DIR, 'comprehensive_evaluation.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=lambda x: float('nan') if x != x else x)
    print(f'\n结果已保存: {out_path}')

    # 生成 Markdown 报告
    generate_markdown_report(output, RESULTS_DIR)

    return output


def generate_markdown_report(results: dict, output_dir: str):
    """生成 Markdown 格式的评估报告"""
    md = """# 点云去噪综合评估报告

## 评估设置

- **噪声水平**: {noise_level}
- **GR阈值**: {threshold} (2 × noise_level)
- **测试形状**: {shapes}
- **评估时间**: {timestamp}

## 按形状评估结果

### 噪声基数 (Noisy Baseline)

| 形状 | CD ↓ | GR ↑ |
|:----:|:----:|:----:|
""".format(**results)

    for shape in results['shapes']:
        r = results['per_shape_results']['Noisy Baseline'][shape]
        md += f"| {shape} | {r['CD']:.6f} | {r['GR']:.4f} |\n"

    md += """
### 传统方法

| 方法 | 形状 | CD ↓ | GR ↑ |
|:----:|:----:|:----:|:----:|
"""
    trad_methods = ['Gaussian', 'Bilateral', 'SOR', 'ROR', 'Median', 'MLS']
    for method in trad_methods:
        for shape in results['shapes']:
            r = results['per_shape_results'][method][shape]
            gr_str = f"{r['GR']:.4f}" if r.get('GR') is not None else "N/A"
            md += f"| {method} | {shape} | {r['CD']:.6f} | {gr_str} |\n"

    md += """
### 深度学习模型

| 模型 | 形状 | CD ↓ | P2P ↓ | GR ↑ |
|:----:|:----:|:----:|:-----:|:----:|
"""
    dl_models = ['PointFilter', 'IterativePFN', 'StraightPCF']
    for model in dl_models:
        for shape in results['shapes']:
            r = results['per_shape_results'][model][shape]
            md += f"| {model} | {shape} | {r['CD']:.6f} | {r['P2P']:.6f} | {r['GR']:.4f} |\n"

    md += """
## 均值对比 (所有形状)

| 类别 | 方法 | Mean CD ↓ | Mean P2P ↓ | Mean GR ↑ |
|:----:|:----:|:---------:|:----------:|:---------:|
"""
    for row in results['summary']:
        p2p_str = f"{row['Mean_P2P']:.6f}" if row['Mean_P2P'] is not None else "N/A"
        gr_str  = f"{row['Mean_GR']:.4f}" if row['Mean_GR'] is not None else "N/A"
        md += f"| {row['Category']} | {row['Method']} | {row['Mean_CD']:.6f} | {p2p_str} | {gr_str} |\n"

    md += """
## 分析结论

"""
    # 自动生成分析结论
    noisy_cd = next(r['Mean_CD'] for r in results['summary'] if r['Method'] == 'Noisy Baseline')
    best_dl_cd = min((r for r in results['summary'] if r['Category'] == '深度学习'), key=lambda x: x['Mean_CD'])
    best_trad_cd = min((r for r in results['summary'] if r['Category'] == '传统方法'), key=lambda x: x['Mean_CD'])

    md += f"""- **噪声基数基准**: CD = {noisy_cd:.6f}
- **最佳传统方法**: {best_trad_cd['Method']}, CD = {best_trad_cd['Mean_CD']:.6f}, 提升 {((noisy_cd - best_trad_cd['Mean_CD']) / noisy_cd * 100):.1f}%
- **最佳深度学习**: {best_dl_cd['Method']}, CD = {best_dl_cd['Mean_CD']:.6f}, 提升 {((noisy_cd - best_dl_cd['Mean_CD']) / noisy_cd * 100):.1f}%
"""

    out_path = os.path.join(output_dir, 'comprehensive_evaluation_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'报告已生成: {out_path}')


if __name__ == '__main__':
    main()
