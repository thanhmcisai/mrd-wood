"""
Swin-Tiny Degenerate Cluster Analysis
======================================
Evidence that Swin-T's high CSI reflects cluster collapse, not anatomical stability.

Three lines of evidence:
1. CSI variance near-zero across ALL perturbation types (collapse, not stability)
2. SGI ≈ 0 (single homogeneous region, not anatomical structure)
3. CAM entropy = 0 (100% activation on C4/lightest cluster = no discrimination)
"""
import logging
import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from wood_spatial.config import BB_ORDER, V4_CSV, V4_FIGURES

logger = logging.getLogger(__name__)


def run_swint_ablation(save: bool = True) -> dict:
    csi = pd.read_csv(V4_CSV / 'exp2_csi.csv')
    sgi = pd.read_csv(V4_CSV / 'exp2_sgi.csv')
    cd = pd.read_csv(V4_CSV / 'exp3_cam_distribution.csv')

    # ── 1. CSI variance analysis ──────────────────────────────────────────────
    # Hypothesis: if CSI is HIGH because of collapse, variance across
    # perturbation types should be NEAR ZERO (perturbation doesn't change
    # what's already degenerate). Real stability would show perturbation-dependent CSI.
    csi_var_rows = []
    for bb, sub in csi.groupby('backbone'):
        per_pert = sub.groupby('perturbation').csi.mean()
        csi_var_rows.append({
            'backbone': bb,
            'csi_mean': float(sub.csi.mean()),
            'csi_std': float(sub.csi.std()),
            'csi_range': float(per_pert.max() - per_pert.min()),
            'csi_cv': float(sub.csi.std() / max(sub.csi.mean(), 1e-9)),
        })
    df_csi_var = pd.DataFrame(csi_var_rows).sort_values('csi_mean', ascending=False)

    # ── 2. ANOVA: does perturbation type affect CSI significantly? ────────────
    anova_rows = []
    for bb, sub in csi.groupby('backbone'):
        groups = [grp.csi.values for _, grp in sub.groupby('perturbation') if len(grp) >= 3]
        if len(groups) < 3:
            continue
        try:
            f, p = f_oneway(*groups)
            anova_rows.append({'backbone': bb, 'F': float(f), 'p': float(p),
                                'sig': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'})
        except Exception:
            pass
    df_anova = pd.DataFrame(anova_rows)

    # ── 3. SGI distribution ───────────────────────────────────────────────────
    clean_sgi = sgi[sgi.condition == 'clean']
    sgi_stats = clean_sgi.groupby('backbone').sgi.agg(['mean', 'std', 'median']).reset_index()

    # ── 4. CAM distribution entropy ───────────────────────────────────────────
    eps = 1e-10
    cam_cols = sorted([c for c in cd.columns if c.startswith('cam_pct_c')])
    def entropy_from_cols(row):
        p = np.array([row[c] for c in cam_cols], dtype=float)
        p = np.clip(p, eps, None); p /= p.sum()
        return float(-np.sum(p * np.log2(p)))
    cd['entropy_comp'] = cd.apply(entropy_from_cols, axis=1)
    cam_ent = cd[cd.condition == 'clean'].groupby('backbone').entropy_comp.agg(['mean', 'std']).reset_index()

    # ── 5. Summary table ──────────────────────────────────────────────────────
    summary = df_csi_var.merge(sgi_stats[['backbone', 'mean']], on='backbone', suffixes=('', '_sgi'))
    summary = summary.merge(cam_ent[['backbone', 'mean']], on='backbone', suffixes=('', '_ent'))
    summary = summary.rename(columns={'mean': 'sgi_mean', 'mean_ent': 'cam_entropy'})
    summary['collapse_verdict'] = summary.apply(
        lambda r: 'DEGENERATE' if r['sgi_mean'] < 0.001 and r['csi_cv'] < 0.05
        else 'STABLE' if r['csi_cv'] > 0.15 else 'MIXED', axis=1
    )

    logger.info('=== Swin-T Degenerate Evidence ===')
    logger.info('\n%s', summary[['backbone', 'csi_mean', 'csi_cv', 'sgi_mean', 'cam_entropy', 'collapse_verdict']].to_string(index=False))
    logger.info('\n=== CSI ANOVA (does perturbation type affect CSI?) ===')
    logger.info('\n%s', df_anova.to_string(index=False))

    results = {
        'csi_variance': df_csi_var,
        'csi_anova': df_anova,
        'sgi_stats': sgi_stats,
        'cam_entropy': cam_ent,
        'summary': summary,
    }

    if save:
        df_csi_var.to_csv(V4_CSV / 'exp_swint_csi_variance.csv', index=False)
        df_anova.to_csv(V4_CSV / 'exp_swint_csi_anova.csv', index=False)
        summary.to_csv(V4_CSV / 'exp_swint_summary.csv', index=False)
        logger.info('Saved Swin-T ablation to %s', V4_CSV)

    return results


def plot_swint_degenerate(results: dict):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    df_var = results['csi_variance']
    df_anova = results['csi_anova']
    csi = pd.read_csv(V4_CSV / 'exp2_csi.csv')
    sgi = pd.read_csv(V4_CSV / 'exp2_sgi.csv')

    from wood_spatial.config import BB_LABEL
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: CSI distribution per backbone (boxplot)
    ax = axes[0]
    order = df_var.sort_values('csi_mean', ascending=False)['backbone'].tolist()
    data = [csi[csi.backbone == bb].csi.values for bb in order]
    bp = ax.boxplot(data, labels=[BB_LABEL.get(b, b) for b in order],
                    patch_artist=True, notch=False)
    colors = ['firebrick' if b == 'swin_tiny' else 'steelblue' for b in order]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([BB_LABEL.get(b, b) for b in order], rotation=25, ha='right')
    ax.set_ylabel('CSI')
    ax.set_title('A. CSI distribution per backbone\n(narrow = collapse, wide = real stability)')
    ax.axhline(0.249, color='firebrick', linestyle='--', linewidth=0.8, alpha=0.6)

    # Panel B: CSI per perturbation for Swin-T vs best backbone (DINOv2)
    ax2 = axes[1]
    pt_order = csi.groupby('perturbation').csi.mean().sort_values().index.tolist()
    for bb, color, label in [('swin_tiny', 'firebrick', 'Swin-T (degenerate)'),
                               ('dinov2_b', 'steelblue', 'DINOv2-B (stable)')]:
        vals = [csi[(csi.backbone == bb) & (csi.perturbation == p)].csi.mean() for p in pt_order]
        ax2.plot(range(len(pt_order)), vals, 'o-', color=color, label=label, markersize=5)
    ax2.set_xticks(range(len(pt_order)))
    ax2.set_xticklabels([p.replace('_', ' ') for p in pt_order], rotation=30, ha='right')
    ax2.set_ylabel('Mean CSI')
    ax2.set_title('B. CSI by perturbation type\n(flat line = not responding to perturbation)')
    ax2.legend(fontsize=8)

    # Panel C: SGI distribution (log scale) — Swin-T near zero
    ax3 = axes[2]
    clean_sgi = sgi[sgi.condition == 'clean']
    bb_order_sgi = clean_sgi.groupby('backbone').sgi.mean().sort_values(ascending=False).index.tolist()
    means = [clean_sgi[clean_sgi.backbone == b].sgi.mean() for b in bb_order_sgi]
    colors3 = ['firebrick' if b == 'swin_tiny' else 'steelblue' for b in bb_order_sgi]
    bars = ax3.bar([BB_LABEL.get(b, b) for b in bb_order_sgi], means, color=colors3, alpha=0.8)
    ax3.set_yscale('log')
    ax3.set_ylabel('Mean SGI (log scale)')
    ax3.set_title('C. Spatial Granularity Index (clean)\n(Swin-T ≈ 0 = single homogeneous region)')
    ax3.set_xticklabels([BB_LABEL.get(b, b) for b in bb_order_sgi], rotation=25, ha='right')
    for bar, val in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width() / 2, val * 1.3, f'{val:.5f}',
                 ha='center', va='bottom', fontsize=6.5, rotation=90)

    handles = [mpatches.Patch(color='firebrick', alpha=0.7, label='Swin-T (degenerate)'),
               mpatches.Patch(color='steelblue', alpha=0.7, label='Other backbones')]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle('Evidence for Swin-T Spatial Cluster Collapse vs Anatomical Stability', fontsize=10, y=1.01)

    V4_FIGURES = __import__('wood_spatial.config', fromlist=['V4_FIGURES']).V4_FIGURES
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(V4_FIGURES / f'fig_swint_degenerate.{ext}', bbox_inches='tight',
                    dpi=300 if ext == 'png' else None)
    logger.info('Saved fig_swint_degenerate')
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    results = run_swint_ablation()
    plot_swint_degenerate(results)
    print('\n=== Summary ===')
    print(results['summary'][['backbone', 'csi_mean', 'csi_cv', 'sgi_mean', 'cam_entropy', 'collapse_verdict']].to_string(index=False))
    print('\n=== ANOVA: does perturbation affect CSI? ===')
    print(results['csi_anova'].to_string(index=False))


if __name__ == '__main__':
    main()
