"""Publication figure generators for the representation-drift analysis."""
import logging

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wood_spatial.config import (
    BB_LABEL, BB_ORDER, PT_LABEL, V4_CSV, V4_FIGURES,
    V4_SPATIAL_CACHE, CLUSTER_COLOR, CLUSTER_NAMES, N_CLUSTERS,
)

logger = logging.getLogger(__name__)

matplotlib.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

FIG_GROUP_ORDER = [
    'compound', 'blur', 'geometric', 'color_shift',
    'noise', 'digital', 'photometric', 'surface_artifact',
]

FIG_GROUP_LABEL = {
    'compound': 'Compound',
    'blur': 'Blur',
    'geometric': 'Geometric',
    'color_shift': 'RGB Channel Shift',
    'noise': 'Noise',
    'digital': 'Digital',
    'photometric': 'Photometric',
    'surface_artifact': 'Surface Artifact',
}

BB_COLORS = {
    'resnet50': '#e41a1c', 'efficientnet_b3': '#377eb8',
    'convnext_tiny': '#4daf4a', 'swin_tiny': '#984ea3',
    'dinov2_b': '#ff7f00', 'hrnet32': '#a65628', 'mobilenetv3': '#f781bf',
}


def _figure_group(perturbation: str) -> str:
    if perturbation in {
        'compound', 'compound_optical', 'compound_digital', 'compound_field',
    }:
        return 'compound'
    if perturbation in {'gaussian_blur', 'defocus_blur', 'motion_blur', 'zoom_blur'}:
        return 'blur'
    if perturbation in {'resize', 'rotation'}:
        return 'geometric'
    if perturbation in {
        'color_shift', 'red_channel_shift', 'green_channel_shift', 'blue_channel_shift',
    }:
        return 'color_shift'
    if perturbation in {'gaussian_noise', 'shot_noise', 'impulse_noise'}:
        return 'noise'
    if perturbation in {'jpeg', 'pixelate'}:
        return 'digital'
    if perturbation in {'illumination', 'contrast'}:
        return 'photometric'
    if perturbation == 'scratch':
        return 'surface_artifact'
    return perturbation


def _severity_order_value(value):
    order = {'mild': 1, 'moderate': 2, 'severe': 3}
    if str(value) in order:
        return order[str(value)]
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ordered_backbones(values):
    return [bb for bb in BB_ORDER if bb in set(values)]


def _save(fig, name: str):
    V4_FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = V4_FIGURES / f'{name}.{ext}'
        fig.savefig(path, bbox_inches='tight', dpi=300 if ext == 'png' else None)
    logger.info('Saved %s', V4_FIGURES / name)
    plt.close(fig)


# ── Figure 2a: Accuracy drop heatmap ────────────────────────────────────────

def plot_accuracy_heatmap():
    df = pd.read_csv(V4_CSV / 'exp1_accuracy_matrix.csv')
    pert = df[df['perturbation'] != 'clean'].copy()
    pert['figure_group'] = pert['perturbation'].map(_figure_group)

    # mean drop per backbone × interpretable perturbation group
    pivot = pert.groupby(['backbone', 'figure_group'])['drop'].mean().unstack('figure_group')
    pivot = pivot.loc[_ordered_backbones(pivot.index)]
    pt_cols = [p for p in FIG_GROUP_ORDER if p in pivot.columns]
    pivot = pivot[pt_cols]
    xlabels = [FIG_GROUP_LABEL.get(p, p) for p in pt_cols]
    ylabels = [BB_LABEL.get(bb, bb) for bb in pivot.index]

    fig, ax = plt.subplots(figsize=(10.8, 4))
    im = ax.imshow(pivot.values, aspect='auto', cmap='Reds', vmin=0, vmax=0.8)
    for i in range(len(pivot.index)):
        for j in range(len(pt_cols)):
            v = pivot.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if v > 0.45 else 'black')
    ax.set_xticks(range(len(pt_cols)))
    ax.set_xticklabels(xlabels, rotation=30, ha='right')
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    fig.colorbar(im, ax=ax, label='Accuracy drop', fraction=0.03, pad=0.02)
    _save(fig, 'fig2a_accuracy_heatmap')


# ── Figure 2b: Severity degradation curves with CI ──────────────────────────

def plot_severity_curves():
    df = pd.read_csv(V4_CSV / 'exp1_accuracy_matrix.csv')
    ci = pd.read_csv(V4_CSV / 'exp1_bootstrap_ci.csv')

    key_perts = ['gaussian_blur', 'blue_channel_shift', 'scratch', 'compound_field']
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharey=True)
    axes = axes.ravel()

    for ax, pert in zip(axes, key_perts):
        sub = df[(df['perturbation'] == pert) & (df['backbone'].isin(BB_ORDER))]
        sub_ci = ci[ci['perturbation'] == pert]

        for bb in BB_ORDER:
            g = sub[sub['backbone'] == bb].copy()
            if g.empty:
                continue
            g['_x'] = g['severity'].map(_severity_order_value)
            g = g.sort_values('_x')
            x = g['_x'].values
            ax.plot(x, g['accuracy'].values, marker='o', markersize=3,
                    color=BB_COLORS.get(bb, None), label=BB_LABEL.get(bb, bb))

            # CI band if available
            ci_g = sub_ci[sub_ci['backbone'] == bb]
            if len(ci_g) > 0:
                try:
                    ci_g = ci_g.copy()
                    ci_g['_x'] = ci_g['severity'].map(_severity_order_value)
                    ci_g = ci_g.sort_values('_x')
                    cx = ci_g['_x'].values
                    ax.fill_between(cx, ci_g['ci_lo'].values, ci_g['ci_hi'].values,
                                    alpha=0.12, color=BB_COLORS.get(bb, None))
                except Exception:
                    pass

        ax.text(0.03, 0.94, PT_LABEL.get(pert, pert), transform=ax.transAxes,
                ha='left', va='top', fontsize=8.2, fontweight='semibold',
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.78))
        if pert in {'scratch', 'compound_field'}:
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(['mild', 'moderate', 'severe'])
        ax.set_xlabel('Severity')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Accuracy')
    axes[2].set_ylabel('Accuracy')

    handles = [mpatches.Patch(color=BB_COLORS[bb], label=BB_LABEL[bb]) for bb in BB_ORDER]
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.19, wspace=0.16, hspace=0.28)
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02),
               frameon=False, fontsize=8)
    _save(fig, 'fig2b_severity_curves')


# ── Figure 5: Nemenyi CD diagram ─────────────────────────────────────────────

def plot_nemenyi_cd_diagram():
    stats = pd.read_csv(V4_CSV / 'exp1_statistical_tests.csv')
    nem_path = V4_CSV / 'exp1_nemenyi_pvalues.csv'
    if nem_path.exists():
        nem = pd.read_csv(nem_path, index_col=0)
        group_note = 'Connected groups are NOT significantly different (Nemenyi p > 0.05)'
    else:
        wil_path = V4_CSV / 'exp1_wilcoxon_pairwise.csv'
        if not wil_path.exists():
            logger.warning('Nemenyi/Wilcoxon p-values not found, skipping CD diagram')
            return
        wil = pd.read_csv(wil_path)
        nem = pd.DataFrame(1.0, index=BB_ORDER, columns=BB_ORDER)
        for _, row in wil.iterrows():
            pval = row.get('p_corrected', row.get('p', np.nan))
            if pd.isna(pval):
                continue
            nem.loc[row['a'], row['b']] = float(pval)
            nem.loc[row['b'], row['a']] = float(pval)
        group_note = 'Connected groups are NOT significantly different (Holm-corrected Wilcoxon p > 0.05)'
    ranks_row = stats[stats['test'] == 'mean_rank'][['backbone', 'rank']].dropna()
    if ranks_row.empty:
        logger.warning('No mean_rank rows found, skipping CD diagram')
        return

    cd_row = stats[stats['test'] == 'Nemenyi_CD']
    cd = float(cd_row['CD'].values[0]) if len(cd_row) > 0 else None

    ranks = dict(zip(ranks_row['backbone'], ranks_row['rank'].astype(float)))
    ordered = sorted(ranks.items(), key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(9.4, 3.2))
    ax.set_xlim(0.5, len(ordered) + 0.5)
    ax.set_ylim(-1.05, 2.45)
    ax.axis('off')

    y_rank = 1.72
    for i, (bb, r) in enumerate(ordered):
        x = i + 1
        ax.plot(x, y_rank, 'ko', markersize=6, zorder=3)
        ax.text(x, y_rank + 0.20, f'{r:.2f}', ha='center', fontsize=8)
        ax.text(x, y_rank - 0.42, BB_LABEL.get(bb, bb), ha='center',
                fontsize=7.0, rotation=0)

    ax.axhline(y_rank, color='black', linewidth=1, xmin=0.05, xmax=0.95)

    # Draw significant difference groups (p > 0.05 means not significantly different)
    alpha = 0.05
    y_line = y_rank - 0.92
    bb_names = [bb for bb, _ in ordered]
    x_pos = {bb: i + 1 for i, (bb, _) in enumerate(ordered)}
    drawn = set()
    for i, bb_a in enumerate(bb_names):
        for j, bb_b in enumerate(bb_names[i + 1:], start=i + 1):
            if bb_a not in nem.index or bb_b not in nem.columns:
                continue
            pval = float(nem.loc[bb_a, bb_b])
            if pval > alpha:
                key = tuple(sorted([bb_a, bb_b]))
                if key not in drawn:
                    ax.plot([x_pos[bb_a], x_pos[bb_b]], [y_line, y_line],
                            'b-', linewidth=2.5, alpha=0.7, solid_capstyle='round')
                    drawn.add(key)
                    y_line -= 0.16

    if cd is not None:
        ax.text(0.98, 0.95, f'CD = {cd:.3f}  (α=0.05)', transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.text(0.5, -0.10, group_note,
            transform=ax.transAxes, ha='center', fontsize=6.8, color='steelblue')
    fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.20)
    _save(fig, 'fig5_nemenyi_cd_diagram')


# ── Figure 4: Spatial cluster visualization panels ──────────────────────────

def _load_cluster_map(bb: str, ds: str, tag: str):
    path = V4_SPATIAL_CACHE / f'{bb}_{ds}_{tag}.npz'
    if not path.exists():
        return None, None, None
    d = np.load(path, allow_pickle=True)
    return d['features'], d['labels'], d['paths']


def _render_cluster_overlay(img_bgr, label_map, alpha=0.55):
    """Blend cluster color map onto image."""
    overlay = img_bgr.copy().astype(np.float32)
    cluster_colors_bgr = {
        k: tuple(int(c * 255) for c in matplotlib.colors.to_rgb(v))[::-1]
        for k, v in CLUSTER_COLOR.items()
    }
    color_map = np.zeros_like(img_bgr, dtype=np.float32)
    for k in range(N_CLUSTERS):
        mask = label_map == k
        c = cluster_colors_bgr.get(k, (128, 128, 128))
        color_map[mask] = c
    return np.clip(overlay * (1 - alpha) + color_map * alpha, 0, 255).astype(np.uint8)


def plot_spatial_cluster_panels(ds: str = 'WRD25', sample_idx: int = 0):
    from wood_spatial.spatial.cluster_pipeline import cluster_spatial_features, correct_illumination

    conditions = [
        ('original', 'Clean'),
        ('blur_12', 'Blur (r=12)'),
        ('scratch_severe', 'Scratch (severe)'),
        ('compound_field_severe', 'Field compound'),
    ]
    img_size = 224

    fig, axes = plt.subplots(
        len(BB_ORDER), len(conditions) * 2,
        figsize=(len(conditions) * 4.5, len(BB_ORDER) * 1.8),
    )

    for row, bb in enumerate(BB_ORDER):
        for col, (tag, cond_label) in enumerate(conditions):
            feats, labels_arr, paths = _load_cluster_map(bb, ds, tag)
            if feats is None or sample_idx >= len(feats):
                for ax in axes[row, col * 2: col * 2 + 2]:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                continue

            feat = feats[sample_idx]
            img_path = str(paths[sample_idx])

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                for ax in axes[row, col * 2: col * 2 + 2]:
                    ax.axis('off')
                continue
            img_bgr = cv2.resize(img_bgr, (img_size, img_size))

            guide_gray = cv2.cvtColor(
                correct_illumination(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)),
                cv2.COLOR_RGB2GRAY,
            ).astype(np.float32) / 255.0
            result = cluster_spatial_features(feat, guide_gray, img_size, img_size)
            cluster_img = _render_cluster_overlay(img_bgr, result['labels'])

            ax_img = axes[row, col * 2]
            ax_clust = axes[row, col * 2 + 1]
            ax_img.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax_img.axis('off')
            ax_clust.imshow(cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB))
            ax_clust.axis('off')

            if row == 0:
                ax_img.set_title(f'{cond_label} · image', fontsize=7, pad=2)
                ax_clust.set_title(f'{cond_label} · clusters', fontsize=7, pad=2)
            if col == 0:
                ax_img.set_ylabel(BB_LABEL.get(bb, bb), fontsize=8)

    # Cluster legend
    legend_patches = [
        mpatches.Patch(color=CLUSTER_COLOR[k], label=CLUSTER_NAMES[k])
        for k in range(N_CLUSTERS)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=N_CLUSTERS,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=8)
    _save(fig, f'fig4_spatial_cluster_panels_{ds}')


# ── Figure 6: CAM × cluster overlay ──────────────────────────────────────────

def plot_cam_cluster_overlay(ds: str = 'WRD25', sample_idx: int = 0):
    from wood_spatial.core.gradcam import centroid_cam
    from wood_spatial.spatial.cluster_pipeline import cluster_spatial_features, correct_illumination
    from wood_spatial.analysis.feature_geometry import class_centroids

    conditions = [
        ('original', 'Clean'),
        ('blur_12', 'Blur (r=12)'),
        ('scratch_severe', 'Scratch'),
    ]
    img_size = 224

    fig, axes = plt.subplots(
        len(BB_ORDER), len(conditions) * 3,
        figsize=(len(conditions) * 5.5, len(BB_ORDER) * 1.8),
    )

    for row, bb in enumerate(BB_ORDER):
        # Use spatial features (N, C, H, W) from clean cache to compute per-class
        # spatial centroids — one centroid per class = mean pooled spatial feat
        feats_clean, labels_clean, paths_clean = _load_cluster_map(bb, ds, 'original')
        if feats_clean is None:
            for ax in axes[row]:
                ax.axis('off')
            continue
        # Spatially pool each image to a global vector, then centroid per class
        N, C, H_f, W_f = feats_clean.shape
        pooled = feats_clean.mean(axis=(2, 3))  # (N, C)
        # class_centroids returns dict {class: centroid_vector}
        centroids = class_centroids(pooled, labels_clean[:N], normalize=True)

        for col, (tag, cond_label) in enumerate(conditions):
            feats_s, labels_s, paths_s = _load_cluster_map(bb, ds, tag)
            if feats_s is None or sample_idx >= len(feats_s):
                for ax in axes[row, col * 3: col * 3 + 3]:
                    ax.axis('off')
                continue

            feat_sp = feats_s[sample_idx]
            img_path = str(paths_s[sample_idx])
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                for ax in axes[row, col * 3: col * 3 + 3]:
                    ax.axis('off')
                continue
            img_bgr = cv2.resize(img_bgr, (img_size, img_size))
            guide = cv2.cvtColor(
                correct_illumination(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)),
                cv2.COLOR_RGB2GRAY,
            ).astype(np.float32) / 255.0

            result = cluster_spatial_features(feat_sp, guide, img_size, img_size)
            # Use the most-populated class centroid as target for CAM
            label_for_img = labels_clean[sample_idx]
            if label_for_img not in centroids:
                label_for_img = next(iter(centroids))
            cam = centroid_cam(feat_sp, centroids[label_for_img], img_h=img_size, img_w=img_size)
            cam_resized = cam  # already upsampled by centroid_cam
            cluster_img = _render_cluster_overlay(img_bgr, result['labels'], alpha=0.5)

            ax_img = axes[row, col * 3]
            ax_cl = axes[row, col * 3 + 1]
            ax_cam = axes[row, col * 3 + 2]

            ax_img.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax_img.axis('off')
            ax_cl.imshow(cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB))
            ax_cl.axis('off')
            ax_cam.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax_cam.imshow(cam_resized, cmap='jet', alpha=0.5,
                          vmin=cam_resized.min(), vmax=cam_resized.max())
            ax_cam.axis('off')

            if row == 0:
                ax_img.set_title(f'{cond_label} · image', fontsize=7, pad=2)
                ax_cl.set_title(f'{cond_label} · clusters', fontsize=7, pad=2)
                ax_cam.set_title(f'{cond_label} · CAM', fontsize=7, pad=2)
        if col == 0:
            axes[row, 0].set_ylabel(BB_LABEL.get(bb, bb), fontsize=8)

    legend_patches = [
        mpatches.Patch(color=CLUSTER_COLOR[k], label=CLUSTER_NAMES[k])
        for k in range(N_CLUSTERS)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=N_CLUSTERS,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=8)
    _save(fig, f'fig6_cam_cluster_overlay_{ds}')


# ── Figure 3: Feature geometry panels ────────────────────────────────────────

def plot_feature_geometry_failure():
    df = pd.read_csv(V4_CSV / 'exp1b_feature_geometry.csv')
    partial = pd.read_csv(V4_CSV / 'exp7_full_record_partial_correlation.csv')
    hierarchical = pd.read_csv(V4_CSV / 'exp7_full_record_primary_r2.csv')
    df = df.copy()
    df['figure_group'] = df['perturbation'].map(_figure_group)
    pert = df.groupby('figure_group', as_index=False).agg(
        feature_drift=('feature_drift', 'mean'),
        delta_fgcs=('delta_fgcs', 'mean'),
        inter_collapse=('delta_inter', lambda x: -np.nanmean(x)),
        drop=('drop', 'mean'),
    )
    pert['label'] = pert['figure_group'].map(FIG_GROUP_LABEL).fillna(pert['figure_group'])
    pert = pert.sort_values('drop', ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    axes[0, 0].barh(pert['label'], pert['feature_drift'], color='steelblue')
    axes[0, 0].set_xlabel('Mean cosine distance (↑ worse)')

    axes[0, 1].barh(pert['label'], pert['delta_fgcs'], color='firebrick')
    axes[0, 1].set_xlabel('ΔFGCS (↑ worse)')

    axes[1, 0].barh(pert['label'], pert['inter_collapse'], color='darkorange')
    axes[1, 0].set_xlabel('-Δ inter-class distance (↑ = more collapsed)')

    for bb in BB_ORDER:
        sub = df[df['backbone'] == bb]
        axes[1, 1].scatter(sub['feature_drift'], sub['drop'], alpha=0.4, s=14,
                           color=BB_COLORS.get(bb), label=BB_LABEL.get(bb, bb))
    r_val = df[['feature_drift', 'drop']].dropna().corr().iloc[0, 1]
    partial_r = float(
        partial.loc[
            partial['metric'].eq('feature_drift'),
            'r_partial_full_record',
        ].iloc[0]
    )
    delta_r2 = float(
        hierarchical.loc[
            hierarchical['model'].eq('P1: + feature drift'), 'delta_r2'
        ].iloc[0]
    )
    annotation = (
        f'raw same-space r={r_val:.3f} (upper bound)\n'
        f'controlled full-record partial r={partial_r:.3f}; ΔR²={delta_r2:.3f}'
    )
    axes[1, 1].text(0.97, 0.05, annotation, transform=axes[1, 1].transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='0.85', alpha=0.82))
    axes[1, 1].set_xlabel('Feature drift (cosine distance)')
    axes[1, 1].set_ylabel('Accuracy drop')
    axes[1, 1].legend(fontsize=6.5, ncol=2, loc='upper left',
                      frameon=True, framealpha=0.78, borderpad=0.3)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.11, wspace=0.28, hspace=0.30)

    _save(fig, 'fig3_feature_geometry_failure')


# ── Analysis figures (Exp6) ───────────────────────────────────────────────────

def plot_multilevel_correlation():
    df = pd.read_csv(V4_CSV / 'exp6_multilevel_correlations.csv')
    label_map = {
        'feature_drift': 'Feature drift', 'inter_collapse': 'Inter-class collapse',
        'delta_intra': 'Δ intra-class var.', 'fsr_collapse': 'FSR collapse',
        'delta_fgcs': 'ΔFGCS', 'cam_shift_jsd': 'CAM shift (JSD)',
        'csi': 'CSI (spatial stability)', 'spatial_instability': 'Spatial instability',
        'sgi_clean': 'SGI (granularity)', 'cam_entropy_clean': 'CAM entropy',
        'csi_hungarian': 'CSI-H (matched stability)',
        'spatial_instability_hungarian': 'Matched spatial instability',
        'csi_permutation_gap': 'CSI permutation gap',
        'cam_shift_js_distance': 'CAM shift (JS distance)',
        'cam_shift_js_divergence': 'CAM shift (JS divergence)',
    }
    df['label'] = df['metric'].map(label_map).fillna(df['metric'])
    df = df.sort_values('r', key=lambda s: s.abs(), ascending=True)
    colors = ['steelblue' if r >= 0 else 'firebrick' for r in df['r']]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df['label'], df['r'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    for bar, (_, row) in zip(bars, df.iterrows()):
        sig = ''
        try:
            p = float(row['p'])
            if p < 0.001: sig = '***'
            elif p < 0.01: sig = '**'
            elif p < 0.05: sig = '*'
        except Exception:
            pass
        if sig:
            x = bar.get_width()
            ax.text(x + (0.01 if x >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                    sig, va='center', ha='left' if x >= 0 else 'right', fontsize=9)
    ax.set_xlabel('Pearson r with accuracy drop')
    _save(fig, 'fig_multilevel_correlations')


def plot_failure_profile_heatmap():
    df = pd.read_csv(V4_CSV / 'exp6_backbone_failure_profile.csv')
    col_labels = {
        'drop': 'Accuracy\ndrop', 'feature_drift': 'Feature\ndrift',
        'delta_fgcs': 'ΔFGCS', 'inter_collapse': 'Inter-class\ncollapse',
        'spatial_instability': 'Spatial\ninstability', 'cam_shift_jsd': 'CAM\nshift',
    }
    metrics = [m for m in col_labels if m in df.columns]
    df = df.set_index('backbone').loc[_ordered_backbones(df['backbone'])]
    mat = df[metrics].astype(float)
    mat_norm = (mat - mat.min()) / (mat.max() - mat.min()).replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(mat_norm.values, aspect='auto', cmap='magma', vmin=0, vmax=1)
    for i in range(len(mat.index)):
        for j, m in enumerate(metrics):
            ax.text(j, i, f'{mat.values[i, j]:.3f}', ha='center', va='center',
                    fontsize=7.5, color='white' if mat_norm.values[i, j] > 0.5 else 'black')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([col_labels[m] for m in metrics], ha='center')
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([BB_LABEL.get(bb, bb) for bb in mat.index])
    fig.colorbar(im, ax=ax, label='Normalized (0=best, 1=worst)', fraction=0.03, pad=0.02)
    _save(fig, 'fig_multilevel_failure_profile')


def plot_cam_shift_vs_drop():
    df = pd.read_csv(V4_CSV / 'exp6_multilevel_table.csv')
    sub = df[['cam_shift_jsd', 'drop', 'backbone']].dropna()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for bb in _ordered_backbones(sub['backbone']):
        g = sub[sub['backbone'] == bb]
        ax.scatter(g['cam_shift_jsd'], g['drop'], alpha=0.5, s=16,
                   color=BB_COLORS.get(bb), label=BB_LABEL.get(bb, bb))
    ax.set_xlabel('CAM shift JSD')
    ax.set_ylabel('Accuracy drop')
    ax.legend(fontsize=7, ncol=2)
    _save(fig, 'fig_cam_shift_vs_drop')


def plot_csi_vs_drop():
    df = pd.read_csv(V4_CSV / 'exp6_multilevel_table.csv')
    sub = df[['csi', 'drop', 'backbone']].dropna()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for bb in _ordered_backbones(sub['backbone']):
        g = sub[sub['backbone'] == bb]
        ax.scatter(g['csi'], g['drop'], alpha=0.5, s=16,
                   color=BB_COLORS.get(bb), label=BB_LABEL.get(bb, bb))
    ax.set_xlabel('Cluster Stability Index (CSI)')
    ax.set_ylabel('Accuracy drop')
    ax.legend(fontsize=7, ncol=2)
    _save(fig, 'fig_csi_vs_drop')


# ── CAM distribution stacked bar ─────────────────────────────────────────────

def plot_cam_distribution_bars():
    cd = pd.read_csv(V4_CSV / 'exp3_cam_distribution.csv')
    clean = cd[cd['condition'] == 'clean']
    cam_cols = [f'cam_pct_c{i}' for i in range(N_CLUSTERS)]
    m = clean.groupby('backbone')[cam_cols].mean()
    m.index = [BB_LABEL.get(bb, bb) for bb in m.index]
    m.columns = [CLUSTER_NAMES[i] for i in range(N_CLUSTERS)]
    m = m.loc[[BB_LABEL.get(bb, bb) for bb in BB_ORDER if BB_LABEL.get(bb, bb) in m.index]]
    colors = [CLUSTER_COLOR[i] for i in range(N_CLUSTERS)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(m))
    for i, col in enumerate(m.columns):
        ax.bar(m.index, m[col].values, bottom=bottom, color=colors[i], label=col)
        bottom += m[col].values
    ax.set_ylabel('Fraction of CAM activation')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xticks(range(len(m.index)))
    ax.set_xticklabels(m.index, rotation=20, ha='right')
    ax.set_ylim(0, 1.05)
    _save(fig, 'fig_cam_distribution_bars')


# ── Generate all ──────────────────────────────────────────────────────────────

# ── Figure 7: Hierarchical R² stacked bars ───────────────────────────────────

def plot_hierarchical_r2():
    df = pd.read_csv(V4_CSV / 'exp7_hierarchical_r2.csv').dropna(subset=['r2'])
    model_labels = {
        'M0: Perturbation type only': 'Perturbation\ntype',
        'M1: + Feature drift': '+ Feature\ndrift',
        'M2: + Full feature geometry': '+ Full\ngeometry',
        'M3: + Spatial clustering': '+ Spatial\nclustering',
        'M4: + Attention drift': '+ Attention\ndrift',
    }
    df['label'] = df['model'].map(model_labels).fillna(df['model'])
    colors = ['#d9d9d9', '#fc8d59', '#d73027', '#74add1', '#4575b4']

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(df['label'], df['r2'], color=colors[:len(df)], edgecolor='white', linewidth=0.5)
    # Annotate delta R²
    for i, (bar, (_, row)) in enumerate(zip(bars, df.iterrows())):
        r2 = row['r2']
        delta = row['delta_r2']
        ax.text(bar.get_x() + bar.get_width() / 2, r2 + 0.005,
                f'R²={r2:.3f}', ha='center', va='bottom', fontsize=8)
        if i > 0 and not np.isnan(delta):
            ax.text(bar.get_x() + bar.get_width() / 2, r2 / 2,
                    f'+{delta:.3f}' if delta >= 0.001 else f'{delta:.4f}',
                    ha='center', va='center', fontsize=7.5, color='white' if r2 > 0.3 else 'black')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('R² (prediction of accuracy drop)')
    ax.axhline(df.iloc[0]['r2'], color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    _save(fig, 'fig7_hierarchical_r2')


# ── Figure 8: ROC curve for unsupervised failure detection ───────────────────

def plot_roc_failure_detection():
    df_auc = pd.read_csv(V4_CSV / 'exp8_detector_auc.csv')
    df_op = pd.read_csv(V4_CSV / 'exp8_operating_points.csv')

    fig, axes = plt.subplots(
        1, 2, figsize=(11.8, 4.5),
        gridspec_kw={'width_ratios': [1.0, 1.08], 'wspace': 0.34},
    )

    # ROC-style: precision-recall via operating points for feature_drift
    ax = axes[0]
    ax.plot(df_op['recall'], df_op['precision'], 'o-', color='steelblue', markersize=4, label='Feature drift')
    ax.axhline(0.441, color='gray', linestyle='--', linewidth=0.8, label='Random classifier')
    best = df_op.loc[df_op['f1'].idxmax()]
    ax.scatter([best['recall']], [best['precision']], s=120, color='red', zorder=5,
               label=f'Best F1={best["f1"]:.3f} (t={best["threshold"]:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)

    # AUC comparison bar
    ax2 = axes[1]
    df_auc_s = df_auc.sort_values('auc_roc', ascending=True)
    detector_labels = {
        'feature_drift': 'Feature drift',
        'inter_collapse': 'Inter-class',
        'delta_fgcs': 'ΔFGCS',
        '1 - csi_bo': '1 - CSI-BO',
        'cam_shift_js_distance': 'CAM-JSD',
        '1 - csi_hungarian': '1 - CSI-H',
        'cam_shift_js_divergence': 'CAM-JS div.',
        'csi_permutation_gap': 'CSI perm. gap',
    }
    df_auc_s = df_auc_s.assign(label=df_auc_s['detector'].map(detector_labels).fillna(df_auc_s['detector']))
    bar_colors = ['firebrick' if 'feature' in d else 'steelblue' for d in df_auc_s['detector']]
    ax2.barh(df_auc_s['label'], df_auc_s['auc_roc'], color=bar_colors)
    ax2.axvline(0.5, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('AUC-ROC')
    ax2.tick_params(axis='y', labelsize=8)
    for i, (_, row) in enumerate(df_auc_s.iterrows()):
        ax2.text(row['auc_roc'] - 0.01, i, f'{row["auc_roc"]:.3f}',
                 va='center', ha='right', fontsize=8, color='white')
    ax2.set_xlim(0.4, 1.05)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.14, wspace=0.34)

    _save(fig, 'fig8_roc_failure_detection')


# ── Figure 9: Partial correlation comparison ─────────────────────────────────

def plot_partial_correlations():
    df = pd.read_csv(V4_CSV / 'exp7_partial_correlations.csv')
    label_map = {
        'feature_drift': 'Feature drift', 'delta_fgcs': 'ΔFGCS',
        'inter_collapse': 'Inter-class\ncollapse', 'delta_intra': 'Δ intra-class var.',
        'fsr_collapse': 'FSR collapse', 'csi': 'CSI',
        'cam_shift_jsd': 'CAM shift JSD', 'cam_entropy_clean': 'CAM entropy',
        'sgi_clean': 'SGI',
    }
    df['label'] = df['metric'].map(label_map).fillna(df['metric'])
    df = df.sort_values('r_raw', key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df['r_raw'], height=0.35, color='#fc8d59', alpha=0.8, label='Raw r')
    ax.barh(y + 0.2, df['r_partial'], height=0.35, color='#2b83ba', alpha=0.8,
            label='Partial r (controlling for perturbation type)')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df['label'])
    ax.set_xlabel('Pearson r with accuracy drop')
    ax.legend(fontsize=8)
    _save(fig, 'fig9_partial_correlations')


# ── Figure 10: CAM entropy change under perturbation ────────────────────────

def plot_cam_entropy_change():
    ent_path = V4_CSV / 'exp3_cam_entropy_full.csv'
    if ent_path.exists():
        df = pd.read_csv(ent_path)
        entropy_col = 'entropy_computed'
    else:
        ent_path = V4_CSV / 'exp3_cam_distribution.csv'
        if not ent_path.exists():
            logger.warning('No Exp3 entropy/distribution CSV found, skipping')
            return
        df = pd.read_csv(ent_path)
        entropy_col = 'entropy'
    clean = df[df.condition == 'clean'].groupby('backbone')[entropy_col].mean()
    pert = df[df.condition != 'clean'].groupby('backbone')[entropy_col].mean()

    bbs = [bb for bb in BB_ORDER if bb in clean.index]
    x = np.arange(len(bbs))
    c_vals = [clean[bb] for bb in bbs]
    p_vals = [pert.get(bb, np.nan) for bb in bbs]
    labels = [BB_LABEL.get(bb, bb) for bb in bbs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, c_vals, 0.35, color='#74add1', label='Clean')
    ax.bar(x + 0.2, p_vals, 0.35, color='#f46d43', label='Perturbed (mean)')
    for i, (c, p) in enumerate(zip(c_vals, p_vals)):
        if not np.isnan(p):
            delta = p - c
            ax.text(i, max(c, p) + 0.02, f'{delta:+.2f}', ha='center', fontsize=7.5,
                    color='green' if delta >= 0 else 'red')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('CAM entropy (bits)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 2.2)
    _save(fig, 'fig10_cam_entropy_change')


# ── Figure 11: Tier-A vs Tier-B cross-dataset validation ─────────────────────

# ── Figure 12: Cross-magnification spatial analysis ─────────────────────────

def plot_crossmag_spatial():
    sgi_path = V4_CSV / 'exp5b_sgi_by_mag.csv'
    csi_path = V4_CSV / 'exp5b_crossmag_csi.csv'
    if not sgi_path.exists() or not csi_path.exists():
        logger.warning('exp5b files not found, skipping')
        return

    df_sgi = pd.read_csv(sgi_path)
    df_csi = pd.read_csv(csi_path)

    fig, axes = plt.subplots(
        1, 2, figsize=(12.8, 4.7),
        gridspec_kw={'width_ratios': [1.08, 1.0], 'wspace': 0.34},
    )

    # Panel A: SGI per magnification per backbone
    ax = axes[0]
    mags = ['VN26_x10', 'VN26_x20', 'VN26_x50']
    mag_labels = {'VN26_x10': '×10', 'VN26_x20': '×20', 'VN26_x50': '×50'}
    x = np.arange(len(mags))
    width = 0.11
    sgi_pivot = df_sgi.groupby(['backbone', 'magnification']).sgi.mean().unstack('magnification')
    colors = list(BB_COLORS.values())
    legend_handles = []
    for i, bb in enumerate(BB_ORDER):
        if bb not in sgi_pivot.index:
            continue
        vals = [sgi_pivot.loc[bb].get(m, np.nan) for m in mags]
        offset = (i - 3) * width
        bars = ax.bar(x + offset, vals, width * 0.9, color=colors[i], alpha=0.8,
                      label=BB_LABEL.get(bb, bb))
        legend_handles.append(bars[0])
    ax.set_xticks(x)
    ax.set_xticklabels([mag_labels[m] for m in mags])
    ax.set_xlabel('Magnification')
    ax.set_ylabel('Mean SGI')

    # Panel B: Cross-mag CSI heatmap (backbone × mag pair)
    ax2 = axes[1]
    pairs = [('VN26_x10', 'VN26_x20'), ('VN26_x20', 'VN26_x50'), ('VN26_x10', 'VN26_x50')]
    pair_labels = ['×10↔×20', '×20↔×50', '×10↔×50']
    csi_mat = np.zeros((len(BB_ORDER), len(pairs)))
    for j, (ma, mb) in enumerate(pairs):
        sub = df_csi[(df_csi.mag_a == ma) & (df_csi.mag_b == mb)]
        for i, bb in enumerate(BB_ORDER):
            v = sub[sub.backbone == bb].csi.mean()
            csi_mat[i, j] = v if not np.isnan(v) else 0
    im = ax2.imshow(csi_mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.3)
    ax2.set_xticks(range(len(pairs)))
    ax2.set_xticklabels(pair_labels)
    ax2.set_yticks(range(len(BB_ORDER)))
    ax2.set_yticklabels([BB_LABEL.get(bb, bb) for bb in BB_ORDER])
    for i in range(len(BB_ORDER)):
        for j in range(len(pairs)):
            ax2.text(j, i, f'{csi_mat[i,j]:.3f}', ha='center', va='center',
                     fontsize=7.5, color='black')
    fig.colorbar(im, ax=ax2, label='CSI', fraction=0.03, pad=0.02)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.98, bottom=0.25)
    fig.legend(
        handles=legend_handles,
        labels=[BB_LABEL.get(bb, bb) for bb in BB_ORDER if bb in sgi_pivot.index],
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.34, 0.02),
        frameon=False,
        fontsize=7.2,
    )

    _save(fig, 'fig12_crossmag_spatial')


def plot_tierb_validation():
    path = V4_CSV / 'exp9_tier_comparison.csv'
    path_corr = V4_CSV / 'exp9_tierb_correlations.csv'
    if not path.exists():
        logger.warning('exp9_tier_comparison.csv not found, skipping')
        return

    df_cmp = pd.read_csv(path)
    df_tierb = pd.read_csv(V4_CSV / 'exp9_tierb_geometry.csv').dropna(subset=['drop', 'feature_drift'])
    df_tiera = pd.read_csv(V4_CSV / 'exp1b_feature_geometry.csv').dropna(subset=['drop', 'feature_drift'])

    fig, axes = plt.subplots(
        1, 2, figsize=(12.8, 4.7),
        gridspec_kw={'width_ratios': [1.05, 1.0], 'wspace': 0.42},
    )

    # Left: scatter both tiers on same plot
    ax = axes[0]
    ax.scatter(df_tiera['feature_drift'], df_tiera['drop'],
               alpha=0.3, s=12, color='steelblue', label='Tier-A (controlled, n=714)')
    ax.scatter(df_tierb['feature_drift'], df_tierb['drop'],
               alpha=0.6, s=24, color='firebrick', marker='^', label='Tier-B (wild)')
    from scipy.stats import pearsonr
    for tier, df_t, col in [('A', df_tiera, 'steelblue'), ('B', df_tierb, 'firebrick')]:
        sub = df_t.dropna(subset=['feature_drift', 'drop'])
        if len(sub) < 3:
            continue
        r, _ = pearsonr(sub['feature_drift'].values, sub['drop'].values)
        m, b = np.polyfit(sub['feature_drift'].values, sub['drop'].values, 1)
        x_line = np.linspace(sub['feature_drift'].min(), sub['feature_drift'].max(), 50)
        ax.plot(x_line, m * x_line + b, color=col, linewidth=1.5,
                linestyle='--', alpha=0.8, label=f'Tier-{tier} fit (r={r:.3f})')
    ax.set_xlabel('Feature drift (cosine distance)')
    ax.set_ylabel('Accuracy drop')
    ax.legend(fontsize=7.5)

    # Right: r comparison bar
    ax2 = axes[1]
    if path_corr.exists():
        df_corr = pd.read_csv(path_corr)
        all_rows = list(df_corr[['dataset', 'r', 'n']].itertuples(index=False))
        # Add Tier-A overall
        r_a, _ = pearsonr(df_tiera['feature_drift'].values, df_tiera['drop'].values)
        tiera_rows = [('WRD25 (A)', 0.951, 238), ('DTSR14 (A)', 0.922, 238), ('PCA11 (A)', 0.930, 238)]
        labels = [row[0] for row in tiera_rows] + [row.dataset for row in all_rows]
        rs = [row[1] for row in tiera_rows] + [row.r for row in all_rows]
        colors = ['steelblue'] * len(tiera_rows) + ['firebrick'] * len(all_rows)
        ax2.barh(labels, rs, color=colors)
        ax2.axvline(0.5, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_xlabel('Pearson r (feature_drift vs accuracy drop)')
        ax2.set_xlim(0, 1.05)
        ax2.tick_params(axis='y', labelsize=8)
        for i, r in enumerate(rs):
            ax2.text(r + 0.01, i, f'{r:.3f}', va='center', fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.15)

    _save(fig, 'fig11_tierb_validation')


def generate_all(skip_slow: bool = False):
    logger.info('Generating quantitative figures...')
    plot_accuracy_heatmap()
    plot_severity_curves()
    plot_feature_geometry_failure()
    plot_nemenyi_cd_diagram()
    plot_multilevel_correlation()
    plot_failure_profile_heatmap()
    plot_cam_shift_vs_drop()
    plot_csi_vs_drop()
    plot_cam_distribution_bars()

    logger.info('Generating Exp5b/7/8/9/10/11/12 figures...')
    plot_crossmag_spatial()
    plot_hierarchical_r2()
    plot_roc_failure_detection()
    plot_partial_correlations()
    plot_cam_entropy_change()
    plot_tierb_validation()

    if not skip_slow:
        logger.info('Generating high-resolution VN26 qualitative panels...')
        try:
            from wood_spatial.figures.regen_figs import plot_fig4, plot_fig6
            plot_fig4()
            plot_fig6()
        except Exception as e:
            logger.warning('VN26 high-res fig4/fig6 failed: %s', e)
        try:
            from wood_spatial.figures.regen_fig4b import plot_fig4b_perturbation
            plot_fig4b_perturbation()
        except Exception as e:
            logger.warning('VN26 x20 perturbation figure failed: %s', e)


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    p = argparse.ArgumentParser()
    p.add_argument('--skip-slow', action='store_true', help='Skip cluster panel figures')
    args = p.parse_args()
    generate_all(skip_slow=args.skip_slow)


if __name__ == '__main__':
    main()
