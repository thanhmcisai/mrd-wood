"""
Wood Spatial — Experiment 2: Spatial Feature Clustering
========================================================
7 backbones, spatial features → K-Means k=3 → Guided Filter → Smart Hole Filling.
Compute CSI, SGI, ARI, Silhouette under clean and perturbed conditions.

Usage:
    python -m wood_spatial.experiments.exp2_spatial_clustering
"""
import logging
import time
import argparse

import cv2
import numpy as np
import pandas as pd

from wood_spatial.config import (
    ALL_DATASETS, TIER_A, BB_ORDER, V4_CSV, V4_FIGURES,
    N_CLUSTERS, SPATIAL_MAX_IMAGES, BACKBONE_CONFIGS,
)
from wood_spatial.core.cache import load_spatial_cache
from wood_spatial.core.perturbations import cache_tag_for
from wood_spatial.core.dataset import denormalize
from wood_spatial.spatial.cluster_pipeline import (
    cluster_spatial_features, make_guide_from_tensor, correct_illumination,
)
from wood_spatial.spatial.metrics import (
    compute_csi, compute_csi_metrics, compute_sgi, compute_ari, compute_silhouette,
    compute_sgi_per_cluster, compute_csi_per_cluster,
    downsample_labels_to_feat,
)
from wood_spatial.analysis.statistical_tests import safe_pearsonr

logger = logging.getLogger(__name__)

# Perturbation conditions for spatial analysis
SPATIAL_PERTURBATIONS = [
    ('gaussian_blur', 4), ('gaussian_blur', 8), ('gaussian_blur', 12),
    ('defocus_blur', 5), ('defocus_blur', 11),
    ('resize', 2.00),
    ('jpeg', 10), ('jpeg', 50),
    ('rotation', 45), ('rotation', 180),
    ('red_channel_shift', -45), ('red_channel_shift', 45),
    ('green_channel_shift', -45), ('green_channel_shift', 45),
    ('blue_channel_shift', -45), ('blue_channel_shift', 45),
    ('gaussian_noise', 0.10), ('shot_noise', 15), ('impulse_noise', 0.05),
    ('motion_blur', 15), ('zoom_blur', 1.20),
    ('contrast', 0.50), ('contrast', 1.50), ('pixelate', 0.25),
    ('scratch', 'severe'),
    ('compound', 'mild'), ('compound', 'severe'),
    ('compound_optical', 'severe'),
    ('compound_digital', 'severe'),
    ('compound_field', 'severe'),
]

SGI_COLUMNS = (
    ['dataset', 'backbone', 'condition', 'image_idx', 'sgi']
    + [f'sgi_c{k}' for k in range(N_CLUSTERS)]
)
CSI_COLUMNS = (
    ['dataset', 'backbone', 'perturbation', 'severity', 'image_idx',
     'csi', 'csi_bo', 'csi_hungarian', 'csi_permutation_gap']
    + [f'csi_c{k}' for k in range(N_CLUSTERS)]
)
ARI_COLUMNS = ['dataset', 'backbone_a', 'backbone_b', 'ari_mean', 'ari_std', 'n_images']
SIL_COLUMNS = ['dataset', 'backbone', 'condition', 'image_idx', 'silhouette']


def _safe_part_name(*parts: str) -> str:
    raw = '__'.join(str(p) for p in parts)
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in raw)


def _checkpoint_paths(ds_name: str) -> dict:
    ckpt_dir = V4_CSV / 'exp2_checkpoints'
    stem = _safe_part_name(ds_name)
    return {
        'dir': ckpt_dir,
        'sgi': ckpt_dir / f'{stem}__sgi.csv',
        'csi': ckpt_dir / f'{stem}__csi.csv',
        'ari': ckpt_dir / f'{stem}__ari.csv',
        'silhouette': ckpt_dir / f'{stem}__silhouette.csv',
    }


def _checkpoint_complete(ds_name: str) -> bool:
    paths = _checkpoint_paths(ds_name)
    if not (
        paths['sgi'].exists() and paths['csi'].exists()
        and paths['ari'].exists() and paths['silhouette'].exists()
    ):
        return False
    try:
        csi = pd.read_csv(paths['csi'], usecols=['perturbation', 'severity'])
    except Exception:
        return False
    expected = {(p, str(v)) for p, v in SPATIAL_PERTURBATIONS}
    observed = set(zip(csi['perturbation'].astype(str), csi['severity'].astype(str)))
    return expected.issubset(observed)


def _read_checkpoint_parts(datasets: list) -> dict:
    frames = {'sgi': [], 'csi': [], 'ari': [], 'silhouette': []}
    for ds_name in datasets:
        if not _checkpoint_complete(ds_name):
            continue
        paths = _checkpoint_paths(ds_name)
        frames['sgi'].append(pd.read_csv(paths['sgi']))
        frames['csi'].append(pd.read_csv(paths['csi']))
        frames['ari'].append(pd.read_csv(paths['ari']))
        frames['silhouette'].append(pd.read_csv(paths['silhouette']))
    return {
        key: pd.concat(vals, ignore_index=True) if vals else pd.DataFrame()
        for key, vals in frames.items()
    }


def _load_guides(paths: np.ndarray, img_size: int) -> list:
    """Load images and compute grayscale guides."""
    guides = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            guides.append(np.zeros((img_size, img_size), dtype=np.float32))
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))
        img_corrected = correct_illumination(img_resized)
        guide = cv2.cvtColor(img_corrected, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        guides.append(guide)
    return guides


def _cluster_all(
    spatial_features: list,
    guides: list,
    img_size: int,
) -> list:
    """Run clustering pipeline on all images."""
    results = []
    for feat, guide in zip(spatial_features, guides):
        result = cluster_spatial_features(feat, guide, img_size, img_size)
        results.append(result)
    return results


def run_exp2(
    datasets: list = None,
    backbones: list = None,
    max_images: int = None,
    save: bool = True,
    resume: bool = True,
    final_save: bool = True,
) -> dict:
    """
    Run Experiment 2: Spatial Feature Clustering Analysis.

    Returns dict of DataFrames: 'sgi', 'csi', 'ari', 'silhouette'.
    """
    if datasets is None:
        datasets = TIER_A
    if backbones is None:
        backbones = BB_ORDER
    if max_images is None:
        max_images = SPATIAL_MAX_IMAGES

    sgi_rows = []
    csi_rows = []
    ari_rows = []
    sil_rows = []

    if save:
        (V4_CSV / 'exp2_checkpoints').mkdir(parents=True, exist_ok=True)

    for ds_name in datasets:
        if save and resume and _checkpoint_complete(ds_name):
            logger.info('Exp 2 checkpoint hit: %s', ds_name)
            continue

        ds_sgi_rows = []
        ds_csi_rows = []
        ds_ari_rows = []
        ds_sil_rows = []
        img_size = 224  # standard output size

        # ── Step 1: Clean clustering per backbone ─────────────────────────
        logger.info('=== Exp 2: %s — Clean clustering ===', ds_name)

        clean_clusters = {}  # {bb: [{'labels': (H,W)}]}

        for bb in backbones:
            cfg = BACKBONE_CONFIGS[bb]
            try:
                feats, labels, paths = load_spatial_cache(bb, ds_name, 'original')
            except FileNotFoundError:
                logger.warning('Missing spatial cache: %s/%s/original', bb, ds_name)
                continue

            n = min(len(feats), max_images)
            feats = list(feats[:n])
            paths_sub = paths[:n]

            guides = _load_guides(paths_sub, img_size)
            clusters = _cluster_all(feats, guides, img_size)

            # Compute SGI and silhouette before stripping large arrays
            for i, result in enumerate(clusters):
                sgi = compute_sgi(result['labels'])
                sgi_per = compute_sgi_per_cluster(result['labels'])
                ds_sgi_rows.append({
                    'dataset': ds_name, 'backbone': bb, 'condition': 'clean',
                    'image_idx': i, 'sgi': sgi,
                    **{f'sgi_c{k}': v for k, v in sgi_per.items()},
                })

            # Silhouette (on feature-resolution labels)
            for i, (feat, result) in enumerate(zip(feats, clusters)):
                C, H_f, W_f = feat.shape
                labels_ds = downsample_labels_to_feat(result['labels'], H_f, W_f)
                sil = compute_silhouette(feat, labels_ds)
                ds_sil_rows.append({
                    'dataset': ds_name, 'backbone': bb, 'condition': 'clean',
                    'image_idx': i, 'silhouette': sil,
                })

            # Strip large arrays — only keep 'labels' for CSI/ARI downstream
            labels_only = [{'labels': r['labels']} for r in clusters]
            clean_clusters[bb] = labels_only
            del clusters, feats

            logger.info('  %s: %d images clustered, mean SGI=%.6f',
                        bb, n, np.mean([r['sgi'] for r in ds_sgi_rows
                                        if r['backbone'] == bb and r['dataset'] == ds_name
                                        and r['condition'] == 'clean']))

        # ── Step 2: Cross-architecture ARI ────────────────────────────────
        logger.info('=== Exp 2: %s — Cross-architecture ARI ===', ds_name)

        bb_list = [b for b in backbones if b in clean_clusters]
        for i, bb_a in enumerate(bb_list):
            for bb_b in bb_list[i + 1:]:
                n_common = min(len(clean_clusters[bb_a]), len(clean_clusters[bb_b]))
                ari_vals = []
                for idx in range(n_common):
                    labels_a = clean_clusters[bb_a][idx]['labels']
                    labels_b = clean_clusters[bb_b][idx]['labels']
                    ari = compute_ari(labels_a, labels_b)
                    ari_vals.append(ari)
                ds_ari_rows.append({
                    'dataset': ds_name, 'backbone_a': bb_a, 'backbone_b': bb_b,
                    'ari_mean': np.mean(ari_vals), 'ari_std': np.std(ari_vals),
                    'n_images': n_common,
                })

        # ── Step 3: CSI under perturbation ────────────────────────────────
        logger.info('=== Exp 2: %s — CSI under perturbation ===', ds_name)

        BATCH = 50  # process images in batches to limit memory usage
        for pert_name, value in SPATIAL_PERTURBATIONS:
            tag = cache_tag_for(pert_name, value)

            for bb in backbones:
                if bb not in clean_clusters:
                    continue
                try:
                    feats_p, labels_p, paths_p = load_spatial_cache(bb, ds_name, tag)
                except FileNotFoundError:
                    logger.debug('Missing spatial: %s/%s/%s', bb, ds_name, tag)
                    continue

                n = min(len(feats_p), len(clean_clusters[bb]))
                logger.info('  CSI: %s/%s/%s/%s (%d imgs)', ds_name, bb, pert_name, value, n)

                for batch_start in range(0, n, BATCH):
                    batch_end = min(batch_start + BATCH, n)
                    guides = _load_guides(paths_p[batch_start:batch_end], img_size)
                    clusters_p = _cluster_all(list(feats_p[batch_start:batch_end]), guides, img_size)

                    for idx, i in enumerate(range(batch_start, batch_end)):
                        labels_c = clean_clusters[bb][i]['labels']
                        labels_pt = clusters_p[idx]['labels']
                        csi_metrics = compute_csi_metrics(labels_c, labels_pt)
                        csi_per = compute_csi_per_cluster(labels_c, labels_pt)
                        ds_csi_rows.append({
                            'dataset': ds_name, 'backbone': bb,
                            'perturbation': pert_name, 'severity': value,
                            'image_idx': i,
                            'csi': csi_metrics['csi'],
                            'csi_bo': csi_metrics['csi_bo'],
                            'csi_hungarian': csi_metrics['csi_hungarian'],
                            'csi_permutation_gap': csi_metrics['csi_permutation_gap'],
                            **{f'csi_c{k}': v for k, v in csi_per.items()},
                        })

                        # SGI for perturbed
                        sgi_p = compute_sgi(labels_pt)
                        ds_sgi_rows.append({
                            'dataset': ds_name, 'backbone': bb,
                            'condition': f'{pert_name}_{value}',
                            'image_idx': i, 'sgi': sgi_p,
                        })

        if save:
            ckpt = _checkpoint_paths(ds_name)
            pd.DataFrame(ds_sgi_rows, columns=SGI_COLUMNS).to_csv(ckpt['sgi'], index=False)
            pd.DataFrame(ds_csi_rows, columns=CSI_COLUMNS).to_csv(ckpt['csi'], index=False)
            pd.DataFrame(ds_ari_rows, columns=ARI_COLUMNS).to_csv(ckpt['ari'], index=False)
            pd.DataFrame(ds_sil_rows, columns=SIL_COLUMNS).to_csv(ckpt['silhouette'], index=False)
            logger.info('Exp 2 checkpoint saved: %s', ds_name)
        else:
            sgi_rows.extend(ds_sgi_rows)
            csi_rows.extend(ds_csi_rows)
            ari_rows.extend(ds_ari_rows)
            sil_rows.extend(ds_sil_rows)

    # ── Compile DataFrames ────────────────────────────────────────────────
    if save:
        checkpoint_frames = _read_checkpoint_parts(datasets)
        df_sgi = checkpoint_frames['sgi']
        df_csi = checkpoint_frames['csi']
        df_ari = checkpoint_frames['ari']
        df_sil = checkpoint_frames['silhouette']
    else:
        df_sgi = pd.DataFrame(sgi_rows)
        df_csi = pd.DataFrame(csi_rows)
        df_ari = pd.DataFrame(ari_rows)
        df_sil = pd.DataFrame(sil_rows)

    results = {
        'sgi': df_sgi,
        'csi': df_csi,
        'ari': df_ari,
        'silhouette': df_sil,
    }

    if save and final_save:
        df_sgi.to_csv(V4_CSV / 'exp2_sgi.csv', index=False)
        df_csi.to_csv(V4_CSV / 'exp2_csi.csv', index=False)
        df_ari.to_csv(V4_CSV / 'exp2_ari_heatmap.csv', index=False)
        df_sil.to_csv(V4_CSV / 'exp2_silhouette.csv', index=False)
        logger.info('Saved Exp 2 results to %s', V4_CSV)

    return results


def analyze_sgi_vs_robustness(exp1_results: dict, exp2_results: dict):
    """
    Correlate SGI with Exp 1 robustness — answers RQ3.

    Parameters
    ----------
    exp1_results : dict from run_exp1()
    exp2_results : dict from run_exp2()

    Returns
    -------
    dict with correlation results
    """
    df_acc = exp1_results['accuracy']
    df_sgi = exp2_results['sgi']

    # Mean SGI per backbone (clean)
    sgi_clean = df_sgi[df_sgi['condition'] == 'clean'].groupby('backbone')['sgi'].mean()

    # Mean accuracy drop per backbone (across all perturbations)
    pert_acc = df_acc[df_acc['perturbation'] != 'clean']
    mean_drop = pert_acc.groupby('backbone')['drop'].mean()

    # Merge
    common = list(set(sgi_clean.index) & set(mean_drop.index))
    if len(common) < 3:
        return {'r': np.nan, 'p': np.nan, 'n': len(common)}

    x = [sgi_clean[bb] for bb in common]
    y = [mean_drop[bb] for bb in common]
    r, p = safe_pearsonr(x, y)

    return {
        'r': r, 'p': p, 'n': len(common),
        'backbone_sgi': {bb: sgi_clean[bb] for bb in common},
        'backbone_drop': {bb: mean_drop[bb] for bb in common},
    }


def main():
    parser = argparse.ArgumentParser(description='Experiment 2 spatial clustering.')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--backbones', nargs='+', default=None)
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--checkpoint-only', action='store_true',
                        help='Write per-dataset checkpoints but do not update final Exp2 CSV files.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    t0 = time.time()
    results = run_exp2(
        datasets=args.datasets,
        backbones=args.backbones,
        max_images=args.max_images,
        resume=not args.no_resume,
        final_save=not args.checkpoint_only,
    )
    logger.info('Exp 2 done in %.1f min', (time.time() - t0) / 60)

    # Summary
    df_sgi = results['sgi']
    clean_sgi = df_sgi[df_sgi['condition'] == 'clean']
    print('\n=== Mean SGI by Backbone (clean) ===')
    for bb in BB_ORDER:
        vals = clean_sgi[clean_sgi['backbone'] == bb]['sgi']
        if len(vals) > 0:
            print(f'  {bb:20s}: {vals.mean():.6f} +/- {vals.std():.6f}')

    df_ari = results['ari']
    if len(df_ari) > 0:
        print('\n=== Cross-Architecture ARI (mean) ===')
        for _, row in df_ari.iterrows():
            print(f"  {row['backbone_a']:15s} vs {row['backbone_b']:15s}: "
                  f"{row['ari_mean']:.3f} +/- {row['ari_std']:.3f}")


if __name__ == '__main__':
    main()
