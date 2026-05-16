"""
Wood Spatial — Experiment 3: Grad-CAM × Cluster Overlap
=========================================================
Show which anatomical structures each backbone uses for classification.
Track how attention shifts under perturbation.

Uses Centroid-CAM (gradient-free) as primary method.
True Grad-CAM via GradCAMExtractor as secondary.

Usage:
    python -m wood_spatial.experiments.exp3_gradcam_cluster
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
from wood_spatial.core.cache import load_cache, load_spatial_cache
from wood_spatial.core.perturbations import cache_tag_for
from wood_spatial.core.gradcam import (
    compute_spatial_centroids, centroid_cam,
    save_gradcam_cache, load_gradcam_cache,
)
from wood_spatial.spatial.cluster_pipeline import (
    cluster_spatial_features, correct_illumination,
)
from wood_spatial.spatial.metrics import (
    cluster_cam_distribution, cam_shift_index, cam_js_divergence, cam_entropy,
)

logger = logging.getLogger(__name__)

# Key perturbations for CAM shift analysis
CAM_PERTURBATIONS = [
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

DIST_COLUMNS = (
    ['dataset', 'backbone', 'condition', 'image_idx', 'label']
    + [f'cam_pct_c{k}' for k in range(N_CLUSTERS)]
    + ['entropy']
)
SHIFT_COLUMNS = [
    'dataset', 'backbone', 'perturbation', 'severity', 'image_idx',
    'cam_shift_jsd', 'cam_shift_js_distance', 'cam_shift_js_divergence',
]
ENTROPY_COLUMNS = ['dataset', 'backbone', 'condition', 'image_idx', 'entropy']


def _safe_part_name(*parts: str) -> str:
    raw = '__'.join(str(p) for p in parts)
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in raw)


def _checkpoint_paths(ds_name: str, backbone: str) -> dict:
    ckpt_dir = V4_CSV / 'exp3_checkpoints'
    stem = _safe_part_name(ds_name, backbone)
    return {
        'dir': ckpt_dir,
        'dist': ckpt_dir / f'{stem}__cam_distribution.csv',
        'shift': ckpt_dir / f'{stem}__cam_shift.csv',
        'entropy': ckpt_dir / f'{stem}__cam_entropy.csv',
    }


def _checkpoint_complete(ds_name: str, backbone: str) -> bool:
    paths = _checkpoint_paths(ds_name, backbone)
    if not (paths['dist'].exists() and paths['shift'].exists() and paths['entropy'].exists()):
        return False
    try:
        shift = pd.read_csv(paths['shift'], usecols=['perturbation', 'severity'])
    except Exception:
        return False
    expected = {(p, str(v)) for p, v in CAM_PERTURBATIONS}
    observed = set(zip(shift['perturbation'].astype(str), shift['severity'].astype(str)))
    return expected.issubset(observed)


def _read_checkpoint_parts(datasets: list, backbones: list) -> dict:
    frames = {'cam_dist': [], 'cam_shift': [], 'cam_entropy': []}
    for ds_name in datasets:
        for bb in backbones:
            if not _checkpoint_complete(ds_name, bb):
                continue
            paths = _checkpoint_paths(ds_name, bb)
            frames['cam_dist'].append(pd.read_csv(paths['dist']))
            frames['cam_shift'].append(pd.read_csv(paths['shift']))
            frames['cam_entropy'].append(pd.read_csv(paths['entropy']))
    return {
        key: pd.concat(vals, ignore_index=True) if vals else pd.DataFrame()
        for key, vals in frames.items()
    }


def run_exp3(
    datasets: list = None,
    backbones: list = None,
    max_images: int = None,
    save: bool = True,
    resume: bool = True,
    final_save: bool = True,
) -> dict:
    """
    Run Experiment 3: Grad-CAM × Cluster Overlap.

    Returns dict of DataFrames: 'cam_dist', 'cam_shift', 'cam_entropy'.
    """
    if datasets is None:
        datasets = TIER_A
    if backbones is None:
        backbones = BB_ORDER
    if max_images is None:
        max_images = SPATIAL_MAX_IMAGES

    dist_rows = []
    shift_rows = []
    entropy_rows = []

    if save:
        (V4_CSV / 'exp3_checkpoints').mkdir(parents=True, exist_ok=True)

    for ds_name in datasets:
        img_size = 224

        for bb in backbones:
            cfg = BACKBONE_CONFIGS[bb]
            if save and resume and _checkpoint_complete(ds_name, bb):
                logger.info('Exp 3 checkpoint hit: %s/%s', bb, ds_name)
                continue

            pair_dist_rows = []
            pair_shift_rows = []
            pair_entropy_rows = []

            # ── Load spatial features + cluster labels ────────────────────
            try:
                feats_spatial, labels_sp, paths = load_spatial_cache(bb, ds_name, 'original')
            except FileNotFoundError:
                logger.warning('Missing spatial cache: %s/%s', bb, ds_name)
                continue

            n = min(len(feats_spatial), max_images)
            feats_spatial = list(feats_spatial[:n])
            labels_sp = labels_sp[:n]
            paths = paths[:n]

            # ── Compute spatial centroids ─────────────────────────────────
            centroids = compute_spatial_centroids(feats_spatial, labels_sp)

            # ── Load guides and run clustering ────────────────────────────
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

            # ── Clean: Centroid-CAM + cluster overlay ─────────────────────
            logger.info('Exp 3: %s/%s — Centroid-CAM (clean)', bb, ds_name)

            clean_dists = []  # Store for CAM shift computation

            for i in range(n):
                feat = feats_spatial[i]
                label = int(labels_sp[i])
                guide = guides[i]

                # Cluster
                cluster_result = cluster_spatial_features(feat, guide, img_size, img_size)
                cluster_labels = cluster_result['labels']

                # Centroid-CAM
                if label < len(centroids):
                    cam = centroid_cam(feat, centroids[label], img_size, img_size)
                else:
                    cam = np.zeros((img_size, img_size), dtype=np.float32)

                # CAM-Cluster distribution
                dist = cluster_cam_distribution(cam, cluster_labels)
                clean_dists.append(dist)
                ent = cam_entropy(dist)

                pair_dist_rows.append({
                    'dataset': ds_name, 'backbone': bb, 'condition': 'clean',
                    'image_idx': i, 'label': label,
                    **{f'cam_pct_c{k}': dist[k] for k in range(N_CLUSTERS)},
                    'entropy': ent,
                })

                pair_entropy_rows.append({
                    'dataset': ds_name, 'backbone': bb, 'condition': 'clean',
                    'image_idx': i, 'entropy': ent,
                })

            # ── Perturbed: CAM shift ──────────────────────────────────────
            for pert_name, value in CAM_PERTURBATIONS:
                tag = cache_tag_for(pert_name, value)
                try:
                    feats_pert, labels_pert, paths_pert = load_spatial_cache(bb, ds_name, tag)
                except FileNotFoundError:
                    continue

                n_pert = min(len(feats_pert), n)

                # Compute perturbed centroids
                centroids_pert = compute_spatial_centroids(
                    list(feats_pert[:n_pert]), labels_pert[:n_pert],
                )

                for i in range(n_pert):
                    feat_p = feats_pert[i]
                    label = int(labels_pert[i])
                    guide = guides[i] if i < len(guides) else np.zeros(
                        (img_size, img_size), dtype=np.float32)

                    # Cluster perturbed
                    cluster_p = cluster_spatial_features(feat_p, guide, img_size, img_size)
                    labels_p = cluster_p['labels']

                    # CAM perturbed
                    if label < len(centroids_pert):
                        cam_p = centroid_cam(feat_p, centroids_pert[label], img_size, img_size)
                    else:
                        cam_p = np.zeros((img_size, img_size), dtype=np.float32)

                    dist_p = cluster_cam_distribution(cam_p, labels_p)
                    ent_p = cam_entropy(dist_p)

                    pair_dist_rows.append({
                        'dataset': ds_name, 'backbone': bb,
                        'condition': f'{pert_name}_{value}',
                        'image_idx': i, 'label': label,
                        **{f'cam_pct_c{k}': dist_p[k] for k in range(N_CLUSTERS)},
                        'entropy': ent_p,
                    })

                    # CAM Shift from clean
                    if i < len(clean_dists):
                        jsd = cam_shift_index(clean_dists[i], dist_p)
                        pair_shift_rows.append({
                            'dataset': ds_name, 'backbone': bb,
                            'perturbation': pert_name, 'severity': value,
                            'image_idx': i, 'cam_shift_jsd': jsd,
                            'cam_shift_js_distance': jsd,
                            'cam_shift_js_divergence': cam_js_divergence(clean_dists[i], dist_p),
                        })

                logger.info('  %s/%s/%s: %d images processed', bb, ds_name, tag, n_pert)

            if save:
                ckpt = _checkpoint_paths(ds_name, bb)
                pd.DataFrame(pair_dist_rows, columns=DIST_COLUMNS).to_csv(ckpt['dist'], index=False)
                pd.DataFrame(pair_shift_rows, columns=SHIFT_COLUMNS).to_csv(ckpt['shift'], index=False)
                pd.DataFrame(pair_entropy_rows, columns=ENTROPY_COLUMNS).to_csv(ckpt['entropy'], index=False)
                logger.info('Exp 3 checkpoint saved: %s/%s', bb, ds_name)
            else:
                dist_rows.extend(pair_dist_rows)
                shift_rows.extend(pair_shift_rows)
                entropy_rows.extend(pair_entropy_rows)

    # ── Compile ───────────────────────────────────────────────────────────
    if save:
        checkpoint_frames = _read_checkpoint_parts(datasets, backbones)
        df_dist = checkpoint_frames['cam_dist']
        df_shift = checkpoint_frames['cam_shift']
        df_entropy = checkpoint_frames['cam_entropy']
    else:
        df_dist = pd.DataFrame(dist_rows)
        df_shift = pd.DataFrame(shift_rows)
        df_entropy = pd.DataFrame(entropy_rows)

    results = {
        'cam_dist': df_dist,
        'cam_shift': df_shift,
        'cam_entropy': df_entropy,
    }

    if save and final_save:
        df_dist.to_csv(V4_CSV / 'exp3_cam_distribution.csv', index=False)
        df_shift.to_csv(V4_CSV / 'exp3_cam_shift.csv', index=False)
        df_entropy.to_csv(V4_CSV / 'exp3_cam_entropy.csv', index=False)
        logger.info('Saved Exp 3 results to %s', V4_CSV)

    return results


def main():
    parser = argparse.ArgumentParser(description='Experiment 3 CAM x cluster analysis.')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--backbones', nargs='+', default=None)
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--checkpoint-only', action='store_true',
                        help='Write per-task checkpoints but do not update final Exp3 CSV files.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    t0 = time.time()
    results = run_exp3(
        datasets=args.datasets,
        backbones=args.backbones,
        max_images=args.max_images,
        resume=not args.no_resume,
        final_save=not args.checkpoint_only,
    )
    logger.info('Exp 3 done in %.1f min', (time.time() - t0) / 60)

    # Summary
    df = results['cam_dist']
    clean = df[df['condition'] == 'clean']
    print('\n=== Mean CAM Distribution (clean) ===')
    for bb in BB_ORDER:
        sub = clean[clean['backbone'] == bb]
        if len(sub) == 0:
            continue
        pcts = [sub[f'cam_pct_c{k}'].mean() for k in range(N_CLUSTERS)]
        ent = sub['entropy'].mean()
        pct_text = ' '.join(f'C{k + 1}={v:.2f}' for k, v in enumerate(pcts))
        print(f'  {bb:20s}: {pct_text} | H={ent:.2f}')

    df_shift = results['cam_shift']
    if len(df_shift) > 0:
        print('\n=== Mean CAM Shift (JSD) ===')
        for bb in BB_ORDER:
            sub = df_shift[df_shift['backbone'] == bb]
            if len(sub) > 0:
                print(f'  {bb:20s}: {sub["cam_shift_jsd"].mean():.4f}')


if __name__ == '__main__':
    main()
