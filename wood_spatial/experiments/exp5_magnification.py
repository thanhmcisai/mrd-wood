"""
Wood Spatial — Experiment 5: Cross-Magnification Stability (VN26)
==================================================================
Magnification change as natural perturbation.
Spatial clustering across magnifications.

Usage:
    python -m wood_spatial.experiments.exp5_magnification
"""
import logging
import time

import cv2
import numpy as np
import pandas as pd

from wood_spatial.config import (
    ALL_DATASETS, TIER_C, BB_ORDER, V4_CSV,
    N_CLUSTERS, SPATIAL_MAX_IMAGES, BACKBONE_CONFIGS,
)
from wood_spatial.core.cache import load_cache, load_spatial_cache, cache_exists
from wood_spatial.analysis.statistical_tests import (
    knn_accuracy_gallery_query,
    knn_accuracy_cv,
    safe_pearsonr,
)
from wood_spatial.spatial.cluster_pipeline import (
    cluster_spatial_features, correct_illumination,
)
from wood_spatial.spatial.metrics import (
    compute_csi, compute_sgi,
)

logger = logging.getLogger(__name__)


def run_exp5(
    backbones: list = None,
    save: bool = True,
) -> dict:
    """
    Run Experiment 5: Cross-Magnification Stability.

    Returns dict of DataFrames: 'crossmag_acc', 'crossmag_csi', 'sgi_by_mag'.
    """
    if backbones is None:
        backbones = BB_ORDER

    acc_rows = []
    csi_rows = []
    sgi_rows = []

    # ── Step 1: Within-magnification accuracy ─────────────────────────────
    logger.info('=== Exp 5 Step 1: Within-magnification accuracy ===')

    for mag in TIER_C:
        ds_info = ALL_DATASETS[mag]
        for bb in backbones:
            try:
                feat, labels, _ = load_cache(bb, mag, 'original')
            except FileNotFoundError:
                logger.debug('Missing: %s/%s', bb, mag)
                continue
            acc, std = knn_accuracy_cv(feat, labels)
            acc_rows.append({
                'train_mag': mag, 'test_mag': mag, 'backbone': bb,
                'accuracy': acc, 'accuracy_std': std,
                'type': 'within',
            })

    # ── Step 2: Cross-magnification accuracy ──────────────────────────────
    logger.info('=== Exp 5 Step 2: Cross-magnification accuracy ===')

    # Train on x20, test on x10 and x50
    train_mag = 'VN26_x20'
    test_mags = ['VN26_x10', 'VN26_x50']

    for bb in backbones:
        try:
            feat_train, labels_train = load_cache(bb, train_mag, 'original')[:2]
        except FileNotFoundError:
            continue

        for test_mag in test_mags:
            try:
                feat_test, labels_test = load_cache(bb, test_mag, 'original')[:2]
            except FileNotFoundError:
                continue

            acc = knn_accuracy_gallery_query(feat_train, labels_train,
                                             feat_test, labels_test)
            acc_rows.append({
                'train_mag': train_mag, 'test_mag': test_mag, 'backbone': bb,
                'accuracy': acc, 'accuracy_std': 0.0,
                'type': 'cross',
            })
            logger.info('  %s: train=%s, test=%s -> %.3f', bb, train_mag, test_mag, acc)

    df_acc = pd.DataFrame(acc_rows)

    # ── Step 3: Spatial clustering at each magnification ──────────────────
    logger.info('=== Exp 5 Step 3: Spatial clustering per magnification ===')

    mag_clusters = {}  # {(mag, bb): [result_dicts]}

    for mag in TIER_C:
        for bb in backbones:
            try:
                feats, labels, paths = load_spatial_cache(bb, mag, 'original')
            except FileNotFoundError:
                continue

            img_size = BACKBONE_CONFIGS[bb]['img_size']
            n = min(len(feats), SPATIAL_MAX_IMAGES)
            feats = list(feats[:n])
            paths = paths[:n]

            # Compute guides
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

            clusters = []
            for feat, guide in zip(feats, guides):
                result = cluster_spatial_features(feat, guide, img_size, img_size)
                clusters.append(result)

            mag_clusters[(mag, bb)] = clusters

            # SGI per image
            for i, result in enumerate(clusters):
                sgi = compute_sgi(result['labels'])
                sgi_rows.append({
                    'magnification': mag, 'backbone': bb,
                    'image_idx': i, 'sgi': sgi,
                })

    df_sgi = pd.DataFrame(sgi_rows)

    # ── Step 4: CSI between magnifications ────────────────────────────────
    logger.info('=== Exp 5 Step 4: Cross-magnification CSI ===')

    # Compare x10 vs x20, x20 vs x50 (same species, different zoom)
    # Note: images are ordered by species, so index i in x10 and x20
    # likely correspond to the same species (not same physical specimen)
    mag_pairs = [('VN26_x10', 'VN26_x20'), ('VN26_x20', 'VN26_x50')]

    for mag_a, mag_b in mag_pairs:
        for bb in backbones:
            key_a = (mag_a, bb)
            key_b = (mag_b, bb)
            if key_a not in mag_clusters or key_b not in mag_clusters:
                continue

            n = min(len(mag_clusters[key_a]), len(mag_clusters[key_b]))
            for i in range(n):
                labels_a = mag_clusters[key_a][i]['labels']
                labels_b = mag_clusters[key_b][i]['labels']
                csi = compute_csi(labels_a, labels_b)
                csi_rows.append({
                    'mag_a': mag_a, 'mag_b': mag_b, 'backbone': bb,
                    'image_idx': i, 'csi': csi,
                })

    df_csi = pd.DataFrame(csi_rows)

    # ── Compile ───────────────────────────────────────────────────────────
    results = {
        'crossmag_acc': df_acc,
        'crossmag_csi': df_csi,
        'sgi_by_mag': df_sgi,
    }

    if save:
        df_acc.to_csv(V4_CSV / 'exp5_crossmag_accuracy.csv', index=False)
        if len(df_csi) > 0:
            df_csi.to_csv(V4_CSV / 'exp5_crossmag_csi.csv', index=False)
        if len(df_sgi) > 0:
            df_sgi.to_csv(V4_CSV / 'exp5_sgi_by_mag.csv', index=False)
        logger.info('Saved Exp 5 results to %s', V4_CSV)

    return results


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    t0 = time.time()
    results = run_exp5()
    logger.info('Exp 5 done in %.1f min', (time.time() - t0) / 60)

    # Summary
    df = results['crossmag_acc']
    print('\n=== Within-Magnification Accuracy ===')
    within = df[df['type'] == 'within']
    for _, row in within.iterrows():
        print(f"  {row['backbone']:20s} | {row['train_mag']:10s} | {row['accuracy']:.3f}")

    print('\n=== Cross-Magnification Accuracy (train x20) ===')
    cross = df[df['type'] == 'cross']
    for _, row in cross.iterrows():
        print(f"  {row['backbone']:20s} | test={row['test_mag']:10s} | {row['accuracy']:.3f}")

    df_sgi = results['sgi_by_mag']
    if len(df_sgi) > 0:
        print('\n=== Mean SGI by Magnification ===')
        for mag in TIER_C:
            sub = df_sgi[df_sgi['magnification'] == mag]
            if len(sub) > 0:
                for bb in BB_ORDER:
                    bb_sub = sub[sub['backbone'] == bb]
                    if len(bb_sub) > 0:
                        print(f'  {bb:20s} | {mag:10s} | SGI={bb_sub["sgi"].mean():.6f}')


if __name__ == '__main__':
    main()
