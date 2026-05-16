"""
Exp5b: Cross-Magnification Spatial Clustering
===============================================
Computes SGI per magnification and CSI between magnification levels
(same species, different zoom) to answer:
"Are spatial anatomical representations stable across magnification?"

Usage:
    python -m wood_spatial.experiments.exp5b_crossmag_spatial
"""
import logging
import time
import warnings

import numpy as np
import pandas as pd

from wood_spatial.config import BB_ORDER, V4_CSV, V4_SPATIAL_CACHE, N_CLUSTERS
from wood_spatial.core.cache import load_spatial_cache
from wood_spatial.spatial.metrics import compute_sgi, compute_csi_metrics

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

MAGS = ['VN26_x10', 'VN26_x20', 'VN26_x50']
MAG_PAIRS = [('VN26_x10', 'VN26_x20'), ('VN26_x20', 'VN26_x50'), ('VN26_x10', 'VN26_x50')]


def _load_spatial(bb: str, ds: str):
    try:
        return load_spatial_cache(bb, ds, 'original')
    except FileNotFoundError:
        return None, None, None


def _cluster_and_label(feat: np.ndarray, img_size: int = 224):
    from wood_spatial.spatial.cluster_pipeline import cluster_spatial_features, correct_illumination
    import cv2
    guide = np.ones((img_size, img_size), dtype=np.float32) * 0.5
    result = cluster_spatial_features(feat, guide, img_size, img_size)
    return result['labels']


def run_exp5b(max_images: int = 100, save: bool = True) -> dict:
    sgi_rows = []
    csi_rows = []

    for bb in BB_ORDER:
        logger.info('=== Exp5b: %s ===', bb)

        # Load and cluster all magnifications
        mag_labels = {}
        for mag in MAGS:
            feats, labels, paths = _load_spatial(bb, mag)
            if feats is None:
                logger.warning('Missing spatial: %s/%s', bb, mag)
                continue
            n = min(len(feats), max_images)
            img_labels = []
            sgi_vals = []
            for i in range(n):
                lbl = _cluster_and_label(feats[i])
                img_labels.append(lbl)
                sgi_vals.append(compute_sgi(lbl, N_CLUSTERS))
                sgi_rows.append({
                    'backbone': bb, 'magnification': mag, 'image_idx': i,
                    'sgi': sgi_vals[-1],
                    'species_label': int(labels[i]) if labels is not None else -1,
                })
            mag_labels[mag] = img_labels
            logger.info('  %s/%s: n=%d, mean SGI=%.6f', bb, mag, n, np.mean(sgi_vals))

        # Cross-mag CSI: same image index across magnification pairs
        for mag_a, mag_b in MAG_PAIRS:
            if mag_a not in mag_labels or mag_b not in mag_labels:
                continue
            n = min(len(mag_labels[mag_a]), len(mag_labels[mag_b]))
            csi_vals = []
            for i in range(n):
                metrics = compute_csi_metrics(
                    mag_labels[mag_a][i], mag_labels[mag_b][i], N_CLUSTERS)
                csi_vals.append(metrics['csi'])
                csi_rows.append({
                    'backbone': bb, 'mag_a': mag_a, 'mag_b': mag_b,
                    'image_idx': i,
                    'csi': metrics['csi'],
                    'csi_bo': metrics['csi_bo'],
                    'csi_hungarian': metrics['csi_hungarian'],
                    'csi_permutation_gap': metrics['csi_permutation_gap'],
                })
            logger.info('  CSI %s vs %s: %.4f', mag_a, mag_b, np.mean(csi_vals))

    df_sgi = pd.DataFrame(sgi_rows)
    df_csi = pd.DataFrame(csi_rows)

    results = {'sgi': df_sgi, 'csi': df_csi}

    if save:
        df_sgi.to_csv(V4_CSV / 'exp5b_sgi_by_mag.csv', index=False)
        df_csi.to_csv(V4_CSV / 'exp5b_crossmag_csi.csv', index=False)
        logger.info('Saved Exp5b to %s', V4_CSV)

    return results


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    t0 = time.time()
    results = run_exp5b()
    logger.info('Exp5b done in %.1f min', (time.time() - t0) / 60)

    df_sgi = results['sgi']
    print('\n=== Mean SGI by backbone × magnification ===')
    print(df_sgi.groupby(['backbone', 'magnification']).sgi.mean().unstack('magnification').round(6).to_string())

    df_csi = results['csi']
    print('\n=== Mean Cross-mag CSI by backbone ===')
    print(df_csi.groupby(['backbone', 'mag_a', 'mag_b']).csi.mean().unstack(['mag_a', 'mag_b']).round(4).to_string())


if __name__ == '__main__':
    main()
