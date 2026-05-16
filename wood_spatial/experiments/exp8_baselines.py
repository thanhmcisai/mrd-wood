#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp8 additional failure-detection baselines.

Computes paired feature drift, Mahalanobis distance delta, and reference-bank
kNN distance from cached features. Supports dataset/backbone checkpoints so the
Colab runner can deep-parallelize the expensive cache reads.
"""
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from wood_spatial.config import BB_ORDER, PERTURB_CONFIGS, TIER_A, V4_CSV, V4_FEAT_CACHE
from wood_spatial.core.perturbations import cache_tag_for

logger = logging.getLogger(__name__)

PERTURB_TAGS = [
    cache_tag_for(pert_name, value)
    for pert_name, pcfg in PERTURB_CONFIGS.items()
    for value in pcfg['values']
]


def _safe_part_name(*parts: str) -> str:
    raw = '__'.join(str(p) for p in parts)
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in raw)


def _checkpoint_dir():
    path = V4_CSV / 'exp8_baselines_checkpoints'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_path(dataset: str, backbone: str):
    return _checkpoint_dir() / f'{_safe_part_name(dataset, backbone)}.csv'


def _load_feat(bb: str, ds: str, tag: str):
    path = V4_FEAT_CACHE / f'{bb}_{ds}_{tag}.npz'
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=True)
    feats = data['features'].astype(np.float32)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return feats, data['labels']


def _checkpoint_complete(dataset: str, backbone: str) -> bool:
    path = _checkpoint_path(dataset, backbone)
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, usecols=['perturbation'])
    except Exception:
        return False
    return set(PERTURB_TAGS).issubset(set(df['perturbation'].astype(str)))


def _run_pair(dataset: str, backbone: str) -> pd.DataFrame:
    clean, labels = _load_feat(backbone, dataset, 'original')
    if clean is None:
        logger.warning('Missing clean cache: %s/%s', backbone, dataset)
        return pd.DataFrame()

    try:
        cov = LedoitWolf(assume_centered=False).fit(clean)
        mah_clean = float(np.mean(cov.mahalanobis(clean)))
        has_mahal = True
    except Exception as exc:
        logger.warning('Mahalanobis unavailable for %s/%s: %s', backbone, dataset, exc)
        cov = None
        mah_clean = 0.0
        has_mahal = False

    le = LabelEncoder()
    y_clean = le.fit_transform(labels)
    knn = KNeighborsClassifier(5, metric='cosine')
    knn.fit(clean, y_clean)
    acc_clean = float(knn.score(clean, y_clean))

    rows = []
    for tag in PERTURB_TAGS:
        shifted, shifted_labels = _load_feat(backbone, dataset, tag)
        if shifted is None:
            continue

        n = min(len(clean), len(shifted))
        cos = np.clip(np.sum(clean[:n] * shifted[:n], axis=1), -1, 1)
        drift = float(np.mean(1 - cos))

        try:
            y_shifted = le.transform(shifted_labels)
        except ValueError:
            y_shifted = y_clean[:len(shifted)]
        acc_pert = float(knn.score(shifted, y_shifted))
        drop = acc_clean - acc_pert

        if has_mahal:
            try:
                mah_delta = float(np.mean(cov.mahalanobis(shifted))) - mah_clean
            except Exception:
                mah_delta = np.nan
        else:
            mah_delta = np.nan

        dists, _ = knn.kneighbors(shifted, n_neighbors=1)
        knn_d = float(np.mean(dists))

        rows.append({
            'backbone': backbone,
            'dataset': dataset,
            'perturbation': tag,
            'feature_drift': drift,
            'mahal_delta': mah_delta,
            'knn_dist': knn_d,
            'reference_bank_drift': knn_d,
            'accuracy_drop': drop,
        })

    return pd.DataFrame(rows)


def run_checkpoint(dataset: str, backbone: str, force: bool = False) -> pd.DataFrame:
    path = _checkpoint_path(dataset, backbone)
    if _checkpoint_complete(dataset, backbone) and not force:
        logger.info('%s/%s checkpoint hit', backbone, dataset)
        return pd.read_csv(path)
    df = _run_pair(dataset, backbone)
    df.to_csv(path, index=False)
    logger.info('%s/%s checkpoint saved: %d rows', backbone, dataset, len(df))
    return df


def _read_checkpoints(datasets: list, backbones: list) -> pd.DataFrame:
    frames = []
    for ds in datasets:
        for bb in backbones:
            path = _checkpoint_path(ds, bb)
            if path.exists():
                frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def finalize_from_checkpoints(datasets: list = None, backbones: list = None, save: bool = True):
    datasets = datasets or TIER_A
    backbones = backbones or BB_ORDER
    df = _read_checkpoints(datasets, backbones)
    if df.empty:
        raise RuntimeError(f'No Exp8 baseline checkpoints found in {_checkpoint_dir()}')
    return _finalize(df, save=save)


def _finalize(df: pd.DataFrame, save: bool = True):
    if save:
        df.to_csv(V4_CSV / 'exp8_additional_baselines.csv', index=False)

    y_true = (df['accuracy_drop'] > 0.20).astype(int)
    results = []
    for name, col in [
        ('Feature drift (cosine)', 'feature_drift'),
        ('Mahalanobis distance delta', 'mahal_delta'),
        ('Reference-bank kNN drift', 'reference_bank_drift'),
    ]:
        scores = pd.to_numeric(df[col], errors='coerce')
        scores = scores.fillna(scores.median())
        try:
            auc = roc_auc_score(y_true, scores)
            ap = average_precision_score(y_true, scores)
            corr = float(np.corrcoef(scores, df['accuracy_drop'])[0, 1])
            results.append({
                'detector': name,
                'auc_roc': float(auc),
                'avg_precision': float(ap),
                'r_vs_drop': corr,
                'n': int(len(df)),
            })
            logger.info('%s: AUC=%.4f AP=%.4f r=%.4f', name, auc, ap, corr)
        except Exception as exc:
            logger.warning('%s failed: %s', name, exc)

    auc_df = pd.DataFrame(results)
    if save:
        auc_df.to_csv(V4_CSV / 'exp8_new_baselines_auc.csv', index=False)
    return {'baselines': df, 'auc': auc_df}


def run(datasets: list = None, backbones: list = None, save: bool = True):
    datasets = datasets or TIER_A
    backbones = backbones or BB_ORDER
    frames = []
    for ds in datasets:
        for bb in backbones:
            frames.append(run_checkpoint(ds, bb))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _finalize(df, save=save)


def main():
    parser = argparse.ArgumentParser(description='Run Exp8 additional baselines.')
    parser.add_argument('--datasets', nargs='*', default=None)
    parser.add_argument('--backbones', nargs='*', default=None)
    parser.add_argument('--checkpoint-only', action='store_true')
    parser.add_argument('--finalize-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

    datasets = args.datasets or TIER_A
    backbones = args.backbones or BB_ORDER
    if args.finalize_only:
        out = finalize_from_checkpoints(datasets, backbones)
    elif args.checkpoint_only:
        for ds in datasets:
            for bb in backbones:
                run_checkpoint(ds, bb, force=args.force)
        out = {'baselines': _read_checkpoints(datasets, backbones), 'auc': pd.DataFrame()}
    else:
        out = run(datasets, backbones)

    print(f"Rows: {len(out['baselines'])}")
    if len(out['auc']):
        print(out['auc'].to_string(index=False))


if __name__ == '__main__':
    main()
