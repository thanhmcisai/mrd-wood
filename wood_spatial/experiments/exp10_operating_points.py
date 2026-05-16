#!/usr/bin/env python3
"""
Experiment: fixed operating points for reference-bank monitors.

Computes TPR at fixed FPR and FPR at fixed TPR from Exp10 monitor scores,
including leave-one-perturbation-family-out threshold transfer.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from wood_spatial.config import V4_CSV
from wood_spatial.experiments.exp10_lopo_monitor import perturbation_family

logger = logging.getLogger(__name__)

DETECTORS = [
    'ref_mmd_rbf',
    'ref_centroid_cosine',
    'ref_knn1_dist',
    'ref_knn5_dist',
    'ref_mahalanobis_delta',
    'ref_mahalanobis',
    'paired_feature_drift_oracle',
]


def _roc_arrays(y, score):
    mask = np.isfinite(score)
    y = np.asarray(y)[mask].astype(int)
    score = np.asarray(score)[mask].astype(float)
    if len(np.unique(y)) < 2:
        return None
    return roc_curve(y, score)


def _tpr_at_fpr(y, score, target_fpr: float):
    roc = _roc_arrays(y, score)
    if roc is None:
        return np.nan, np.nan
    fpr, tpr, thr = roc
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        idx = int(np.argmin(fpr))
    else:
        idx = ok[np.argmax(tpr[ok])]
    return float(tpr[idx]), float(thr[idx])


def _fpr_at_tpr(y, score, target_tpr: float):
    roc = _roc_arrays(y, score)
    if roc is None:
        return np.nan, np.nan
    fpr, tpr, thr = roc
    ok = np.where(tpr >= target_tpr)[0]
    if len(ok) == 0:
        return np.nan, np.nan
    idx = ok[np.argmin(fpr[ok])]
    return float(fpr[idx]), float(thr[idx])


def _threshold_for_train_fpr(y_train, s_train, target_fpr: float):
    y_train = np.asarray(y_train).astype(int)
    s_train = np.asarray(s_train).astype(float)
    mask = np.isfinite(s_train)
    y_train, s_train = y_train[mask], s_train[mask]
    neg = s_train[y_train == 0]
    if len(neg) == 0:
        return np.nan
    # Alarm when score >= threshold. Choose the highest-sensitivity threshold
    # whose empirical false-positive rate on training negatives is <= target.
    return float(np.quantile(neg, 1.0 - target_fpr, method='higher'))


def _eval_threshold(y, s, threshold):
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(y) == 0 or not np.isfinite(threshold):
        return {'tpr': np.nan, 'fpr': np.nan, 'precision': np.nan, 'n': int(len(y))}
    pred = s >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        'tpr': tp / (tp + fn) if (tp + fn) else np.nan,
        'fpr': fp / (fp + tn) if (fp + tn) else np.nan,
        'precision': tp / (tp + fp) if (tp + fp) else np.nan,
        'n': int(len(y)),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def run(save: bool = True) -> dict:
    scores = pd.read_csv(V4_CSV / 'exp10_reference_monitor_scores.csv')
    scores['family'] = scores['perturbation'].map(perturbation_family)

    pooled_rows = []
    for det in DETECTORS:
        if det not in scores:
            continue
        for target in [0.05, 0.10]:
            tpr, thr = _tpr_at_fpr(scores['failure'], scores[det], target)
            pooled_rows.append({
                'detector': det,
                'metric': f'tpr_at_fpr_{target:.2f}',
                'target': target,
                'value': tpr,
                'threshold': thr,
                'n': int(scores[det].notna().sum()),
            })
        for target in [0.90, 0.95]:
            fpr, thr = _fpr_at_tpr(scores['failure'], scores[det], target)
            pooled_rows.append({
                'detector': det,
                'metric': f'fpr_at_tpr_{target:.2f}',
                'target': target,
                'value': fpr,
                'threshold': thr,
                'n': int(scores[det].notna().sum()),
            })

    lopo_rows = []
    for det in DETECTORS:
        if det not in scores:
            continue
        for family in sorted(scores['family'].dropna().unique()):
            train = scores[scores['family'] != family]
            test = scores[scores['family'] == family]
            for target in [0.05, 0.10]:
                thr = _threshold_for_train_fpr(train['failure'], train[det], target)
                metrics = _eval_threshold(test['failure'], test[det], thr)
                lopo_rows.append({
                    'detector': det,
                    'heldout_family': family,
                    'target_train_fpr': target,
                    'threshold': thr,
                    'test_failure_rate': float(test['failure'].mean()),
                    **metrics,
                })

    pooled = pd.DataFrame(pooled_rows)
    lopo = pd.DataFrame(lopo_rows)

    if save:
        pooled.to_csv(V4_CSV / 'exp10_operating_points_fixed_fpr.csv', index=False)
        lopo.to_csv(V4_CSV / 'exp10_lopo_operating_points.csv', index=False)

    return {'pooled': pooled, 'lopo': lopo}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    out = run(save=True)
    print(out['pooled'].head(20).to_string(index=False))
    print(out['lopo'].head(20).to_string(index=False))
