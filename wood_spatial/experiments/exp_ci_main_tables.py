#!/usr/bin/env python3
"""
Experiment: bootstrap confidence intervals for main manuscript tables.

The bootstrap operates over existing experiment records, not raw images. It is
intended to provide uncertainty for aggregate manuscript tables.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from wood_spatial.config import V4_CSV

logger = logging.getLogger(__name__)

RNG_SEED = 42
B = 1000


def _ci(vals, alpha=0.05):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return (
        float(np.quantile(vals, alpha / 2)),
        float(np.quantile(vals, 1 - alpha / 2)),
    )


def _boot_mean(df, group_cols, value_cols, b=B):
    rng = np.random.RandomState(RNG_SEED)
    rows = []
    for key, sub in df.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        n = len(sub)
        idx = np.arange(n)
        for col in value_cols:
            point = float(sub[col].mean())
            boots = []
            arr = sub[col].astype(float).values
            for _ in range(b):
                sample = arr[rng.choice(idx, n, replace=True)]
                boots.append(np.nanmean(sample))
            lo, hi = _ci(boots)
            rows.append({
                **dict(zip(group_cols, key)),
                'metric': col,
                'point': point,
                'ci_lo': lo,
                'ci_hi': hi,
                'n': n,
                'bootstrap_B': b,
            })
    return pd.DataFrame(rows)


def _best_f1(y, score):
    precision, recall, _ = precision_recall_curve(y, score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(np.nanmax(f1))


def _boot_monitor(scores, detectors, b=B):
    rng = np.random.RandomState(RNG_SEED)
    rows = []
    n = len(scores)
    idx = np.arange(n)
    for det in detectors:
        if det not in scores:
            continue
        sub = scores[['failure', det]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 30 or sub['failure'].nunique() < 2:
            continue
        y = sub['failure'].astype(int).values
        s = sub[det].astype(float).values
        point_auc = float(roc_auc_score(y, s))
        point_ap = float(average_precision_score(y, s))
        point_f1 = _best_f1(y, s)
        aucs, aps, f1s = [], [], []
        local_idx = np.arange(len(sub))
        for _ in range(b):
            take = rng.choice(local_idx, len(sub), replace=True)
            yy = y[take]
            ss = s[take]
            if len(np.unique(yy)) < 2:
                continue
            aucs.append(roc_auc_score(yy, ss))
            aps.append(average_precision_score(yy, ss))
            f1s.append(_best_f1(yy, ss))
        auc_lo, auc_hi = _ci(aucs)
        ap_lo, ap_hi = _ci(aps)
        f1_lo, f1_hi = _ci(f1s)
        rows.extend([
            {'detector': det, 'metric': 'roc_auc', 'point': point_auc, 'ci_lo': auc_lo, 'ci_hi': auc_hi, 'n': len(sub), 'bootstrap_B': b},
            {'detector': det, 'metric': 'avg_precision', 'point': point_ap, 'ci_lo': ap_lo, 'ci_hi': ap_hi, 'n': len(sub), 'bootstrap_B': b},
            {'detector': det, 'metric': 'best_f1', 'point': point_f1, 'ci_lo': f1_lo, 'ci_hi': f1_hi, 'n': len(sub), 'bootstrap_B': b},
        ])
    return pd.DataFrame(rows)


def run(save: bool = True) -> dict:
    acc = pd.read_csv(V4_CSV / 'exp1_accuracy_matrix.csv')
    pert = acc[acc['perturbation'] != 'clean'].copy()

    backbone_base = (
        pert.groupby('backbone', as_index=False)
        .agg(clean_acc=('acc_clean', 'mean'), pert_acc=('accuracy', 'mean'), drop=('drop', 'mean'))
    )
    backbone_ci = _boot_mean(pert, ['backbone'], ['acc_clean', 'accuracy', 'drop'])
    backbone_ci = backbone_ci.merge(backbone_base, on='backbone', how='left')

    perturbation_base = (
        pert.groupby('perturbation', as_index=False)
        .agg(accuracy=('accuracy', 'mean'), drop=('drop', 'mean'))
    )
    perturbation_ci = _boot_mean(pert, ['perturbation'], ['accuracy', 'drop'])
    perturbation_ci = perturbation_ci.merge(perturbation_base, on='perturbation', how='left')

    scores = pd.read_csv(V4_CSV / 'exp10_reference_monitor_scores.csv')
    detectors = [
        'ref_mmd_rbf',
        'ref_centroid_cosine',
        'ref_knn1_dist',
        'ref_knn5_dist',
        'ref_mahalanobis_delta',
        'ref_mahalanobis',
        'paired_feature_drift_oracle',
    ]
    monitor_ci = _boot_monitor(scores, detectors)

    if save:
        backbone_ci.to_csv(V4_CSV / 'exp_ci_backbone_robustness.csv', index=False)
        perturbation_ci.to_csv(V4_CSV / 'exp_ci_perturbation_drop.csv', index=False)
        monitor_ci.to_csv(V4_CSV / 'exp_ci_monitor_auc.csv', index=False)

    return {
        'backbone': backbone_ci,
        'perturbation': perturbation_ci,
        'monitor': monitor_ci,
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    out = run(save=True)
    print(out['backbone'].head(20).to_string(index=False))
    print(out['perturbation'].head(20).to_string(index=False))
    print(out['monitor'].head(20).to_string(index=False))
