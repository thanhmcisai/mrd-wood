#!/usr/bin/env python3
"""
Experiment 10b: leave-one-perturbation-family-out monitor evaluation.

Uses Exp10 reference-monitor scores. For each held-out perturbation family,
thresholds are selected on all other families and evaluated on the held-out
family. This tests whether a monitor threshold transfers to unseen shift types.
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from wood_spatial.config import V4_CSV

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


def perturbation_family(tag: str) -> str:
    tag = str(tag)
    if tag.startswith('blur_'):
        return 'gaussian_blur'
    if tag.startswith('defocus_'):
        return 'defocus_blur'
    if tag.startswith('resize_'):
        return 'resize'
    if tag.startswith('brightness_'):
        return 'illumination'
    if tag.startswith('jpeg_'):
        return 'jpeg'
    if tag.startswith('color_shift_'):
        return 'red_channel_shift'
    if tag.startswith('green_channel_shift_'):
        return 'green_channel_shift'
    if tag.startswith('blue_channel_shift_'):
        return 'blue_channel_shift'
    if tag.startswith('gaussian_noise_'):
        return 'gaussian_noise'
    if tag.startswith('shot_noise_'):
        return 'shot_noise'
    if tag.startswith('impulse_noise_'):
        return 'impulse_noise'
    if tag.startswith('motion_blur_'):
        return 'motion_blur'
    if tag.startswith('zoom_blur_'):
        return 'zoom_blur'
    if tag.startswith('contrast_'):
        return 'contrast'
    if tag.startswith('pixelate_'):
        return 'pixelate'
    if tag.startswith('scratch_'):
        return 'scratch'
    if tag.startswith('rotation_'):
        return 'rotation'
    if tag.startswith('compound_optical_'):
        return 'compound_optical'
    if tag.startswith('compound_digital_'):
        return 'compound_digital'
    if tag.startswith('compound_field_'):
        return 'compound_field'
    if tag.startswith('compound_'):
        return 'compound'
    return tag.split('_')[0]


def best_f1_threshold(y_true, scores):
    prec, rec, th = precision_recall_curve(y_true, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    idx = int(np.argmax(f1))
    if idx >= len(th):
        return float(np.max(scores) + 1e-12), float(f1[idx])
    return float(th[idx]), float(f1[idx])


def eval_at_threshold(y_true, scores, threshold):
    pred = scores >= threshold
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    return precision, recall, f1, fpr, tp, fp, fn, tn


def run_exp10_lopo(input_csv=None, save=True):
    if input_csv is None:
        input_csv = V4_CSV / 'exp10_reference_monitor_scores.csv'
    df = pd.read_csv(input_csv)
    if 'family' not in df.columns:
        df['family'] = df['perturbation'].map(perturbation_family)

    rows = []
    families = sorted(df['family'].dropna().unique())
    for det in DETECTORS:
        if det not in df.columns:
            continue
        values = pd.to_numeric(df[det], errors='coerce')
        valid = values.notna() & df['failure'].notna()
        work = df.loc[valid].copy()
        work['_score'] = values.loc[valid].values
        work['_failure'] = work['failure'].astype(int)

        for heldout in families:
            train = work[work['family'] != heldout]
            test = work[work['family'] == heldout]
            if len(train) < 20 or len(test) < 10:
                continue
            if train['_failure'].nunique() < 2:
                continue

            threshold, train_f1 = best_f1_threshold(
                train['_failure'].values, train['_score'].values)

            y = test['_failure'].values
            s = test['_score'].values
            auc = np.nan
            ap = np.nan
            if len(np.unique(y)) == 2:
                auc = float(roc_auc_score(y, s))
                ap = float(average_precision_score(y, s))
            precision, recall, f1, fpr, tp, fp, fn, tn = eval_at_threshold(y, s, threshold)

            rows.append({
                'detector': det,
                'heldout_family': heldout,
                'train_n': int(len(train)),
                'test_n': int(len(test)),
                'test_failure_rate': float(np.mean(y)),
                'threshold_from_train': threshold,
                'train_best_f1': train_f1,
                'test_auc_roc': auc,
                'test_avg_precision': ap,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'test_fpr': fpr,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            })
            logger.info('%s heldout=%s AUC=%s F1=%.4f',
                        det, heldout, f'{auc:.4f}' if not np.isnan(auc) else 'nan', f1)

    out = pd.DataFrame(rows)
    summary = pd.DataFrame()
    if len(out):
        summary = out.groupby('detector').agg(
            mean_test_auc=('test_auc_roc', 'mean'),
            mean_test_ap=('test_avg_precision', 'mean'),
            mean_test_f1=('test_f1', 'mean'),
            mean_test_recall=('test_recall', 'mean'),
            mean_test_fpr=('test_fpr', 'mean'),
            n_families=('heldout_family', 'nunique'),
        ).reset_index().sort_values('mean_test_auc', ascending=False)

    if save:
        out.to_csv(V4_CSV / 'exp10_lopo_monitor.csv', index=False)
        summary.to_csv(V4_CSV / 'exp10_lopo_monitor_summary.csv', index=False)
        logger.info('Saved LOPO monitor results to %s', V4_CSV)
    return {'lopo': out, 'summary': summary}


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Run leave-one-perturbation-family-out monitor evaluation.')
    parser.add_argument('--input-csv', default=None)
    args = parser.parse_args()

    t0 = time.time()
    res = run_exp10_lopo(args.input_csv)
    print('\n=== LOPO Monitor Summary ===')
    if len(res['summary']):
        print(res['summary'].to_string(index=False))
    print(f'\nRows: {len(res["lopo"])}')
    print(f'Done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
