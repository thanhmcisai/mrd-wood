#!/usr/bin/env python3
"""
Experiment 10d: leave-one-dataset-out monitor evaluation.

Uses Exp10 reference-monitor scores. For each held-out dataset, thresholds are
selected on all other datasets and evaluated on the held-out dataset. This is
stricter than pooled Exp10 because it tests dataset-level threshold transfer.
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from wood_spatial.config import V4_CSV
from wood_spatial.experiments.exp10_lopo_monitor import DETECTORS, best_f1_threshold, eval_at_threshold

logger = logging.getLogger(__name__)


def run_exp10_lodo(input_csv=None, save=True):
    if input_csv is None:
        input_csv = V4_CSV / 'exp10_reference_monitor_scores.csv'
    df = pd.read_csv(input_csv)

    rows = []
    datasets = sorted(df['dataset'].dropna().unique())
    for det in DETECTORS:
        if det not in df.columns:
            continue
        scores = pd.to_numeric(df[det], errors='coerce')
        valid = scores.notna() & df['failure'].notna()
        work = df.loc[valid].copy()
        work['_score'] = scores.loc[valid].values
        work['_failure'] = work['failure'].astype(int)

        for heldout in datasets:
            train = work[work['dataset'] != heldout]
            test = work[work['dataset'] == heldout]
            if len(train) < 20 or len(test) < 10 or train['_failure'].nunique() < 2:
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
                'heldout_dataset': heldout,
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
            min_test_auc=('test_auc_roc', 'min'),
            mean_test_ap=('test_avg_precision', 'mean'),
            mean_test_f1=('test_f1', 'mean'),
            min_test_f1=('test_f1', 'min'),
            mean_test_recall=('test_recall', 'mean'),
            mean_test_fpr=('test_fpr', 'mean'),
            n_datasets=('heldout_dataset', 'nunique'),
        ).reset_index().sort_values('mean_test_auc', ascending=False)

    if save:
        out.to_csv(V4_CSV / 'exp10_lodo_monitor.csv', index=False)
        summary.to_csv(V4_CSV / 'exp10_lodo_monitor_summary.csv', index=False)
        logger.info('Saved LODO monitor results to %s', V4_CSV)
    return {'lodo': out, 'summary': summary}


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Run leave-one-dataset-out monitor evaluation.')
    parser.add_argument('--input-csv', default=None)
    args = parser.parse_args()
    t0 = time.time()
    res = run_exp10_lodo(args.input_csv)
    print('\n=== LODO Monitor Summary ===')
    if len(res['summary']):
        print(res['summary'].to_string(index=False))
    print(f'\nRows: {len(res["lodo"])}')
    print(f'Done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
