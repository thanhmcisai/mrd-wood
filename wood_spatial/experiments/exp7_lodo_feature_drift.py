#!/usr/bin/env python3
"""
Experiment 7b: leave-one-dataset-out feature-drift validation.

This checks whether the feature-drift/accuracy-drop relationship remains useful
when an entire Tier-A dataset is held out. It is a lightweight CSV-only
analysis for reviewer concerns about grouped dependence and dataset transfer.
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd
from scipy import stats

from wood_spatial.config import V4_CSV

logger = logging.getLogger(__name__)


def _fit_ols(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1])


def _rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def run_lodo_feature(input_csv=None, save=True):
    if input_csv is None:
        input_csv = V4_CSV / 'exp1b_feature_geometry.csv'
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=['dataset', 'backbone', 'perturbation', 'feature_drift', 'drop'])

    rows = []
    datasets = sorted(df['dataset'].unique())
    for heldout in datasets:
        train = df[df['dataset'] != heldout]
        test = df[df['dataset'] == heldout]
        if len(train) < 20 or len(test) < 20:
            continue

        intercept, slope = _fit_ols(train['feature_drift'].values, train['drop'].values)
        pred = intercept + slope * test['feature_drift'].values
        r, p = stats.pearsonr(test['feature_drift'], test['drop'])
        rho, p_rho = stats.spearmanr(test['feature_drift'], test['drop'])
        rows.append({
            'heldout_dataset': heldout,
            'train_n': int(len(train)),
            'test_n': int(len(test)),
            'train_datasets': ','.join(sorted(train['dataset'].unique())),
            'train_slope': slope,
            'train_intercept': intercept,
            'test_pearson_r': float(r),
            'test_pearson_p': float(p),
            'test_spearman_rho': float(rho),
            'test_spearman_p': float(p_rho),
            'test_rmse': _rmse(test['drop'].values, pred),
            'test_mae': float(np.mean(np.abs(test['drop'].values - pred))),
            'test_drop_mean': float(test['drop'].mean()),
            'test_drift_mean': float(test['feature_drift'].mean()),
        })
        logger.info('Heldout %s: r=%.4f rho=%.4f rmse=%.4f',
                    heldout, r, rho, rows[-1]['test_rmse'])

    out = pd.DataFrame(rows)
    summary = pd.DataFrame()
    if len(out):
        summary = pd.DataFrame([{
            'n_heldout_datasets': int(len(out)),
            'mean_test_pearson_r': float(out['test_pearson_r'].mean()),
            'min_test_pearson_r': float(out['test_pearson_r'].min()),
            'mean_test_spearman_rho': float(out['test_spearman_rho'].mean()),
            'min_test_spearman_rho': float(out['test_spearman_rho'].min()),
            'mean_test_rmse': float(out['test_rmse'].mean()),
        }])

    if save:
        out.to_csv(V4_CSV / 'exp7_lodo_feature_drift.csv', index=False)
        summary.to_csv(V4_CSV / 'exp7_lodo_feature_drift_summary.csv', index=False)
        logger.info('Saved LODO feature-drift validation to %s', V4_CSV)
    return {'lodo': out, 'summary': summary}


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Run leave-one-dataset-out feature-drift validation.')
    parser.add_argument('--input-csv', default=None)
    args = parser.parse_args()
    t0 = time.time()
    res = run_lodo_feature(args.input_csv)
    print('\n=== LODO Feature-Drift Summary ===')
    if len(res['summary']):
        print(res['summary'].to_string(index=False))
    print('\nDetails:')
    if len(res['lodo']):
        print(res['lodo'].to_string(index=False))
    print(f'\nDone in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
