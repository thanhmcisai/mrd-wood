#!/usr/bin/env python3
"""
Experiment 5c: full VN26 cross-magnification transfer matrix.

This is a lightweight follow-up to Exp5. It uses the same frozen-feature
cosine kNN gallery-query protocol, but evaluates all 3x3 train/test
magnification pairs instead of only x20 as the cross-magnification source.
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd

from wood_spatial.analysis.statistical_tests import knn_accuracy_cv, knn_accuracy_gallery_query
from wood_spatial.config import BB_ORDER, TIER_C, V4_CSV, V4_FEAT_CACHE

logger = logging.getLogger(__name__)


def load_feature_cache(backbone: str, dataset: str, tag: str = 'original'):
    path = V4_FEAT_CACHE / f'{backbone}_{dataset}_{tag}.npz'
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return data['features'], data['labels']


def run_exp5_full_crossmag(backbones=None, mags=None, save=True):
    if backbones is None:
        backbones = BB_ORDER
    if mags is None:
        mags = TIER_C

    rows = []
    for bb in backbones:
        cache = {}
        for mag in mags:
            try:
                feat, labels = load_feature_cache(bb, mag, 'original')
            except FileNotFoundError:
                logger.warning('Missing cache: %s/%s/original', bb, mag)
                continue
            cache[mag] = (feat, labels)

        for train_mag in mags:
            if train_mag not in cache:
                continue
            feat_train, labels_train = cache[train_mag]

            for test_mag in mags:
                if test_mag not in cache:
                    continue
                feat_test, labels_test = cache[test_mag]
                if train_mag == test_mag:
                    acc, std = knn_accuracy_cv(feat_train, labels_train)
                    protocol = 'within_cv'
                else:
                    acc = knn_accuracy_gallery_query(
                        feat_train, labels_train, feat_test, labels_test)
                    std = 0.0
                    protocol = 'cross_gallery_query'

                rows.append({
                    'backbone': bb,
                    'train_mag': train_mag,
                    'test_mag': test_mag,
                    'accuracy': acc,
                    'accuracy_std': std,
                    'protocol': protocol,
                    'n_train': int(len(labels_train)),
                    'n_test': int(len(labels_test)),
                })
                logger.info('%s train=%s test=%s acc=%.4f',
                            bb, train_mag, test_mag, acc)

    df = pd.DataFrame(rows)
    summary = pd.DataFrame()
    if len(df):
        summary = df.groupby(['train_mag', 'test_mag']).agg(
            mean_accuracy=('accuracy', 'mean'),
            std_accuracy=('accuracy', 'std'),
            n_backbones=('backbone', 'nunique'),
        ).reset_index()

    if save and len(df):
        df.to_csv(V4_CSV / 'exp5_full_crossmag_accuracy.csv', index=False)
        summary.to_csv(V4_CSV / 'exp5_full_crossmag_summary.csv', index=False)
        logger.info('Saved Exp5 full cross-mag results to %s', V4_CSV)
    elif save:
        logger.warning('No full cross-magnification rows were produced; existing outputs were not overwritten.')

    return {'accuracy': df, 'summary': summary}


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Run full VN26 3x3 cross-magnification transfer.')
    parser.add_argument('--backbones', nargs='+', default=None)
    parser.add_argument('--mags', nargs='+', default=None)
    args = parser.parse_args()

    t0 = time.time()
    res = run_exp5_full_crossmag(args.backbones, args.mags)
    print('\n=== Full Cross-Magnification Summary ===')
    if len(res['summary']):
        print(res['summary'].pivot(
            index='train_mag', columns='test_mag', values='mean_accuracy'
        ).round(4).to_string())
    print(f'\nRows: {len(res["accuracy"])}')
    print(f'Done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
