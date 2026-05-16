"""
Wood Spatial — Experiment 9: Cross-Dataset Feature Geometry Validation
=======================================================================
Validates that the feature geometry → failure link (r=0.924 on Tier-A)
generalizes to external acquisition protocols
(Tier-B: BFS46, FSDM41, GOIMAI, WOODAUTH, BD11).

Usage:
    python -m wood_spatial.experiments.exp9_tierb_validation
"""
import logging
import time
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from wood_spatial.config import BB_ORDER, TIER_B, V4_CSV
from wood_spatial.core.cache import load_cache
from wood_spatial.core.perturbations import cache_tag_for
from wood_spatial.analysis.feature_geometry import feature_drift, geometry_summary
from wood_spatial.analysis.statistical_tests import knn_accuracy_gallery_query

logger = logging.getLogger(__name__)


def _safe_part_name(*parts: str) -> str:
    raw = '__'.join(str(p) for p in parts)
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in raw)


def _checkpoint_dir():
    path = V4_CSV / 'exp9_checkpoints'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_path(dataset: str, backbone: str):
    return _checkpoint_dir() / f'{_safe_part_name(dataset, backbone)}.csv'

TIERB_PERTURBATIONS = [
    ('gaussian_blur', 12),
    ('compound', 'severe'),
    ('resize', 2.0),
    ('defocus_blur', 5),
    ('illumination', 0.7),
    ('jpeg', 10),
    ('red_channel_shift', -45),
    ('green_channel_shift', -45),
    ('blue_channel_shift', -45),
    ('gaussian_noise', 0.10),
    ('shot_noise', 15),
    ('impulse_noise', 0.05),
    ('motion_blur', 15),
    ('zoom_blur', 1.20),
    ('contrast', 0.50),
    ('pixelate', 0.25),
    ('scratch', 'severe'),
    ('rotation', 45),
    ('compound_optical', 'severe'),
    ('compound_digital', 'severe'),
    ('compound_field', 'severe'),
]


def _load_accuracy_tiera() -> pd.DataFrame:
    """Load Tier-A accuracy for comparison."""
    path = V4_CSV / 'exp1_accuracy_matrix.csv'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df.perturbation != 'clean'].copy()
    df['severity_key'] = df['severity'].astype(str)
    return df


def _summarize_tierb(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=['drop', 'feature_drift'])
    r_overall, p_overall = pearsonr(valid['feature_drift'].values, valid['drop'].values) if len(valid) > 5 else (np.nan, np.nan)

    corr_by_ds = []
    for ds, sub in valid.groupby('dataset'):
        if len(sub) < 5:
            continue
        r, p = pearsonr(sub['feature_drift'].values, sub['drop'].values)
        corr_by_ds.append({'dataset': ds, 'r': float(r), 'p': float(p), 'n': len(sub)})
    df_corr = pd.DataFrame(corr_by_ds)
    logger.info('Tier-B overall: n=%d, r=%.3f, p=%.2e', len(valid), r_overall, p_overall)

    df_a = pd.read_csv(V4_CSV / 'exp1b_feature_geometry.csv')
    summary_rows = []
    for tier, df_t, label in [('Tier-A', df_a, 'Tier-A (controlled)'), ('Tier-B', df, 'Tier-B (wild)')]:
        sub = df_t.dropna(subset=['drop', 'feature_drift'])
        if len(sub) < 5:
            continue
        r, p = pearsonr(sub['feature_drift'].values, sub['drop'].values)
        summary_rows.append({
            'tier': tier, 'label': label,
            'r': float(r), 'p': float(p), 'n': len(sub),
            'mean_drift': float(sub['feature_drift'].mean()),
            'mean_drop': float(sub['drop'].mean()),
        })
    df_summary = pd.DataFrame(summary_rows)
    return {'tierb_correlations': df_corr, 'tier_comparison': df_summary}


def _read_checkpoints(datasets: list, backbones: list) -> pd.DataFrame:
    frames = []
    for ds in datasets:
        for bb in backbones:
            path = _checkpoint_path(ds, bb)
            if path.exists():
                frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_exp9(datasets: list = None, backbones: list = None, save: bool = True) -> dict:
    if datasets is None:
        datasets = TIER_B
    if backbones is None:
        backbones = BB_ORDER
    rows = []

    for ds_name in datasets:
        logger.info('=== Exp 9: %s ===', ds_name)

        for bb in backbones:
            try:
                feat_clean, labels_clean, _ = load_cache(bb, ds_name, 'original')
            except FileNotFoundError:
                logger.warning('Missing clean cache: %s/%s', bb, ds_name)
                continue

            geom_clean = geometry_summary(feat_clean, labels_clean)

            for pert_name, value in TIERB_PERTURBATIONS:
                tag = cache_tag_for(pert_name, value)
                try:
                    feat_shift, labels_shift, _ = load_cache(bb, ds_name, tag)
                except FileNotFoundError:
                    logger.debug('Missing: %s/%s/%s', bb, ds_name, tag)
                    continue

                drift = feature_drift(feat_clean, feat_shift)
                geom_shift = geometry_summary(feat_shift, labels_shift)
                acc = knn_accuracy_gallery_query(feat_clean, labels_clean, feat_shift, labels_shift)

                # Clean baseline accuracy
                try:
                    from wood_spatial.analysis.statistical_tests import knn_accuracy_cv
                    acc_clean, _ = knn_accuracy_cv(feat_clean, labels_clean)
                except Exception:
                    acc_clean = np.nan

                drop = (acc_clean - acc) if not np.isnan(acc_clean) else np.nan

                rows.append({
                    'dataset': ds_name,
                    'backbone': bb,
                    'perturbation': pert_name,
                    'severity': value,
                    'feature_drift': drift,
                    'delta_fgcs': geom_shift['fgcs'] - geom_clean['fgcs'],
                    'inter_collapse': geom_clean['inter'] - geom_shift['inter'],
                    'accuracy': acc,
                    'acc_clean': acc_clean,
                    'drop': drop,
                })

    df = pd.DataFrame(rows)

    summaries = _summarize_tierb(df) if len(df) else {
        'tierb_correlations': pd.DataFrame(),
        'tier_comparison': pd.DataFrame(),
    }
    df_corr = summaries['tierb_correlations']
    df_summary = summaries['tier_comparison']
    logger.info('\n=== Tier-A vs Tier-B comparison ===\n%s', df_summary.to_string(index=False))

    results = {
        'tierb_geometry': df,
        'tierb_correlations': df_corr,
        'tier_comparison': df_summary,
    }

    if save:
        df.to_csv(V4_CSV / 'exp9_tierb_geometry.csv', index=False)
        df_corr.to_csv(V4_CSV / 'exp9_tierb_correlations.csv', index=False)
        df_summary.to_csv(V4_CSV / 'exp9_tier_comparison.csv', index=False)
        logger.info('Saved Exp 9 results to %s', V4_CSV)

    return results


def run_checkpoint(dataset: str, backbone: str, force: bool = False):
    path = _checkpoint_path(dataset, backbone)
    if path.exists() and not force:
        logger.info('%s/%s checkpoint hit: %s', dataset, backbone, path.name)
        return pd.read_csv(path)
    result = run_exp9(datasets=[dataset], backbones=[backbone], save=False)
    df = result['tierb_geometry']
    df.to_csv(path, index=False)
    logger.info('%s/%s checkpoint saved: %d rows', dataset, backbone, len(df))
    return df


def finalize_from_checkpoints(datasets: list = None, backbones: list = None):
    datasets = datasets or TIER_B
    backbones = backbones or BB_ORDER
    df = _read_checkpoints(datasets, backbones)
    if df.empty:
        raise RuntimeError(f'No Exp9 checkpoints found in {_checkpoint_dir()}')
    summaries = _summarize_tierb(df)
    df.to_csv(V4_CSV / 'exp9_tierb_geometry.csv', index=False)
    summaries['tierb_correlations'].to_csv(V4_CSV / 'exp9_tierb_correlations.csv', index=False)
    summaries['tier_comparison'].to_csv(V4_CSV / 'exp9_tier_comparison.csv', index=False)
    return {'tierb_geometry': df, **summaries}


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Run Exp9 Tier-B validation.')
    parser.add_argument('--datasets', nargs='*', default=None)
    parser.add_argument('--backbones', nargs='*', default=None)
    parser.add_argument('--checkpoint-only', action='store_true')
    parser.add_argument('--finalize-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    t0 = time.time()
    datasets = args.datasets or TIER_B
    backbones = args.backbones or BB_ORDER
    if args.finalize_only:
        results = finalize_from_checkpoints(datasets, backbones)
    elif args.checkpoint_only:
        for ds in datasets:
            for bb in backbones:
                run_checkpoint(ds, bb, force=args.force)
        results = {'tierb_geometry': _read_checkpoints(datasets, backbones),
                   'tierb_correlations': pd.DataFrame(), 'tier_comparison': pd.DataFrame()}
    else:
        results = run_exp9(datasets=datasets, backbones=backbones)
    logger.info('Exp 9 done in %.1f min', (time.time() - t0) / 60)

    print('\n=== Tier-B Feature Geometry Correlations ===')
    print(results['tierb_correlations'].to_string(index=False))
    print('\n=== Tier-A vs Tier-B Comparison ===')
    print(results['tier_comparison'].to_string(index=False))


if __name__ == '__main__':
    main()
