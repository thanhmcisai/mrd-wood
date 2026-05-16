"""
Wood Spatial — Experiment 1b: Feature Geometry Distortion
==========================================================
Quantifies how distribution shift distorts global feature geometry.

Usage:
    python -m wood_spatial.experiments.exp1b_feature_geometry
"""
import logging
import time
import argparse

import numpy as np
import pandas as pd

from wood_spatial.config import BB_ORDER, PERTURB_CONFIGS, TIER_A, V4_CSV
from wood_spatial.core.cache import load_cache
from wood_spatial.core.perturbations import cache_tag_for
from wood_spatial.analysis.feature_geometry import feature_drift, geometry_summary
from wood_spatial.analysis.statistical_tests import safe_pearsonr

logger = logging.getLogger(__name__)


def _safe_part_name(*parts: str) -> str:
    raw = '__'.join(str(p) for p in parts)
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in raw)


def _checkpoint_dir():
    path = V4_CSV / 'exp1b_checkpoints'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_path(dataset: str, backbone: str):
    return _checkpoint_dir() / f'{_safe_part_name(dataset, backbone)}.csv'


def _read_checkpoints(datasets: list, backbones: list) -> pd.DataFrame:
    frames = []
    for ds in datasets:
        for bb in backbones:
            path = _checkpoint_path(ds, bb)
            if path.exists():
                frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _correlations(df_geom: pd.DataFrame) -> pd.DataFrame:
    correlations = []
    if len(df_geom) > 0 and 'drop' in df_geom:
        for metric in ['feature_drift', 'delta_intra', 'delta_inter', 'delta_fgcs', 'delta_fsr']:
            r, p = safe_pearsonr(df_geom[metric].values, df_geom['drop'].values)
            correlations.append({'metric': metric, 'target': 'drop', 'r': r, 'p': p, 'n': len(df_geom)})
    return pd.DataFrame(correlations)


def _severity_key(value) -> str:
    return str(value)


def _load_exp1_accuracy() -> pd.DataFrame:
    path = V4_CSV / 'exp1_accuracy_matrix.csv'
    if not path.exists():
        logger.warning('Exp 1 accuracy matrix not found: %s', path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['severity_key'] = df['severity'].astype(str)
    return df


def _lookup_accuracy(df_acc: pd.DataFrame, dataset: str, backbone: str, perturbation: str, severity) -> dict:
    if df_acc.empty:
        return {'accuracy': np.nan, 'acc_clean': np.nan, 'drop': np.nan}
    sev_key = _severity_key(severity)
    row = df_acc[
        (df_acc['dataset'] == dataset)
        & (df_acc['backbone'] == backbone)
        & (df_acc['perturbation'] == perturbation)
        & (df_acc['severity_key'] == sev_key)
    ]
    if len(row) == 0:
        return {'accuracy': np.nan, 'acc_clean': np.nan, 'drop': np.nan}
    r = row.iloc[0]
    return {
        'accuracy': float(r['accuracy']),
        'acc_clean': float(r['acc_clean']),
        'drop': float(r['drop']),
    }


def run_exp1b(
    datasets: list = None,
    backbones: list = None,
    save: bool = True,
) -> dict:
    """Run feature geometry distortion analysis."""
    if datasets is None:
        datasets = TIER_A
    if backbones is None:
        backbones = BB_ORDER

    df_acc = _load_exp1_accuracy()
    rows = []

    for ds_name in datasets:
        logger.info('=== Exp 1b: %s ===', ds_name)
        clean_geometry = {}
        clean_features = {}
        clean_labels = {}

        for bb in backbones:
            try:
                feat_clean, labels_clean, _ = load_cache(bb, ds_name, 'original')
            except FileNotFoundError:
                logger.warning('Missing clean cache: %s/%s', bb, ds_name)
                continue

            clean_features[bb] = feat_clean
            clean_labels[bb] = labels_clean
            clean_geometry[bb] = geometry_summary(feat_clean, labels_clean)

        for pert_name, pcfg in PERTURB_CONFIGS.items():
            for value in pcfg['values']:
                tag = cache_tag_for(pert_name, value)
                for bb in backbones:
                    if bb not in clean_features:
                        continue
                    try:
                        feat_shift, labels_shift, _ = load_cache(bb, ds_name, tag)
                    except FileNotFoundError:
                        logger.debug('Missing shifted cache: %s/%s/%s', bb, ds_name, tag)
                        continue

                    geom_clean = clean_geometry[bb]
                    geom_shift = geometry_summary(feat_shift, labels_shift)
                    acc = _lookup_accuracy(df_acc, ds_name, bb, pert_name, value)

                    row = {
                        'dataset': ds_name,
                        'backbone': bb,
                        'perturbation': pert_name,
                        'severity': value,
                        'tag': tag,
                        'feature_drift': feature_drift(clean_features[bb], feat_shift),
                        'intra_clean': geom_clean['intra'],
                        'intra_shift': geom_shift['intra'],
                        'delta_intra': geom_shift['intra'] - geom_clean['intra'],
                        'inter_clean': geom_clean['inter'],
                        'inter_shift': geom_shift['inter'],
                        'delta_inter': geom_shift['inter'] - geom_clean['inter'],
                        'fgcs_clean': geom_clean['fgcs'],
                        'fgcs_shift': geom_shift['fgcs'],
                        'delta_fgcs': geom_shift['fgcs'] - geom_clean['fgcs'],
                        'fsr_clean': geom_clean['fsr'],
                        'fsr_shift': geom_shift['fsr'],
                        'delta_fsr': geom_shift['fsr'] - geom_clean['fsr'],
                    }
                    row.update(acc)
                    rows.append(row)

        logger.info('  %s: %d geometry rows so far', ds_name, len(rows))

    df_geom = pd.DataFrame(rows)

    df_corr = _correlations(df_geom)

    if save:
        df_geom.to_csv(V4_CSV / 'exp1b_feature_geometry.csv', index=False)
        df_corr.to_csv(V4_CSV / 'exp1b_feature_geometry_correlations.csv', index=False)
        logger.info('Saved Exp 1b results to %s', V4_CSV)

    return {'feature_geometry': df_geom, 'correlations': df_corr}


def run_checkpoint(dataset: str, backbone: str, force: bool = False):
    path = _checkpoint_path(dataset, backbone)
    if path.exists() and not force:
        logger.info('%s/%s checkpoint hit: %s', dataset, backbone, path.name)
        return pd.read_csv(path)
    result = run_exp1b(datasets=[dataset], backbones=[backbone], save=False)
    df = result['feature_geometry']
    df.to_csv(path, index=False)
    logger.info('%s/%s checkpoint saved: %d rows', dataset, backbone, len(df))
    return df


def finalize_from_checkpoints(datasets: list = None, backbones: list = None):
    datasets = datasets or TIER_A
    backbones = backbones or BB_ORDER
    df_geom = _read_checkpoints(datasets, backbones)
    if df_geom.empty:
        raise RuntimeError(f'No Exp1b checkpoints found in {_checkpoint_dir()}')
    df_corr = _correlations(df_geom)
    df_geom.to_csv(V4_CSV / 'exp1b_feature_geometry.csv', index=False)
    df_corr.to_csv(V4_CSV / 'exp1b_feature_geometry_correlations.csv', index=False)
    return {'feature_geometry': df_geom, 'correlations': df_corr}


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Run Exp1b feature geometry.')
    parser.add_argument('--datasets', nargs='*', default=None)
    parser.add_argument('--backbones', nargs='*', default=None)
    parser.add_argument('--checkpoint-only', action='store_true')
    parser.add_argument('--finalize-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    t0 = time.time()
    datasets = args.datasets or TIER_A
    backbones = args.backbones or BB_ORDER
    if args.finalize_only:
        results = finalize_from_checkpoints(datasets, backbones)
    elif args.checkpoint_only:
        for ds in datasets:
            for bb in backbones:
                run_checkpoint(ds, bb, force=args.force)
        results = {'feature_geometry': _read_checkpoints(datasets, backbones), 'correlations': pd.DataFrame()}
    else:
        results = run_exp1b(datasets=datasets, backbones=backbones)
    logger.info('Exp 1b done in %.1f min', (time.time() - t0) / 60)

    df = results['feature_geometry']
    if len(df) > 0:
        print('\n=== Feature Drift by Perturbation ===')
        for pert, grp in df.groupby('perturbation'):
            print(f'  {pert:25s}: {grp["feature_drift"].mean():.4f}')

    df_corr = results['correlations']
    if len(df_corr) > 0:
        print('\n=== Feature Geometry vs Accuracy Drop ===')
        for _, row in df_corr.iterrows():
            print(f"  {row['metric']:15s}: r={row['r']:.3f}, p={row['p']:.4g}")


if __name__ == '__main__':
    main()
