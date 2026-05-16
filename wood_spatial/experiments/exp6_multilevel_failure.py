"""
Wood Spatial — Experiment 6: Multi-level Failure Analysis
==========================================================
Integrates decision failure, feature geometry distortion, spatial stability,
and attention drift into one analysis table.

Usage:
    python -m wood_spatial.experiments.exp6_multilevel_failure
"""
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wood_spatial.config import BB_ORDER, V4_CSV
from wood_spatial.analysis.statistical_tests import safe_pearsonr

logger = logging.getLogger(__name__)


MERGE_KEYS = ['dataset', 'backbone', 'perturbation', 'severity_key']


def _severity_key(value) -> str:
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return f'{f:g}'
    except (TypeError, ValueError):
        return str(value)


def _read_csv(name: str, required: bool = True) -> pd.DataFrame:
    path = V4_CSV / name
    if not path.exists():
        msg = f'Missing required CSV: {path}'
        if required:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'severity' in df.columns:
        df['severity_key'] = df['severity'].map(_severity_key)
    return df


def _aggregate_spatial() -> pd.DataFrame:
    df_csi = _read_csv('exp2_csi.csv', required=False)
    if df_csi.empty:
        return pd.DataFrame(columns=MERGE_KEYS + ['csi'])
    agg_cols = {'csi': ('csi', 'mean')}
    for col in ['csi_bo', 'csi_hungarian', 'csi_permutation_gap']:
        if col in df_csi.columns:
            agg_cols[col] = (col, 'mean')
    return df_csi.groupby(MERGE_KEYS, as_index=False).agg(**agg_cols)


def _aggregate_attention() -> pd.DataFrame:
    df_shift = _read_csv('exp3_cam_shift.csv', required=False)
    if df_shift.empty:
        return pd.DataFrame(columns=MERGE_KEYS + ['cam_shift_jsd'])
    agg_cols = {'cam_shift_jsd': ('cam_shift_jsd', 'mean')}
    for col in ['cam_shift_js_distance', 'cam_shift_js_divergence']:
        if col in df_shift.columns:
            agg_cols[col] = (col, 'mean')
    return df_shift.groupby(MERGE_KEYS, as_index=False).agg(**agg_cols)


def _aggregate_entropy() -> pd.DataFrame:
    df_entropy = _read_csv('exp3_cam_entropy.csv', required=False)
    if df_entropy.empty:
        return pd.DataFrame(columns=['dataset', 'backbone', 'cam_entropy_clean'])
    clean = df_entropy[df_entropy['condition'] == 'clean']
    return clean.groupby(['dataset', 'backbone'], as_index=False).agg(
        cam_entropy_clean=('entropy', 'mean')
    )


def _aggregate_sgi() -> pd.DataFrame:
    df_sgi = _read_csv('exp2_sgi.csv', required=False)
    if df_sgi.empty:
        return pd.DataFrame(columns=['dataset', 'backbone', 'sgi_clean'])
    clean = df_sgi[df_sgi['condition'] == 'clean']
    return clean.groupby(['dataset', 'backbone'], as_index=False).agg(sgi_clean=('sgi', 'mean'))


def build_multilevel_table() -> pd.DataFrame:
    df_geom = _read_csv('exp1b_feature_geometry.csv')
    df_acc = _read_csv('exp1_accuracy_matrix.csv')

    acc_cols = MERGE_KEYS + ['accuracy', 'acc_clean', 'drop']
    df_acc = df_acc[df_acc['perturbation'] != 'clean'][acc_cols].drop_duplicates(MERGE_KEYS)

    table = df_geom.merge(df_acc, on=MERGE_KEYS, how='left', suffixes=('', '_exp1'))
    for col in ['accuracy', 'acc_clean', 'drop']:
        alt = f'{col}_exp1'
        if alt in table.columns:
            table[col] = table[col].combine_first(table[alt]) if col in table.columns else table[alt]
            table = table.drop(columns=[alt])

    spatial = _aggregate_spatial()
    attention = _aggregate_attention()
    entropy = _aggregate_entropy()
    sgi = _aggregate_sgi()

    table = table.merge(spatial, on=MERGE_KEYS, how='left')
    table = table.merge(attention, on=MERGE_KEYS, how='left')
    table = table.merge(entropy, on=['dataset', 'backbone'], how='left')
    table = table.merge(sgi, on=['dataset', 'backbone'], how='left')

    table['spatial_instability'] = 1.0 - table['csi']
    if 'csi_hungarian' in table.columns:
        table['spatial_instability_hungarian'] = 1.0 - table['csi_hungarian']
    table['inter_collapse'] = -table['delta_inter']
    table['fsr_collapse'] = -table['delta_fsr']
    return table


def _corr_rows(df: pd.DataFrame) -> list:
    targets = ['drop']
    metrics = [
        'feature_drift', 'delta_intra', 'inter_collapse', 'delta_fgcs',
        'fsr_collapse', 'spatial_instability', 'csi', 'cam_shift_jsd',
        'csi_hungarian', 'spatial_instability_hungarian',
        'csi_permutation_gap', 'cam_shift_js_distance',
        'cam_shift_js_divergence', 'cam_entropy_clean', 'sgi_clean',
    ]
    rows = []
    for target in targets:
        for metric in metrics:
            if metric not in df.columns or target not in df.columns:
                continue
            sub = df[[metric, target]].replace([np.inf, -np.inf], np.nan).dropna()
            r, p = safe_pearsonr(sub[metric].values, sub[target].values)
            rows.append({'metric': metric, 'target': target, 'r': r, 'p': p, 'n': len(sub)})
    return rows


def _profile(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    metrics = [
        'drop', 'feature_drift', 'delta_fgcs', 'inter_collapse',
        'spatial_instability', 'csi', 'csi_hungarian', 'csi_permutation_gap',
        'cam_shift_jsd', 'cam_shift_js_distance', 'cam_shift_js_divergence',
        'cam_entropy_clean', 'sgi_clean',
    ]
    available = [m for m in metrics if m in df.columns]
    return df.groupby(group_cols, as_index=False)[available].mean()


def run_exp6(save: bool = True) -> dict:
    """Run multi-level failure integration and correlation analysis."""
    table = build_multilevel_table()
    corr = pd.DataFrame(_corr_rows(table))
    backbone_profile = _profile(table, ['backbone'])
    perturbation_profile = _profile(table, ['perturbation'])

    if 'backbone' in backbone_profile.columns:
        backbone_profile['backbone'] = pd.Categorical(
            backbone_profile['backbone'], categories=BB_ORDER, ordered=True,
        )
        backbone_profile = backbone_profile.sort_values('backbone')

    if save:
        table.to_csv(V4_CSV / 'exp6_multilevel_table.csv', index=False)
        corr.to_csv(V4_CSV / 'exp6_multilevel_correlations.csv', index=False)
        backbone_profile.to_csv(V4_CSV / 'exp6_backbone_failure_profile.csv', index=False)
        perturbation_profile.to_csv(V4_CSV / 'exp6_perturbation_failure_profile.csv', index=False)
        logger.info('Saved Exp 6 results to %s', V4_CSV)

    return {
        'multilevel_table': table,
        'correlations': corr,
        'backbone_profile': backbone_profile,
        'perturbation_profile': perturbation_profile,
    }


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    t0 = time.time()
    results = run_exp6()
    logger.info('Exp 6 done in %.1f min', (time.time() - t0) / 60)

    corr = results['correlations'].sort_values('r', key=lambda s: s.abs(), ascending=False)
    print('\n=== Multi-level Correlations with Accuracy Drop ===')
    for _, row in corr.iterrows():
        print(f"  {row['metric']:20s}: r={row['r']:.3f}, p={row['p']:.4g}, n={int(row['n'])}")

    profile = results['backbone_profile']
    if len(profile) > 0:
        print('\n=== Backbone Failure Profile ===')
        cols = ['backbone', 'drop', 'feature_drift', 'delta_fgcs', 'csi', 'cam_shift_jsd']
        cols = [c for c in cols if c in profile.columns]
        print(profile[cols].to_string(index=False))


if __name__ == '__main__':
    main()
