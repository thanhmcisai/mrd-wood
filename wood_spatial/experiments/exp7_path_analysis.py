"""
Wood Spatial — Experiment 7: Multi-level Path Analysis
=======================================================
Hierarchical regression, partial correlations, and incremental R² showing
the contribution of each framework level to prediction failure.

Key questions:
  Q1: Does feature geometry predict drop beyond perturbation type alone?
  Q2: Does each level of the framework add incremental explanatory power?
  Q3: Is the feature geometry effect robust to backbone / dataset differences?

Usage:
    python -m wood_spatial.experiments.exp7_path_analysis
"""
import logging
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from wood_spatial.config import BB_ORDER, V4_CSV

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def partial_corr(df: pd.DataFrame, x_col: str, y_col: str, control_cols: list):
    """Partial Pearson r between x and y after regressing out controls."""
    sub = df[[x_col, y_col] + control_cols].dropna()
    if len(sub) < 30:
        return np.nan, np.nan, len(sub)
    Xc = sub[control_cols].values
    x = sub[x_col].values
    y = sub[y_col].values
    res_x = x - LinearRegression().fit(Xc, x).predict(Xc)
    res_y = y - LinearRegression().fit(Xc, y).predict(Xc)
    r, p = pearsonr(res_x, res_y)
    return float(r), float(p), len(sub)


def hierarchical_r2(df: pd.DataFrame, y_col: str, model_specs: list):
    """
    Hierarchical regression: compute R² for each model in sequence.

    model_specs: list of (label, feature_cols)
    Returns DataFrame with model, r2, delta_r2, n.
    """
    rows = []
    prev_r2 = 0.0
    y = df[y_col].values
    for label, cols in model_specs:
        sub = df[[y_col] + cols].dropna()
        if len(sub) < 30:
            rows.append({'model': label, 'r2': np.nan, 'delta_r2': np.nan, 'n': len(sub)})
            continue
        X = StandardScaler().fit_transform(sub[cols].values)
        r2 = LinearRegression().fit(X, sub[y_col].values).score(X, sub[y_col].values)
        rows.append({'model': label, 'r2': float(r2), 'delta_r2': float(r2 - prev_r2), 'n': len(sub)})
        prev_r2 = r2
    return pd.DataFrame(rows)


def robust_corr_by_group(df: pd.DataFrame, x_col: str, y_col: str, group_col: str):
    """Pearson r per group to assess consistency."""
    rows = []
    for g, sub in df.groupby(group_col):
        sub = sub[[x_col, y_col]].dropna()
        if len(sub) < 10:
            continue
        r, p = pearsonr(sub[x_col].values, sub[y_col].values)
        rows.append({'group': g, 'r': float(r), 'p': float(p), 'n': len(sub)})
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_exp7(save: bool = True) -> dict:
    df = pd.read_csv(V4_CSV / 'exp6_multilevel_table.csv')
    df = df.dropna(subset=['drop', 'feature_drift'])

    # Dummy-encode perturbation type for controls
    pert_dummies = pd.get_dummies(df['perturbation'], prefix='pert', drop_first=True)
    df_aug = pd.concat([df.reset_index(drop=True), pert_dummies.reset_index(drop=True)], axis=1)
    control_cols = list(pert_dummies.columns)

    # ── 1. Partial correlations controlling for perturbation type ─────────────
    logger.info('=== Exp 7 Step 1: Partial correlations ===')
    metrics = [
        'feature_drift', 'delta_fgcs', 'inter_collapse', 'delta_intra',
        'fsr_collapse', 'csi', 'csi_hungarian', 'csi_permutation_gap',
        'cam_shift_jsd', 'cam_shift_js_distance', 'cam_shift_js_divergence',
        'cam_entropy_clean', 'sgi_clean',
    ]
    partial_rows = []
    for metric in metrics:
        if metric not in df_aug.columns:
            continue
        r_raw, p_raw = pearsonr(
            df_aug[metric].dropna().values,
            df_aug.loc[df_aug[metric].notna(), 'drop'].values,
        ) if df_aug[metric].notna().sum() > 30 else (np.nan, np.nan)
        r_part, p_part, n = partial_corr(df_aug, metric, 'drop', control_cols)
        partial_rows.append({
            'metric': metric, 'r_raw': r_raw, 'p_raw': p_raw,
            'r_partial': r_part, 'p_partial': p_part, 'n': n,
        })
        logger.info('  %-20s: raw r=%.3f | partial r=%.3f (p=%.2e)',
                    metric, r_raw or np.nan, r_part or np.nan, p_part or np.nan)
    df_partial = pd.DataFrame(partial_rows)

    # ── 2. Hierarchical regression ────────────────────────────────────────────
    logger.info('=== Exp 7 Step 2: Hierarchical regression ===')
    model_specs = [
        ('M0: Perturbation type only', control_cols),
        ('M1: + Feature drift', control_cols + ['feature_drift']),
        ('M2: + Full feature geometry', control_cols + ['feature_drift', 'delta_fgcs', 'inter_collapse']),
        ('M3: + Spatial clustering', control_cols + ['feature_drift', 'delta_fgcs', 'inter_collapse', 'csi']),
        ('M4: + Attention drift', control_cols + ['feature_drift', 'delta_fgcs', 'inter_collapse', 'csi', 'cam_shift_jsd']),
    ]
    df_hier = hierarchical_r2(df_aug.dropna(subset=['drop', 'feature_drift', 'csi', 'cam_shift_jsd']),
                              'drop', model_specs)
    for _, row in df_hier.iterrows():
        logger.info('  %-50s R²=%.4f  ΔR²=%+.4f', row['model'], row['r2'], row['delta_r2'])

    # ── 3. Cross-group robustness ─────────────────────────────────────────────
    logger.info('=== Exp 7 Step 3: Cross-group robustness ===')
    df_by_ds = robust_corr_by_group(df, 'feature_drift', 'drop', 'dataset')
    df_by_bb = robust_corr_by_group(df, 'feature_drift', 'drop', 'backbone')
    df_by_pt = robust_corr_by_group(df, 'feature_drift', 'drop', 'perturbation')
    logger.info('By dataset:\n%s', df_by_ds.to_string(index=False))
    logger.info('By backbone:\n%s', df_by_bb.to_string(index=False))

    # ── 4. Spearman rank correlation (non-parametric) ─────────────────────────
    logger.info('=== Exp 7 Step 4: Spearman rank correlations ===')
    spearman_rows = []
    for metric in ['feature_drift', 'delta_fgcs', 'inter_collapse', 'csi',
                   'csi_hungarian', 'csi_permutation_gap',
                   'cam_shift_jsd', 'cam_shift_js_distance', 'cam_shift_js_divergence']:
        if metric not in df.columns:
            continue
        sub = df[['drop', metric]].dropna()
        r, p = spearmanr(sub['drop'].values, sub[metric].values)
        spearman_rows.append({'metric': metric, 'rho': float(r), 'p': float(p), 'n': len(sub)})
        logger.info('  %-20s: rho=%.3f (p=%.2e)', metric, r, p)
    df_spearman = pd.DataFrame(spearman_rows)

    # ── 5. Effect sizes per backbone ──────────────────────────────────────────
    logger.info('=== Exp 7 Step 5: Per-backbone path coefficients ===')
    bb_path_rows = []
    for bb, sub in df.groupby('backbone'):
        sub = sub[['drop', 'feature_drift', 'delta_fgcs']].dropna()
        if len(sub) < 20:
            continue
        r_fd, p_fd = pearsonr(sub['feature_drift'].values, sub['drop'].values)
        r_fgcs, p_fgcs = pearsonr(sub['delta_fgcs'].values, sub['drop'].values)
        bb_path_rows.append({
            'backbone': bb,
            'r_feature_drift': float(r_fd), 'p_feature_drift': float(p_fd),
            'r_delta_fgcs': float(r_fgcs), 'p_delta_fgcs': float(p_fgcs),
            'n': len(sub),
        })
    df_bb_path = pd.DataFrame(bb_path_rows)
    logger.info('\n%s', df_bb_path.to_string(index=False))

    results = {
        'partial_correlations': df_partial,
        'hierarchical_r2': df_hier,
        'by_dataset': df_by_ds,
        'by_backbone': df_by_bb,
        'by_perturbation': df_by_pt,
        'spearman': df_spearman,
        'backbone_path': df_bb_path,
    }

    if save:
        df_partial.to_csv(V4_CSV / 'exp7_partial_correlations.csv', index=False)
        df_hier.to_csv(V4_CSV / 'exp7_hierarchical_r2.csv', index=False)
        df_by_ds.to_csv(V4_CSV / 'exp7_consistency_by_dataset.csv', index=False)
        df_by_bb.to_csv(V4_CSV / 'exp7_consistency_by_backbone.csv', index=False)
        df_by_pt.to_csv(V4_CSV / 'exp7_consistency_by_perturbation.csv', index=False)
        df_spearman.to_csv(V4_CSV / 'exp7_spearman_correlations.csv', index=False)
        df_bb_path.to_csv(V4_CSV / 'exp7_backbone_path_coefficients.csv', index=False)
        logger.info('Saved Exp 7 results to %s', V4_CSV)

    return results


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    t0 = time.time()
    results = run_exp7()
    logger.info('Exp 7 done in %.1f min', (time.time() - t0) / 60)

    print('\n=== Hierarchical R² ===')
    print(results['hierarchical_r2'].to_string(index=False))
    print('\n=== Partial Correlations (controlling for perturbation type) ===')
    print(results['partial_correlations'][['metric', 'r_raw', 'r_partial', 'p_partial', 'n']].to_string(index=False))


if __name__ == '__main__':
    main()
