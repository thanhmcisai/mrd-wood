"""
Wood Spatial — Experiment 4: Cross-Dataset & External Generalization
=================================================================
4a: Verify degradation patterns replicate across Tier-A datasets.
4b: Test on unseen Tier-B external-acquisition datasets.

Answers RQ4: Does the deployment gap replicate across datasets and external acquisition protocols?

Usage:
    python -m wood_spatial.experiments.exp4_cross_dataset
"""
import logging
import time

import numpy as np
import pandas as pd

from wood_spatial.config import (
    ALL_DATASETS, TIER_A, TIER_B, BB_ORDER, V4_CSV,
    PERTURB_CONFIGS,
)
from wood_spatial.core.cache import load_cache, cache_exists
from wood_spatial.core.perturbations import cache_tag_for
from wood_spatial.analysis.statistical_tests import (
    knn_accuracy_cv,
    knn_accuracy_gallery_query,
    kendalls_w,
    safe_pearsonr,
    sig_str,
)

logger = logging.getLogger(__name__)


def run_exp4(
    backbones: list = None,
    save: bool = True,
) -> dict:
    """
    Run Experiment 4: Cross-Dataset & Wild Generalization.

    Returns dict of DataFrames: 'tierB_accuracy', 'degradation_corr', 'kendallW'.
    """
    if backbones is None:
        backbones = BB_ORDER

    # ── 4a: Tier-B within-dataset kNN (clean) ─────────────────────────────
    logger.info('=== Exp 4a: Tier-B clean accuracy ===')

    tier_b_rows = []
    for ds_name in TIER_B:
        for bb in backbones:
            try:
                feat, labels, paths = load_cache(bb, ds_name, 'original')
            except FileNotFoundError:
                logger.debug('Missing: %s/%s', bb, ds_name)
                continue
            acc, std = knn_accuracy_cv(feat, labels)
            tier_b_rows.append({
                'dataset': ds_name, 'backbone': bb,
                'accuracy': acc, 'accuracy_std': std,
            })
            logger.info('  %s/%s: %.3f +/- %.3f', bb, ds_name, acc, std)

    df_tierB = pd.DataFrame(tier_b_rows)

    # ── 4b: Tier-B perturbed accuracy ─────────────────────────────────────
    logger.info('=== Exp 4b: Tier-B perturbed accuracy ===')

    tierB_pert_rows = []
    for ds_name in TIER_B:
        for bb in backbones:
            try:
                feat_clean, labels_clean = load_cache(bb, ds_name, 'original')[:2]
            except FileNotFoundError:
                continue

            for pert_name, pcfg in PERTURB_CONFIGS.items():
                for value in pcfg['values']:
                    tag = cache_tag_for(pert_name, value)
                    if not cache_exists(bb, ds_name, tag):
                        continue
                    try:
                        feat_pert, labels_pert, _ = load_cache(bb, ds_name, tag)
                    except FileNotFoundError:
                        continue

                    acc = knn_accuracy_gallery_query(
                        feat_clean, labels_clean, feat_pert, labels_pert,
                    )
                    # Get clean baseline
                    clean_row = [r for r in tier_b_rows
                                 if r['backbone'] == bb and r['dataset'] == ds_name]
                    acc_clean = clean_row[0]['accuracy'] if clean_row else np.nan
                    drop = acc_clean - acc if not np.isnan(acc_clean) else np.nan

                    tierB_pert_rows.append({
                        'dataset': ds_name, 'backbone': bb,
                        'perturbation': pert_name, 'severity': value,
                        'accuracy': acc, 'acc_clean': acc_clean, 'drop': drop,
                    })

    df_tierB_pert = pd.DataFrame(tierB_pert_rows)

    # ── 4c: Degradation pattern comparison (Tier-A vs Tier-B) ─────────────
    logger.info('=== Exp 4c: Degradation pattern comparison ===')

    # Compare backbone rankings between tiers
    # Build per-perturbation accuracy rankings for each tier
    corr_rows = []

    for pert_name in PERTURB_CONFIGS:
        # Tier-A mean accuracy per backbone for this perturbation
        try:
            from wood_spatial.experiments.exp1_robustness import run_exp1
            # Try loading cached Exp 1 results
            exp1_csv = V4_CSV / 'exp1_accuracy_matrix.csv'
            if exp1_csv.exists():
                df_exp1 = pd.read_csv(exp1_csv)
                tierA_pert = df_exp1[(df_exp1['perturbation'] == pert_name)]
                tierA_mean = tierA_pert.groupby('backbone')['accuracy'].mean()
            else:
                tierA_mean = pd.Series(dtype=float)
        except Exception:
            tierA_mean = pd.Series(dtype=float)

        # Tier-B mean accuracy for this perturbation
        if len(df_tierB_pert) == 0:
            continue
        tierB_pert = df_tierB_pert[df_tierB_pert['perturbation'] == pert_name]
        if len(tierB_pert) == 0:
            continue
        tierB_mean = tierB_pert.groupby('backbone')['accuracy'].mean()

        # Correlation of backbone accuracy vectors
        common = list(set(tierA_mean.index) & set(tierB_mean.index))
        if len(common) >= 3:
            x = [tierA_mean[bb] for bb in common]
            y = [tierB_mean[bb] for bb in common]
            r, p = safe_pearsonr(x, y)
            corr_rows.append({
                'perturbation': pert_name, 'pearson_r': r, 'p_value': p,
                'sig': sig_str(p), 'n_backbones': len(common),
            })

    df_corr = pd.DataFrame(corr_rows)

    # ── 4d: Kendall's W across tiers ──────────────────────────────────────
    logger.info('=== Exp 4d: Kendall W across datasets ===')

    # Build ranking matrix: rows = datasets (both tiers), cols = backbones
    w_rows = []
    all_datasets_check = TIER_A + TIER_B

    # Rankings from clean accuracy
    rank_matrix_clean = []
    for ds_name in all_datasets_check:
        row = []
        for bb in backbones:
            matching = [r for r in tier_b_rows if r['backbone'] == bb and r['dataset'] == ds_name]
            if matching:
                row.append(matching[0]['accuracy'])
            else:
                # Try Exp 1 clean
                try:
                    exp1_csv = V4_CSV / 'exp1_accuracy_matrix.csv'
                    if exp1_csv.exists():
                        df_exp1 = pd.read_csv(exp1_csv)
                        match = df_exp1[(df_exp1['dataset'] == ds_name) &
                                        (df_exp1['perturbation'] == 'clean') &
                                        (df_exp1['backbone'] == bb)]
                        if len(match) > 0:
                            row.append(match.iloc[0]['accuracy'])
                            continue
                except Exception:
                    pass
                row.append(np.nan)

        if not any(np.isnan(v) for v in row):
            rank_matrix_clean.append(row)

    if len(rank_matrix_clean) >= 2:
        w_clean = kendalls_w(np.array(rank_matrix_clean))
        w_rows.append({'metric': 'clean_accuracy', 'kendall_w': w_clean,
                       'n_datasets': len(rank_matrix_clean)})
        logger.info('Kendall W (clean): %.3f across %d datasets', w_clean, len(rank_matrix_clean))

    df_kendall = pd.DataFrame(w_rows)

    # ── Compile ───────────────────────────────────────────────────────────
    results = {
        'tierB_accuracy': df_tierB,
        'tierB_perturbed': df_tierB_pert,
        'degradation_corr': df_corr,
        'kendallW': df_kendall,
    }

    if save:
        df_tierB.to_csv(V4_CSV / 'exp4_tierB_accuracy.csv', index=False)
        if len(df_tierB_pert) > 0:
            df_tierB_pert.to_csv(V4_CSV / 'exp4_tierB_perturbed.csv', index=False)
        df_corr.to_csv(V4_CSV / 'exp4_degradation_correlation.csv', index=False)
        df_kendall.to_csv(V4_CSV / 'exp4_kendallW.csv', index=False)
        logger.info('Saved Exp 4 results to %s', V4_CSV)

    return results


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    t0 = time.time()
    results = run_exp4()
    logger.info('Exp 4 done in %.1f min', (time.time() - t0) / 60)

    # Summary
    print('\n=== Tier-B Clean Accuracy ===')
    for _, row in results['tierB_accuracy'].iterrows():
        print(f"  {row['backbone']:20s} | {row['dataset']:10s} | {row['accuracy']:.3f}")

    print('\n=== Degradation Pattern Correlation (Tier-A vs Tier-B) ===')
    for _, row in results['degradation_corr'].iterrows():
        print(f"  {row['perturbation']:20s}: r={row['pearson_r']:.3f}, "
              f"p={row['p_value']:.4g} {row['sig']}")


if __name__ == '__main__':
    main()
