"""
Wood Spatial — Experiment 1: Perturbation Robustness Benchmark
================================================================
7 backbones × expanded perturbation families × multiple severities × Tier-A datasets.

Protocol: Gallery-query kNN (train on clean, test on perturbed).
Outputs: accuracy matrix, bootstrap CIs, Friedman/Nemenyi/Wilcoxon tests.

Usage:
    python -m wood_spatial.experiments.exp1_robustness
"""
import logging
import time
import argparse

import numpy as np
import pandas as pd

from wood_spatial.config import (
    BACKBONE_CONFIGS, PERTURB_CONFIGS, ALL_DATASETS,
    TIER_A, BB_ORDER, V4_CSV, KNN_K,
)
from wood_spatial.core.cache import load_cache
from wood_spatial.core.perturbations import cache_tag_for
from wood_spatial.analysis.statistical_tests import (
    knn_accuracy_gallery_query,
    bootstrap_gallery_query_ci,
    knn_accuracy_cv,
    bootstrap_ci,
    friedman_test,
    nemenyi_posthoc,
    nemenyi_cd,
    compute_mean_ranks,
    wilcoxon_pairwise,
    kendalls_w,
    sig_str,
)

logger = logging.getLogger(__name__)


def _safe_part_name(*parts: str) -> str:
    raw = '__'.join(str(p) for p in parts)
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in raw)


def _checkpoint_dir():
    path = V4_CSV / 'exp1_checkpoints'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_paths(dataset: str, backbone: str) -> dict:
    stem = _safe_part_name(dataset, backbone)
    return {
        'accuracy': _checkpoint_dir() / f'{stem}__accuracy.csv',
        'ci': _checkpoint_dir() / f'{stem}__ci.csv',
    }


def _checkpoint_complete(dataset: str, backbone: str) -> bool:
    paths = _checkpoint_paths(dataset, backbone)
    if not (paths['accuracy'].exists() and paths['ci'].exists()):
        return False
    try:
        acc = pd.read_csv(paths['accuracy'], usecols=['perturbation', 'severity'])
    except Exception:
        return False
    expected = {(p, str(v)) for p, cfg in PERTURB_CONFIGS.items() for v in cfg['values']}
    observed = set(zip(acc['perturbation'].astype(str), acc['severity'].astype(str)))
    return expected.issubset(observed) and ('clean', '0') in observed


def _read_checkpoint_parts(datasets: list, backbones: list) -> dict:
    frames = {'accuracy': [], 'ci': []}
    for ds in datasets:
        for bb in backbones:
            if not _checkpoint_complete(ds, bb):
                continue
            paths = _checkpoint_paths(ds, bb)
            frames['accuracy'].append(pd.read_csv(paths['accuracy']))
            frames['ci'].append(pd.read_csv(paths['ci']))
    return {
        key: pd.concat(vals, ignore_index=True) if vals else pd.DataFrame()
        for key, vals in frames.items()
    }


def _compute_pair(ds_name: str, bb: str) -> tuple[list, list]:
    rows = []
    ci_rows = []
    try:
        feat_clean, labels_clean, _ = load_cache(bb, ds_name, 'original')
    except FileNotFoundError:
        logger.warning('Missing clean cache: %s/%s', bb, ds_name)
        return rows, ci_rows

    acc_clean, std_clean = knn_accuracy_cv(feat_clean, labels_clean)
    rows.append({
        'dataset': ds_name, 'perturbation': 'clean', 'severity': 0,
        'backbone': bb, 'accuracy': acc_clean, 'accuracy_std': std_clean,
        'acc_clean': acc_clean, 'drop': 0.0,
    })

    for pert_name, pcfg in PERTURB_CONFIGS.items():
        for value in pcfg['values']:
            tag = cache_tag_for(pert_name, value)
            try:
                feat_pert, labels_pert, _ = load_cache(bb, ds_name, tag)
            except FileNotFoundError:
                continue
            acc = knn_accuracy_gallery_query(feat_clean, labels_clean, feat_pert, labels_pert)
            rows.append({
                'dataset': ds_name, 'perturbation': pert_name,
                'severity': value, 'backbone': bb,
                'accuracy': acc, 'accuracy_std': 0.0,
                'acc_clean': acc_clean, 'drop': acc_clean - acc,
            })

    # Bootstrap CIs are retained for clean and representative severe conditions.
    pt, lo, hi, std = bootstrap_ci(feat_clean, labels_clean)
    ci_rows.append({
        'dataset': ds_name, 'perturbation': 'clean', 'severity': 0,
        'backbone': bb, 'point': pt, 'ci_lo': lo, 'ci_hi': hi, 'ci_std': std,
    })

    key_perts = [
        ('gaussian_blur', 12), ('defocus_blur', 11),
        ('resize', 2.00), ('jpeg', 10), ('scratch', 'severe'),
        ('compound', 'severe'), ('compound_optical', 'severe'),
        ('compound_digital', 'severe'), ('compound_field', 'severe'),
    ]
    for pert_name, value in key_perts:
        tag = cache_tag_for(pert_name, value)
        try:
            feat_pert, labels_pert, _ = load_cache(bb, ds_name, tag)
        except FileNotFoundError:
            continue
        pt, lo, hi, std = bootstrap_gallery_query_ci(
            feat_clean, labels_clean, feat_pert, labels_pert,
        )
        ci_rows.append({
            'dataset': ds_name, 'perturbation': pert_name,
            'severity': value, 'backbone': bb,
            'point': pt, 'ci_lo': lo, 'ci_hi': hi, 'ci_std': std,
        })
    return rows, ci_rows


def _finalize_results(df_acc: pd.DataFrame, df_ci: pd.DataFrame, backbones: list, save: bool) -> dict:
    logger.info('=== Exp 1 Step 3: Statistical tests ===')

    conditions = df_acc[df_acc['perturbation'] != 'clean'].groupby(
        ['dataset', 'perturbation', 'severity']
    )

    acc_matrix_rows = []
    condition_labels = []
    for (ds, pert, sev), grp in conditions:
        row = []
        for bb in backbones:
            bb_row = grp[grp['backbone'] == bb]
            if len(bb_row) == 1:
                row.append(bb_row.iloc[0]['accuracy'])
            else:
                row.append(np.nan)
        if not any(np.isnan(r) for r in row):
            acc_matrix_rows.append(row)
            condition_labels.append(f'{ds}_{pert}_{sev}')

    acc_matrix = np.array(acc_matrix_rows) if acc_matrix_rows else np.empty((0, len(backbones)))

    # Friedman
    chi2, p_friedman = friedman_test(acc_matrix)
    logger.info('Friedman: chi2=%.2f, p=%.4g %s', chi2, p_friedman, sig_str(p_friedman))

    # Nemenyi
    nemenyi_df = nemenyi_posthoc(acc_matrix, backbones)
    cd_value = nemenyi_cd(acc_matrix) if len(acc_matrix) > 0 else np.nan
    mean_ranks = compute_mean_ranks(acc_matrix, backbones)

    # Wilcoxon pairwise
    wilcoxon_result = wilcoxon_pairwise(acc_matrix, backbones)

    # Kendall's W
    w_val = kendalls_w(acc_matrix)
    logger.info('Kendall W: %.3f', w_val)

    # Compile statistical results
    stat_rows = [{
        'test': 'Friedman', 'chi2': chi2, 'p': p_friedman,
        'sig': sig_str(p_friedman), 'n_conditions': len(acc_matrix),
    }]
    stat_rows.append({
        'test': 'Kendall_W', 'chi2': np.nan, 'p': np.nan,
        'sig': '', 'n_conditions': len(acc_matrix),
        'W': w_val,
    })
    stat_rows.append({
        'test': 'Nemenyi_CD', 'chi2': np.nan, 'p': np.nan,
        'sig': '', 'n_conditions': len(acc_matrix),
        'CD': cd_value,
    })
    for bb, mr in mean_ranks.items():
        stat_rows.append({
            'test': 'mean_rank', 'backbone': bb, 'rank': mr,
        })
    df_stats = pd.DataFrame(stat_rows)

    # Wilcoxon to DataFrame
    df_wilcoxon = pd.DataFrame(wilcoxon_result['pairs'])

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        'accuracy': df_acc,
        'ci': df_ci,
        'stats': df_stats,
        'nemenyi': nemenyi_df,
        'wilcoxon': df_wilcoxon,
        'acc_matrix': acc_matrix,
        'condition_labels': condition_labels,
        'mean_ranks': mean_ranks,
    }

    if save:
        df_acc.to_csv(V4_CSV / 'exp1_accuracy_matrix.csv', index=False)
        df_ci.to_csv(V4_CSV / 'exp1_bootstrap_ci.csv', index=False)
        df_stats.to_csv(V4_CSV / 'exp1_statistical_tests.csv', index=False)
        df_wilcoxon.to_csv(V4_CSV / 'exp1_wilcoxon_pairwise.csv', index=False)
        if nemenyi_df is not None:
            nemenyi_df.to_csv(V4_CSV / 'exp1_nemenyi_pvalues.csv')
        logger.info('Saved Exp 1 results to %s', V4_CSV)

    return results


def run_exp1(
    datasets: list = None,
    backbones: list = None,
    save: bool = True,
) -> dict:
    """
    Run Experiment 1: Perturbation Robustness Benchmark.

    Returns dict of DataFrames: 'accuracy', 'ci', 'friedman', 'nemenyi', 'wilcoxon'.
    """
    if datasets is None:
        datasets = TIER_A
    if backbones is None:
        backbones = BB_ORDER

    logger.info('=== Exp 1 Step 1: kNN accuracy + bootstrap CIs ===')
    rows = []
    ci_rows = []
    for ds_name in datasets:
        for bb in backbones:
            pair_rows, pair_ci = _compute_pair(ds_name, bb)
            rows.extend(pair_rows)
            ci_rows.extend(pair_ci)
            logger.info('  %s/%s: %d accuracy rows', ds_name, bb, len(pair_rows))

    df_acc = pd.DataFrame(rows)
    df_ci = pd.DataFrame(ci_rows)
    logger.info('Accuracy matrix: %d rows', len(df_acc))
    return _finalize_results(df_acc, df_ci, backbones, save=save)


def run_checkpoint(dataset: str, backbone: str, force: bool = False):
    paths = _checkpoint_paths(dataset, backbone)
    if _checkpoint_complete(dataset, backbone) and not force:
        logger.info('%s/%s checkpoint hit', dataset, backbone)
        return
    rows, ci_rows = _compute_pair(dataset, backbone)
    pd.DataFrame(rows).to_csv(paths['accuracy'], index=False)
    pd.DataFrame(ci_rows).to_csv(paths['ci'], index=False)
    logger.info('%s/%s checkpoint saved: %d rows', dataset, backbone, len(rows))


def finalize_from_checkpoints(datasets: list = None, backbones: list = None):
    datasets = datasets or TIER_A
    backbones = backbones or BB_ORDER
    parts = _read_checkpoint_parts(datasets, backbones)
    if parts['accuracy'].empty:
        raise RuntimeError(f'No complete Exp1 checkpoints found in {_checkpoint_dir()}')
    return _finalize_results(parts['accuracy'], parts['ci'], backbones, save=True)


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Run Exp1 robustness benchmark.')
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
        results = {'accuracy': _read_checkpoint_parts(datasets, backbones)['accuracy'],
                   'stats': pd.DataFrame(), 'mean_ranks': {}}
    else:
        results = run_exp1(datasets=datasets, backbones=backbones)
    logger.info('Exp 1 done in %.1f min', (time.time() - t0) / 60)

    # Print summary
    df = results['accuracy']
    clean = df[df['perturbation'] == 'clean']
    print('\n=== Clean Baseline Accuracy ===')
    for ds in TIER_A:
        subset = clean[clean['dataset'] == ds]
        for _, row in subset.iterrows():
            print(f"  {row['backbone']:20s} | {ds:8s} | {row['accuracy']:.3f}")

    if len(results.get('stats', pd.DataFrame())):
        print(f"\n=== Friedman p = {results['stats'].iloc[0]['p']:.4g} ===")
        print(f"=== Kendall W = {results['stats'].iloc[1].get('W', 'N/A')} ===")
        print('\nMean ranks:')
        for bb, r in sorted(results['mean_ranks'].items(), key=lambda x: x[1]):
            print(f'  {bb:20s}: {r:.2f}')


if __name__ == '__main__':
    main()
