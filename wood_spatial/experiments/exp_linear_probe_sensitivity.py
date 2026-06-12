#!/usr/bin/env python3
"""
Frozen linear-probe decision-rule sensitivity.

This experiment addresses the circularity concern for the drift/drop result by
evaluating a non-kNN decision rule on the same frozen Tier-A features. For each
stratified split, it trains a multinomial logistic-regression probe on clean
training-split features and evaluates it on perturbed test-split features. Thus,
the clean counterpart of a perturbed query is not in the probe training split.
The resulting accuracy drop is correlated with paired feature drift from
exp1b_feature_geometry.csv.

Usage:
    python -m wood_spatial.experiments.exp_linear_probe_sensitivity --jobs 2
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from wood_spatial.config import BB_ORDER, PERTURB_CONFIGS, TIER_A, V4_CSV, V4_FEAT_CACHE
from wood_spatial.core.perturbations import cache_tag_for

logger = logging.getLogger(__name__)

PROTOCOL = 'split_disjoint_v1'
N_SPLITS = 5
TEST_SIZE = 0.2
SPLIT_SEED = 42


def _load_cache_np(backbone: str, dataset: str, tag: str):
    path = V4_FEAT_CACHE / f'{backbone}_{dataset}_{tag}.npz'
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    feats = data['features'].astype(np.float32)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return feats, data['labels'], data['paths']


def _fit_probe(x_train, y_train):
    clf = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver='lbfgs',
        multi_class='auto',
        n_jobs=1,
    )
    clf.fit(x_train, y_train)
    return clf


def _make_split_probes(features, labels, seed: int = SPLIT_SEED):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    splitter = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=seed,
    )
    probes = []
    clean_scores = []
    for train_idx, test_idx in splitter.split(features, y):
        clf = _fit_probe(features[train_idx], y[train_idx])
        pred = clf.predict(features[test_idx])
        clean_scores.append(accuracy_score(y[test_idx], pred))
        probes.append((train_idx, test_idx, clf))
    return le, y, probes, clean_scores


def _linear_probe_split_accuracy(probes, y_query, query_features):
    scores = []
    for _train_idx, test_idx, clf in probes:
        pred = clf.predict(query_features[test_idx])
        scores.append(accuracy_score(y_query[test_idx], pred))
    return float(np.mean(scores)), float(np.std(scores))


def _checkpoint_dir():
    path = V4_CSV / 'exp_linear_probe_checkpoints'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_path(dataset: str, backbone: str):
    return _checkpoint_dir() / f'{dataset}__{backbone}.csv'


def _expected_tags():
    tags = {
        cache_tag_for(pert_name, value)
        for pert_name, pcfg in PERTURB_CONFIGS.items()
        for value in pcfg['values']
    }
    tags.add('original')
    return tags


def _read_checkpoint(dataset: str, backbone: str):
    path = _checkpoint_path(dataset, backbone)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning('Could not read checkpoint %s: %s', path, exc)
        return None
    required = {'dataset', 'backbone', 'perturbation', 'severity', 'tag', 'accuracy', 'drop'}
    if not required.issubset(df.columns) or df.empty:
        return None
    if 'protocol' not in df.columns or set(df['protocol'].astype(str)) != {PROTOCOL}:
        return None
    if not _expected_tags().issubset(set(df['tag'].astype(str))):
        return None
    return df


def _write_checkpoint(dataset: str, backbone: str, rows: list):
    path = _checkpoint_path(dataset, backbone)
    tmp = path.with_suffix('.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def _run_pair(task):
    ds, bb = task
    rows = []
    try:
        clean, labels, clean_paths = _load_cache_np(bb, ds, 'original')
    except FileNotFoundError:
        return ds, bb, rows, f'Missing clean cache: {bb}/{ds}'

    le, y_clean, probes, clean_scores = _make_split_probes(clean, labels)
    clean_acc = float(np.mean(clean_scores))
    clean_std = float(np.std(clean_scores))
    rows.append({
        'dataset': ds,
        'backbone': bb,
        'perturbation': 'clean',
        'severity': 0,
        'tag': 'original',
        'rule': 'linear_probe',
        'protocol': PROTOCOL,
        'n_splits': N_SPLITS,
        'test_size': TEST_SIZE,
        'split_seed': SPLIT_SEED,
        'accuracy': clean_acc,
        'accuracy_std': clean_std,
        'acc_clean': clean_acc,
        'drop': 0.0,
    })

    for pert_name, pcfg in PERTURB_CONFIGS.items():
        for value in pcfg['values']:
            tag = cache_tag_for(pert_name, value)
            try:
                shifted, shifted_labels, shifted_paths = _load_cache_np(bb, ds, tag)
            except FileNotFoundError:
                continue
            if len(shifted) != len(clean):
                raise RuntimeError(f'Cache length mismatch for {bb}/{ds}/{tag}')
            if not np.array_equal(labels, shifted_labels):
                raise RuntimeError(f'Label order mismatch for {bb}/{ds}/{tag}')
            if not np.array_equal(clean_paths, shifted_paths):
                logger.warning('Path order differs for %s/%s/%s; using aligned indices by label order',
                               bb, ds, tag)
            y_shifted = le.transform(shifted_labels)
            acc, acc_std = _linear_probe_split_accuracy(probes, y_shifted, shifted)
            rows.append({
                'dataset': ds,
                'backbone': bb,
                'perturbation': pert_name,
                'severity': value,
                'tag': tag,
                'rule': 'linear_probe',
                'protocol': PROTOCOL,
                'n_splits': N_SPLITS,
                'test_size': TEST_SIZE,
                'split_seed': SPLIT_SEED,
                'accuracy': acc,
                'accuracy_std': acc_std,
                'acc_clean': clean_acc,
                'drop': clean_acc - acc,
            })
    return ds, bb, rows, None


def _summarize(acc: pd.DataFrame) -> pd.DataFrame:
    geom = pd.read_csv(V4_CSV / 'exp1b_feature_geometry.csv')
    geom_key = geom[
        ['dataset', 'backbone', 'perturbation', 'severity', 'tag', 'feature_drift']
    ]
    pert = acc[acc['perturbation'] != 'clean'].merge(
        geom_key,
        on=['dataset', 'backbone', 'perturbation', 'severity', 'tag'],
        how='left',
    )
    sub = pert.dropna(subset=['feature_drift', 'drop'])
    r, p = pearsonr(sub['feature_drift'], sub['drop'])
    rho, sp = spearmanr(sub['feature_drift'], sub['drop'])
    clean = acc[acc['perturbation'] == 'clean']
    return pd.DataFrame([{
        'rule': 'linear_probe',
        'protocol': PROTOCOL,
        'n_splits': N_SPLITS,
        'test_size': TEST_SIZE,
        'split_seed': SPLIT_SEED,
        'mean_clean_acc': float(clean['accuracy'].mean()),
        'mean_pert_acc': float(pert['accuracy'].mean()),
        'mean_drop': float(pert['drop'].mean()),
        'feature_drop_pearson_r': float(r),
        'feature_drop_pearson_p': float(p),
        'feature_drop_spearman_rho': float(rho),
        'feature_drop_spearman_p': float(sp),
        'n': int(len(sub)),
    }])


def run(
    save: bool = True,
    datasets: list = None,
    backbones: list = None,
    jobs: int = 1,
    resume: bool = True,
    force: bool = False,
    finalize_only: bool = False,
) -> dict:
    datasets = TIER_A if datasets is None else datasets
    backbones = BB_ORDER if backbones is None else backbones

    rows = []
    tasks = []
    if not finalize_only:
        for ds in datasets:
            for bb in backbones:
                ckpt = None if force or not resume else _read_checkpoint(ds, bb)
                if ckpt is not None:
                    rows.extend(ckpt.to_dict('records'))
                    logger.info('%s/%s checkpoint hit (%d rows)', bb, ds, len(ckpt))
                else:
                    tasks.append((ds, bb))

        jobs = max(1, int(jobs))
        if jobs == 1:
            for task in tasks:
                ds, bb, pair_rows, warning = _run_pair(task)
                if warning:
                    logger.warning(warning)
                rows.extend(pair_rows)
                if save and pair_rows:
                    _write_checkpoint(ds, bb, pair_rows)
                logger.info('%s/%s done (%d rows)', bb, ds, len(pair_rows))
        else:
            logger.info('Running %d dataset/backbone jobs with %d workers', len(tasks), jobs)
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futures = [ex.submit(_run_pair, task) for task in tasks]
                for fut in as_completed(futures):
                    ds, bb, pair_rows, warning = fut.result()
                    if warning:
                        logger.warning(warning)
                    rows.extend(pair_rows)
                    if save and pair_rows:
                        _write_checkpoint(ds, bb, pair_rows)
                    logger.info('%s/%s done (%d rows)', bb, ds, len(pair_rows))

    # Rebuild the final table from checkpoints when possible. This avoids
    # appending checkpoint hits a second time after they were loaded above.
    ckpt_rows = []
    for ds in datasets:
        for bb in backbones:
            ckpt = _read_checkpoint(ds, bb)
            if ckpt is not None:
                ckpt_rows.extend(ckpt.to_dict('records'))
    if ckpt_rows:
        rows = ckpt_rows

    acc = pd.DataFrame(rows).drop_duplicates(
        subset=['dataset', 'backbone', 'perturbation', 'severity', 'tag', 'rule'],
        keep='last',
    )
    if acc.empty:
        raise RuntimeError(f'No feature caches/checkpoints found in {V4_FEAT_CACHE}.')

    summary = _summarize(acc)
    if save:
        acc.to_csv(V4_CSV / 'exp_linear_probe_accuracy.csv', index=False)
        summary.to_csv(V4_CSV / 'exp_linear_probe_summary.csv', index=False)
    return {'accuracy': acc, 'summary': summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run frozen linear-probe sensitivity.')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--datasets', nargs='*', default=None)
    parser.add_argument('--backbones', nargs='*', default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--finalize-only', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    out = run(
        save=True,
        datasets=args.datasets,
        backbones=args.backbones,
        jobs=args.jobs,
        resume=not args.no_resume,
        force=args.force,
        finalize_only=args.finalize_only,
    )
    print(out['summary'].to_string(index=False))
