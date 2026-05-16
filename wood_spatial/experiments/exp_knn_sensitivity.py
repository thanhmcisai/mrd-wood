#!/usr/bin/env python3
"""
Experiment: frozen-feature decision-rule sensitivity.

This experiment checks whether the main robustness conclusions depend on the
choice of cosine 5NN. It recomputes Tier-A clean/perturbed accuracy with
k in {1, 3, 5, 10} and a nearest-class-centroid classifier.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from wood_spatial.config import BB_ORDER, PERTURB_CONFIGS, TIER_A, V4_CSV, V4_FEAT_CACHE
from wood_spatial.core.perturbations import cache_tag_for

logger = logging.getLogger(__name__)

K_VALUES = [1, 3, 5, 10]


def _load_cache_np(backbone: str, dataset: str, tag: str):
    path = V4_FEAT_CACHE / f'{backbone}_{dataset}_{tag}.npz'
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    feats = data['features'].astype(np.float32)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return feats, data['labels'], data['paths']


def _encode(train_labels, test_labels=None):
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    if test_labels is None:
        return y_train, None
    return y_train, le.transform(test_labels)


def _knn_cv_accuracy(features, labels, k: int, seed: int = 42):
    labels_enc, _ = _encode(labels)
    splitter = StratifiedShuffleSplit(
        n_splits=5, test_size=0.2, random_state=seed,
    )
    scores = []
    for train_idx, test_idx in splitter.split(features, labels_enc):
        n_neighbors = min(k, len(train_idx))
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        clf.fit(features[train_idx], labels_enc[train_idx])
        scores.append(clf.score(features[test_idx], labels_enc[test_idx]))
    return float(np.mean(scores)), float(np.std(scores))


def _knn_gallery_accuracy(clean, clean_labels, query, query_labels, k: int):
    y_clean, y_query = _encode(clean_labels, query_labels)
    clf = KNeighborsClassifier(n_neighbors=min(k, len(clean)), metric='cosine')
    clf.fit(clean, y_clean)
    return float(clf.score(query, y_query))


def _centroids(features, labels):
    labels_enc, _ = _encode(labels)
    classes = np.unique(labels_enc)
    cents = []
    for c in classes:
        v = features[labels_enc == c].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-8)
        cents.append(v)
    return np.vstack(cents), classes


def _ncc_cv_accuracy(features, labels, seed: int = 42):
    labels_enc, _ = _encode(labels)
    splitter = StratifiedShuffleSplit(
        n_splits=5, test_size=0.2, random_state=seed,
    )
    scores = []
    for train_idx, test_idx in splitter.split(features, labels_enc):
        cents, classes = _centroids(features[train_idx], labels[train_idx])
        sim = features[test_idx] @ cents.T
        pred = classes[np.argmax(sim, axis=1)]
        scores.append(accuracy_score(labels_enc[test_idx], pred))
    return float(np.mean(scores)), float(np.std(scores))


def _ncc_gallery_accuracy(clean, clean_labels, query, query_labels):
    _, y_query = _encode(clean_labels, query_labels)
    cents, classes = _centroids(clean, clean_labels)
    sim = query @ cents.T
    pred = classes[np.argmax(sim, axis=1)]
    return float(accuracy_score(y_query, pred))


def _iter_rules():
    for k in K_VALUES:
        yield f'knn{k}', k
    yield 'nearest_centroid', None


def _checkpoint_dir():
    path = V4_CSV / 'exp_knn_sensitivity_checkpoints'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_path(dataset: str, backbone: str):
    return _checkpoint_dir() / f'{dataset}__{backbone}.csv'


def _read_checkpoint(dataset: str, backbone: str):
    path = _checkpoint_path(dataset, backbone)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning('Could not read checkpoint %s: %s', path, exc)
        return None
    required = {'dataset', 'backbone', 'rule', 'accuracy', 'drop'}
    if not required.issubset(df.columns) or len(df) == 0:
        logger.warning('Ignoring malformed/empty checkpoint: %s', path)
        return None
    expected_tags = {
        cache_tag_for(pert_name, value)
        for pert_name, pcfg in PERTURB_CONFIGS.items()
        for value in pcfg['values']
    }
    expected_tags.add('original')
    observed_tags = set(df['tag'].astype(str)) if 'tag' in df.columns else set()
    if not expected_tags.issubset(observed_tags):
        logger.info('Recomputing incomplete checkpoint %s (%d missing tags)',
                    path.name, len(expected_tags - observed_tags))
        return None
    return df


def _write_checkpoint(dataset: str, backbone: str, rows: list):
    path = _checkpoint_path(dataset, backbone)
    tmp = path.with_suffix('.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def _knn_gallery_all_k(clean, clean_labels, query, query_labels):
    y_clean, y_query = _encode(clean_labels, query_labels)
    kmax = min(max(K_VALUES), len(clean))
    clf = KNeighborsClassifier(n_neighbors=kmax, metric='cosine')
    clf.fit(clean, y_clean)
    neigh = clf.kneighbors(query, return_distance=False)
    out = {}
    n_classes = int(np.max(y_clean)) + 1
    for k in K_VALUES:
        kk = min(k, kmax)
        pred = []
        for row in neigh[:, :kk]:
            counts = np.bincount(y_clean[row], minlength=n_classes)
            pred.append(int(np.argmax(counts)))
        out[f'knn{k}'] = float(accuracy_score(y_query, pred))
    return out


def _run_pair(args):
    ds, bb = args
    rows = []
    try:
        clean, labels, _ = _load_cache_np(bb, ds, 'original')
    except FileNotFoundError:
        return ds, bb, [], f'Missing clean cache: {bb}/{ds}'

    clean_acc = {}
    for rule, k in _iter_rules():
        if rule == 'nearest_centroid':
            acc, std = _ncc_cv_accuracy(clean, labels)
        else:
            acc, std = _knn_cv_accuracy(clean, labels, k)
        clean_acc[rule] = acc
        rows.append({
            'dataset': ds, 'backbone': bb, 'perturbation': 'clean',
            'severity': 0, 'tag': 'original', 'rule': rule,
            'accuracy': acc, 'accuracy_std': std,
            'acc_clean': acc, 'drop': 0.0,
        })

    for pert_name, pcfg in PERTURB_CONFIGS.items():
        for value in pcfg['values']:
            tag = cache_tag_for(pert_name, value)
            try:
                shifted, shifted_labels, _ = _load_cache_np(bb, ds, tag)
            except FileNotFoundError:
                continue

            knn_acc = _knn_gallery_all_k(clean, labels, shifted, shifted_labels)
            knn_acc['nearest_centroid'] = _ncc_gallery_accuracy(
                clean, labels, shifted, shifted_labels,
            )
            for rule, acc in knn_acc.items():
                rows.append({
                    'dataset': ds, 'backbone': bb,
                    'perturbation': pert_name, 'severity': value,
                    'tag': tag, 'rule': rule,
                    'accuracy': acc, 'accuracy_std': 0.0,
                    'acc_clean': clean_acc[rule],
                    'drop': clean_acc[rule] - acc,
                })

    return ds, bb, rows, None


def _summarize(acc: pd.DataFrame) -> pd.DataFrame:
    if acc.empty:
        raise RuntimeError(
            f'No feature caches found in {V4_FEAT_CACHE}. '
            'Run this experiment on the Colab/results directory that contains feature_cache/*.npz.'
        )
    geom = pd.read_csv(V4_CSV / 'exp1b_feature_geometry.csv')
    geom_key = geom[['dataset', 'backbone', 'perturbation', 'severity', 'tag', 'feature_drift']]

    pert = acc[acc['perturbation'] != 'clean'].merge(
        geom_key,
        on=['dataset', 'backbone', 'perturbation', 'severity', 'tag'],
        how='left',
    )

    summary_rows = []
    for rule, sub in pert.groupby('rule'):
        clean_summary = acc[(acc['rule'] == rule) & (acc['perturbation'] == 'clean')]
        sub_fd = sub.dropna(subset=['feature_drift', 'drop'])
        r, p = pearsonr(sub_fd['feature_drift'], sub_fd['drop'])
        rho, sp = spearmanr(sub_fd['feature_drift'], sub_fd['drop'])
        by_bb = (
            sub.groupby('backbone', as_index=False)
            .agg(clean_acc=('acc_clean', 'mean'), pert_acc=('accuracy', 'mean'), drop=('drop', 'mean'))
            .sort_values('pert_acc', ascending=False)
        )
        ranking = ' > '.join(by_bb['backbone'].tolist())
        summary_rows.append({
            'rule': rule,
            'mean_clean_acc': float(clean_summary['accuracy'].mean()),
            'mean_pert_acc': float(sub['accuracy'].mean()),
            'mean_drop': float(sub['drop'].mean()),
            'feature_drop_pearson_r': float(r),
            'feature_drop_pearson_p': float(p),
            'feature_drop_spearman_rho': float(rho),
            'feature_drop_spearman_p': float(sp),
            'n': int(len(sub_fd)),
            'backbone_ranking_by_pert_acc': ranking,
        })

    summary = pd.DataFrame(summary_rows)
    return summary


def _load_all_checkpoints(datasets: list, backbones: list) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        for bb in backbones:
            ckpt = _read_checkpoint(ds, bb)
            if ckpt is not None:
                rows.extend(ckpt.to_dict('records'))
    return pd.DataFrame(rows)


def run(
    save: bool = True,
    datasets: list = None,
    backbones: list = None,
    jobs: int = 1,
    resume: bool = True,
    force: bool = False,
    finalize_only: bool = False,
) -> dict:
    if datasets is None:
        datasets = TIER_A
    if backbones is None:
        backbones = BB_ORDER

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

    acc = pd.DataFrame(rows)
    ckpt_acc = _load_all_checkpoints(datasets, backbones)
    if not ckpt_acc.empty:
        acc = pd.concat([ckpt_acc, acc], ignore_index=True)
        acc = acc.drop_duplicates(
            subset=['dataset', 'backbone', 'perturbation', 'severity', 'tag', 'rule'],
            keep='last',
        )
    if acc.empty:
        raise RuntimeError(
            f'No feature caches/checkpoints found in {V4_FEAT_CACHE} or {_checkpoint_dir()}.'
        )

    summary = _summarize(acc)

    if save:
        acc.to_csv(V4_CSV / 'exp_knn_sensitivity_accuracy.csv', index=False)
        summary.to_csv(V4_CSV / 'exp_knn_sensitivity_summary.csv', index=False)

    return {'accuracy': acc, 'summary': summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run frozen-feature kNN/centroid sensitivity.')
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
