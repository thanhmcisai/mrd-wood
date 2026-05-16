#!/usr/bin/env python3
"""
Experiment 10c: reference-bank monitor sensitivity.

This experiment stress-tests Exp10 by varying incoming batch size and clean
reference-bank size. It reads global feature caches only; no GPU is required.
The monitor scores are label-free, while labels are used only to define
retrospective batch-level failure for evaluation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from wood_spatial.config import BB_ORDER, PERTURB_CONFIGS, TIER_A, TIER_B, V4_CSV, V4_FEAT_CACHE
from wood_spatial.core.perturbations import cache_tag_for
from wood_spatial.experiments.exp10_reference_monitor import _mmd_rbf, _norm
from wood_spatial.experiments.exp10_lopo_monitor import perturbation_family

logger = logging.getLogger(__name__)

FAILURE_THRESHOLD = 0.20
DEFAULT_BATCH_SIZES = [8, 16, 32, 64]
DEFAULT_REF_SIZES = [100, 250, 500, 1000]

PERTURB_TAGS = [
    cache_tag_for(pert_name, value)
    for pert_name, pcfg in PERTURB_CONFIGS.items()
    for value in pcfg['values']
]


def _load_norm_cache(backbone: str, dataset: str, tag: str):
    path = V4_FEAT_CACHE / f'{backbone}_{dataset}_{tag}.npz'
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=True)
    feats = _norm(data['features'].astype(np.float32))
    labels = data['labels']
    return feats, labels


def _parse_int_list(text, default):
    if text is None:
        return list(default)
    return [int(x.strip()) for x in str(text).split(',') if x.strip()]


def _checkpoint_dir():
    path = V4_CSV / 'exp10_sensitivity_checkpoints'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_path(bb, ds, batch_sizes, ref_sizes, repeats):
    bs = '-'.join(str(x) for x in batch_sizes)
    rs = '-'.join(str(x) for x in ref_sizes)
    return _checkpoint_dir() / f'{bb}__{ds}__bs{bs}__rs{rs}__r{repeats}.csv'


def _run_pair(args):
    bb, ds, batch_sizes, ref_sizes, repeats, failure_threshold, families = args
    clean, labels = _load_norm_cache(bb, ds, 'original')
    if clean is None:
        return bb, ds, [], f'Missing clean cache: {bb}/{ds}'

    le = LabelEncoder()
    y_clean = le.fit_transform(labels)
    clf = KNeighborsClassifier(5, metric='cosine')
    clf.fit(clean, y_clean)
    acc_clean_full = float(clf.score(clean, y_clean))

    seed_bytes = hashlib.md5(f'{bb}/{ds}'.encode('utf-8')).hexdigest()[:8]
    rng_base = np.random.RandomState(int(seed_bytes, 16))
    rows = []

    for tag in PERTURB_TAGS:
        family = perturbation_family(tag)
        if families and family not in families:
            continue
        shifted, shifted_labels = _load_norm_cache(bb, ds, tag)
        if shifted is None:
            continue
        try:
            y_shifted = le.transform(shifted_labels)
        except ValueError:
            # If labels are already encoded differently, fall back to clean order
            # only when lengths match, matching the main Exp10 protocol.
            if len(shifted) != len(y_clean):
                continue
            y_shifted = y_clean

        for batch_size in batch_sizes:
            n_batch = min(int(batch_size), len(shifted))
            if n_batch < 2:
                continue
            for ref_size in ref_sizes:
                n_ref = min(int(ref_size), len(clean))
                if n_ref < 2:
                    continue
                for rep in range(int(repeats)):
                    seed = int(rng_base.randint(0, 2**31 - 1))
                    rng = np.random.RandomState(seed + rep)
                    bidx = rng.choice(len(shifted), n_batch, replace=False)
                    ridx = rng.choice(len(clean), n_ref, replace=False)
                    batch = shifted[bidx]
                    ref = clean[ridx]

                    acc_batch = float(clf.score(batch, y_shifted[bidx]))
                    drop = acc_clean_full - acc_batch

                    nn = NearestNeighbors(n_neighbors=min(5, n_ref), metric='cosine').fit(ref)
                    d5, _ = nn.kneighbors(batch)

                    ref_centroid = ref.mean(axis=0)
                    ref_centroid /= np.linalg.norm(ref_centroid) + 1e-8
                    batch_centroid = batch.mean(axis=0)
                    batch_centroid /= np.linalg.norm(batch_centroid) + 1e-8

                    rows.append({
                        'dataset': ds,
                        'backbone': bb,
                        'perturbation': tag,
                        'family': family,
                        'batch_size': int(n_batch),
                        'reference_size': int(n_ref),
                        'repeat': int(rep),
                        'acc_clean_full': acc_clean_full,
                        'acc_batch': acc_batch,
                        'accuracy_drop': drop,
                        'failure': int(drop > failure_threshold),
                        'ref_knn1_dist': float(np.mean(d5[:, :1])),
                        'ref_knn5_dist': float(np.mean(d5)),
                        'ref_centroid_cosine': float(1 - np.dot(ref_centroid, batch_centroid)),
                        'ref_mmd_rbf': _mmd_rbf(ref, batch, max_samples=512),
                        'seed': seed,
                    })
    return bb, ds, rows, None


def _auc_rows(df):
    rows = []
    detectors = ['ref_mmd_rbf', 'ref_centroid_cosine', 'ref_knn1_dist', 'ref_knn5_dist']
    for (bs, rs), grp in df.groupby(['batch_size', 'reference_size']):
        y = grp['failure'].astype(int)
        for det in detectors:
            s = pd.to_numeric(grp[det], errors='coerce')
            valid = s.notna() & y.notna()
            if valid.sum() < 20 or y[valid].nunique() < 2:
                continue
            vals = s[valid].values
            yy = y[valid].values
            prec, rec, th = precision_recall_curve(yy, vals)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            best = int(np.argmax(f1))
            rows.append({
                'batch_size': int(bs),
                'reference_size': int(rs),
                'detector': det,
                'auc_roc': float(roc_auc_score(yy, vals)),
                'avg_precision': float(average_precision_score(yy, vals)),
                'best_f1': float(f1[best]),
                'best_threshold': float(th[best]) if best < len(th) else np.nan,
                'failure_rate': float(np.mean(yy)),
                'n': int(valid.sum()),
            })
    return pd.DataFrame(rows)


def run_exp10_sensitivity(
    datasets=None,
    backbones=None,
    batch_sizes=None,
    ref_sizes=None,
    repeats=5,
    failure_threshold=FAILURE_THRESHOLD,
    families=None,
    jobs=1,
    resume=True,
    force=False,
    save=True,
):
    if datasets is None:
        datasets = TIER_A + TIER_B
    if backbones is None:
        backbones = BB_ORDER
    batch_sizes = list(batch_sizes or DEFAULT_BATCH_SIZES)
    ref_sizes = list(ref_sizes or DEFAULT_REF_SIZES)
    family_set = set(families) if families else None

    rows = []
    tasks = []
    for bb in backbones:
        for ds in datasets:
            ckpt = _checkpoint_path(bb, ds, batch_sizes, ref_sizes, repeats)
            if resume and not force and ckpt.exists():
                try:
                    part = pd.read_csv(ckpt)
                    expected = [
                        tag for tag in PERTURB_TAGS
                        if family_set is None or perturbation_family(tag) in family_set
                    ]
                    observed = set(part['perturbation'].astype(str)) if 'perturbation' in part.columns else set()
                    if len(part) and set(expected).issubset(observed):
                        rows.extend(part.to_dict('records'))
                        logger.info('%s/%s checkpoint hit (%d rows)', bb, ds, len(part))
                        continue
                    logger.info('%s/%s sensitivity checkpoint incomplete; recomputing', bb, ds)
                except Exception as exc:
                    logger.warning('Ignoring bad checkpoint %s: %s', ckpt, exc)
            tasks.append((bb, ds, batch_sizes, ref_sizes, repeats, failure_threshold, family_set))

    jobs = max(1, int(jobs))
    if jobs == 1:
        for task in tasks:
            bb, ds, part_rows, warning = _run_pair(task)
            if warning:
                logger.warning(warning)
            rows.extend(part_rows)
            if save and part_rows:
                pd.DataFrame(part_rows).to_csv(
                    _checkpoint_path(bb, ds, batch_sizes, ref_sizes, repeats), index=False)
            logger.info('%s/%s done (%d rows)', bb, ds, len(part_rows))
    else:
        logger.info('Running %d sensitivity jobs with %d workers', len(tasks), jobs)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(_run_pair, task) for task in tasks]
            for fut in as_completed(futures):
                bb, ds, part_rows, warning = fut.result()
                if warning:
                    logger.warning(warning)
                rows.extend(part_rows)
                if save and part_rows:
                    pd.DataFrame(part_rows).to_csv(
                        _checkpoint_path(bb, ds, batch_sizes, ref_sizes, repeats), index=False)
                logger.info('%s/%s done (%d rows)', bb, ds, len(part_rows))

    scores = pd.DataFrame(rows)
    auc = _auc_rows(scores) if len(scores) else pd.DataFrame()
    if save:
        scores.to_csv(V4_CSV / 'exp10_monitor_sensitivity_scores.csv', index=False)
        auc.to_csv(V4_CSV / 'exp10_monitor_sensitivity_auc.csv', index=False)
    return {'scores': scores, 'auc': auc}


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Run Exp10 monitor batch/reference sensitivity.')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--backbones', nargs='+', default=None)
    parser.add_argument('--batch-sizes', default=None,
                        help='Comma-separated list, e.g. 8,16,32,64')
    parser.add_argument('--ref-sizes', default=None,
                        help='Comma-separated list, e.g. 100,250,500,1000')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--failure-threshold', type=float, default=FAILURE_THRESHOLD)
    parser.add_argument('--families', nargs='+', default=None)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    t0 = time.time()
    res = run_exp10_sensitivity(
        datasets=args.datasets,
        backbones=args.backbones,
        batch_sizes=_parse_int_list(args.batch_sizes, DEFAULT_BATCH_SIZES),
        ref_sizes=_parse_int_list(args.ref_sizes, DEFAULT_REF_SIZES),
        repeats=args.repeats,
        failure_threshold=args.failure_threshold,
        families=args.families,
        jobs=args.jobs,
        resume=not args.no_resume,
        force=args.force,
    )
    print('\n=== Monitor Sensitivity AUC ===')
    if len(res['auc']):
        print(res['auc'].sort_values(['batch_size', 'reference_size', 'auc_roc'],
                                     ascending=[True, True, False]).to_string(index=False))
    print(f'\nRows: {len(res["scores"])}')
    print(f'Done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
