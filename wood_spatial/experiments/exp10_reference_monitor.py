#!/usr/bin/env python3
"""
Experiment 10: deployable reference-bank failure monitor.

This experiment avoids requiring clean-vs-shifted image pairs at deployment
time. It builds a clean reference feature bank per dataset/backbone, then scores
each shifted deployment batch by its distance to that reference bank.

The paired feature drift is still reported as an oracle/explanatory upper-bound,
but the deployable detectors are:
  - nearest/5-NN distance to the clean feature bank
  - global Mahalanobis distance to the clean feature distribution
  - batch centroid cosine distance to the clean feature centroid
  - RBF-MMD between clean and deployment batch features
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import time

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from wood_spatial.config import BB_ORDER, PERTURB_CONFIGS, TIER_A, TIER_B, V4_CSV, V4_FEAT_CACHE
from wood_spatial.core.perturbations import cache_tag_for

logger = logging.getLogger(__name__)

FAILURE_THRESHOLD = 0.20
PERTURB_TAGS = [
    cache_tag_for(pert_name, value)
    for pert_name, pcfg in PERTURB_CONFIGS.items()
    for value in pcfg['values']
]


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _load_norm_cache(backbone: str, dataset: str, tag: str):
    path = V4_FEAT_CACHE / f'{backbone}_{dataset}_{tag}.npz'
    if not path.exists():
        return None, None, None
    data = np.load(path, allow_pickle=True)
    feats, labels, paths = data['features'], data['labels'], data['paths']
    return _norm(feats.astype(np.float32)), labels, paths


def _median_rbf_gamma(x: np.ndarray, y: np.ndarray, max_samples: int = 512) -> float:
    rng = np.random.RandomState(42)
    z = np.vstack([x, y])
    if len(z) > max_samples:
        z = z[rng.choice(len(z), max_samples, replace=False)]
    # Squared Euclidean distances without materializing huge arrays.
    gram = z @ z.T
    sq = np.maximum(np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2 * gram, 0)
    vals = sq[np.triu_indices_from(sq, k=1)]
    med = float(np.median(vals[vals > 1e-12])) if np.any(vals > 1e-12) else 1.0
    return 1.0 / max(2.0 * med, 1e-12)


def _mmd_rbf(x: np.ndarray, y: np.ndarray, max_samples: int = 512) -> float:
    rng = np.random.RandomState(42)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), max_samples, replace=False)]
    gamma = _median_rbf_gamma(x, y, max_samples=max_samples)

    def kernel_mean(a, b):
        gram = a @ b.T
        aa = np.sum(a * a, axis=1)[:, None]
        bb = np.sum(b * b, axis=1)[None, :]
        sq = np.maximum(aa + bb - 2 * gram, 0)
        return float(np.exp(-gamma * sq).mean())

    return max(kernel_mean(x, x) + kernel_mean(y, y) - 2 * kernel_mean(x, y), 0.0)


def _auc_row(name: str, scores: pd.Series, y: pd.Series, drop: pd.Series):
    s = pd.to_numeric(scores, errors='coerce')
    valid = s.notna() & y.notna()
    if valid.sum() < 20 or y[valid].nunique() < 2:
        return None
    vals = s[valid].values
    yy = y[valid].values
    prec, rec, th = precision_recall_curve(yy, vals)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best = int(np.argmax(f1))
    return {
        'detector': name,
        'deployable': not name.startswith('paired_'),
        'auc_roc': float(roc_auc_score(yy, vals)),
        'avg_precision': float(average_precision_score(yy, vals)),
        'best_f1': float(f1[best]),
        'best_threshold': float(th[best]) if best < len(th) else np.nan,
        'r_vs_drop': float(np.corrcoef(vals, drop[valid].values)[0, 1]),
        'n': int(valid.sum()),
    }


def _checkpoint_dir():
    path = V4_CSV / 'exp10_checkpoints'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_path(backbone: str, dataset: str):
    return _checkpoint_dir() / f'{backbone}__{dataset}.csv'


def _read_checkpoint(backbone: str, dataset: str):
    path = _checkpoint_path(backbone, dataset)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning('Could not read checkpoint %s: %s', path, exc)
        return None
    if 'perturbation' not in df.columns:
        logger.warning('Ignoring malformed checkpoint without perturbation column: %s', path)
        return None
    if len(df) == 0:
        logger.info('Recomputing empty checkpoint %s', path)
        return None
    missing = set(PERTURB_TAGS) - set(df['perturbation'].astype(str))
    if missing:
        logger.info('Recomputing incomplete checkpoint %s (%d missing tags)', path.name, len(missing))
        return None
    return df


def _write_checkpoint(backbone: str, dataset: str, rows: list):
    path = _checkpoint_path(backbone, dataset)
    tmp = path.with_suffix('.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def _run_pair(args):
    bb, ds, failure_threshold = args
    clean, labels, _ = _load_norm_cache(bb, ds, 'original')
    if clean is None:
        return bb, ds, [], f'Missing clean cache: {bb}/{ds}'

    le = LabelEncoder()
    y_clean = le.fit_transform(labels)
    clf = KNeighborsClassifier(5, metric='cosine')
    clf.fit(clean, y_clean)
    acc_clean = float(clf.score(clean, y_clean))

    nn5 = NearestNeighbors(n_neighbors=min(5, len(clean)), metric='cosine').fit(clean)
    centroid = clean.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-8

    try:
        cov = LedoitWolf().fit(clean)
        mahal_clean = float(np.mean(cov.mahalanobis(clean)))
    except Exception as exc:
        cov = None
        mahal_clean = np.nan
        logger.warning('Mahalanobis unavailable for %s/%s: %s', bb, ds, exc)

    rows = []
    for tag in PERTURB_TAGS:
        shifted, shifted_labels, _ = _load_norm_cache(bb, ds, tag)
        if shifted is None:
            continue
        # Accuracy/drop are used only for evaluation labels, not as monitor input.
        acc_shift = float(clf.score(shifted, y_clean))
        drop = acc_clean - acc_shift

        d5, _ = nn5.kneighbors(shifted)
        d1 = d5[:, :1]
        shift_centroid = shifted.mean(axis=0)
        shift_centroid /= np.linalg.norm(shift_centroid) + 1e-8

        if cov is not None:
            mahal = float(np.mean(cov.mahalanobis(shifted)))
            mahal_delta = mahal - mahal_clean
        else:
            mahal = np.nan
            mahal_delta = np.nan

        # Oracle paired drift, valid for controlled analysis only.
        n = min(len(clean), len(shifted))
        paired_drift = float(np.mean(1 - np.sum(clean[:n] * shifted[:n], axis=1)))

        rows.append({
            'dataset': ds,
            'backbone': bb,
            'perturbation': tag,
            'acc_clean': acc_clean,
            'acc_shifted': acc_shift,
            'accuracy_drop': drop,
            'failure': int(drop > failure_threshold),
            'ref_knn1_dist': float(np.mean(d1)),
            'ref_knn5_dist': float(np.mean(d5)),
            'ref_mahalanobis': mahal,
            'ref_mahalanobis_delta': mahal_delta,
            'ref_centroid_cosine': float(1 - np.dot(centroid, shift_centroid)),
            'ref_mmd_rbf': _mmd_rbf(clean, shifted),
            'paired_feature_drift_oracle': paired_drift,
            'n_images': int(len(shifted)),
        })

    return bb, ds, rows, None


def run_exp10(
    datasets: list = None,
    backbones: list = None,
    failure_threshold: float = FAILURE_THRESHOLD,
    save: bool = True,
    jobs: int = 1,
    resume: bool = True,
    force: bool = False,
) -> dict:
    if datasets is None:
        datasets = TIER_A + TIER_B
    if backbones is None:
        backbones = BB_ORDER

    rows = []
    tasks = []
    for bb in backbones:
        for ds in datasets:
            ckpt = None if force or not resume else _read_checkpoint(bb, ds)
            if ckpt is not None:
                rows.extend(ckpt.to_dict('records'))
                logger.info('%s/%s checkpoint hit (%d tags)', bb, ds, len(ckpt))
            else:
                tasks.append((bb, ds, failure_threshold))

    jobs = max(1, int(jobs))
    if jobs == 1:
        for task in tasks:
            bb, ds, pair_rows, warning = _run_pair(task)
            if warning:
                logger.warning(warning)
            rows.extend(pair_rows)
            if save and pair_rows:
                _write_checkpoint(bb, ds, pair_rows)
            logger.info('%s/%s done (%d tags)', bb, ds, len(pair_rows))
    else:
        logger.info('Running %d backbone/dataset jobs with %d workers', len(tasks), jobs)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(_run_pair, task) for task in tasks]
            for fut in as_completed(futures):
                bb, ds, pair_rows, warning = fut.result()
                if warning:
                    logger.warning(warning)
                rows.extend(pair_rows)
                if save and pair_rows:
                    _write_checkpoint(bb, ds, pair_rows)
                logger.info('%s/%s done (%d tags)', bb, ds, len(pair_rows))

    scores = pd.DataFrame(rows)
    y = scores['failure'].astype(int) if len(scores) else pd.Series(dtype=int)
    auc_rows = []
    detectors = [
        'ref_knn1_dist',
        'ref_knn5_dist',
        'ref_mahalanobis',
        'ref_mahalanobis_delta',
        'ref_centroid_cosine',
        'ref_mmd_rbf',
        'paired_feature_drift_oracle',
    ]
    for det in detectors:
        row = _auc_row(det, scores[det], y, scores['accuracy_drop']) if len(scores) else None
        if row is not None:
            auc_rows.append(row)
    auc = pd.DataFrame(auc_rows)
    if len(auc):
        auc = auc.sort_values(['deployable', 'auc_roc'], ascending=[False, False])

    if save:
        scores.to_csv(V4_CSV / 'exp10_reference_monitor_scores.csv', index=False)
        auc.to_csv(V4_CSV / 'exp10_reference_monitor_auc.csv', index=False)

    return {'scores': scores, 'auc': auc}


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Run reference-bank deployment monitor experiment.')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--backbones', nargs='+', default=None)
    parser.add_argument('--failure-threshold', type=float, default=FAILURE_THRESHOLD)
    parser.add_argument('--jobs', type=int, default=1,
                        help='Number of parallel backbone/dataset worker processes.')
    parser.add_argument('--no-resume', action='store_true',
                        help='Ignore existing per-backbone/dataset checkpoints.')
    parser.add_argument('--force', action='store_true',
                        help='Recompute all requested backbone/dataset pairs and overwrite checkpoints.')
    args = parser.parse_args()

    t0 = time.time()
    results = run_exp10(
        args.datasets,
        args.backbones,
        args.failure_threshold,
        jobs=args.jobs,
        resume=not args.no_resume,
        force=args.force,
    )
    print('\n=== Reference-Bank Monitor AUC ===')
    print(results['auc'].to_string(index=False))
    print(f'\nRows: {len(results["scores"])}')
    print(f'Done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
