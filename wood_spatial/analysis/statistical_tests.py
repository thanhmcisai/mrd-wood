"""
Wood Spatial — Statistical Tests & Metric Wrappers
=====================================================
Reusable statistical test functions for the experiment scripts.

Wraps or re-exports proven functions from wood_benchmark.core.metrics
and adds Friedman/Nemenyi/Wilcoxon tests for the new experiments.
"""
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats
from scipy.stats import pearsonr, friedmanchisquare, wilcoxon

from wood_spatial.config import (
    RANDOM_SEED, BOOTSTRAP_B, BOOTSTRAP_ALPHA, KNN_K, BB_ORDER,
)

logger = logging.getLogger(__name__)


# ── kNN accuracy (adapted from v2, gallery-query protocol) ────────────────

def knn_accuracy_gallery_query(
    feat_gallery: np.ndarray,
    labels_gallery: np.ndarray,
    feat_query: np.ndarray,
    labels_query: np.ndarray,
    k: int = KNN_K,
) -> float:
    """
    Gallery-query kNN accuracy: train on gallery, predict query.

    Used for perturbation robustness: gallery=clean, query=perturbed.

    Returns
    -------
    float : accuracy in [0, 1]
    """
    from sklearn.neighbors import KNeighborsClassifier

    k_use = max(1, min(k, len(feat_gallery) - 1))
    knn = KNeighborsClassifier(n_neighbors=k_use, metric='cosine', algorithm='brute')
    knn.fit(feat_gallery, labels_gallery)
    return float(knn.score(feat_query, labels_query))


def knn_accuracy_cv(
    feat: np.ndarray,
    labels: np.ndarray,
    k: int = KNN_K,
    n_splits: int = 5,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    """
    Cross-validated kNN accuracy with class-sparsity handling.

    Returns (mean_acc, std_acc). NaN if insufficient data.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedShuffleSplit

    counts = np.bincount(labels)
    valid = np.where(counts >= 2)[0]
    if len(valid) < 2:
        return np.nan, np.nan
    mask = np.isin(labels, valid)
    fv, lv = feat[mask], labels[mask]
    k_use = max(1, min(k, int(counts[valid].min()) - 1, len(valid) - 1))
    if int(len(lv) * test_size) < k_use + 1:
        return np.nan, np.nan
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    accs = []
    try:
        for tr, te in sss.split(fv, lv):
            knn = KNeighborsClassifier(n_neighbors=k_use, metric='cosine', algorithm='brute')
            knn.fit(fv[tr], lv[tr])
            accs.append(knn.score(fv[te], lv[te]))
    except Exception:
        return np.nan, np.nan
    return float(np.mean(accs)), float(np.std(accs))


# ── Bootstrap CI ──────────────────────────────────────────────────────────

def bootstrap_ci(
    feat: np.ndarray,
    labels: np.ndarray,
    B: int = BOOTSTRAP_B,
    seed: int = RANDOM_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
) -> Tuple[float, float, float, float]:
    """
    Bootstrap 95% CI for kNN accuracy.

    Fix train set (80/20 split), resample test predictions B times.

    Returns (point, ci_lo, ci_hi, std).
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedShuffleSplit

    rng = np.random.default_rng(seed)

    counts = np.bincount(labels)
    valid = np.where(counts >= 2)[0]
    if len(valid) < 2:
        return np.nan, np.nan, np.nan, np.nan
    mask = np.isin(labels, valid)
    fv, lv = feat[mask], labels[mask]
    k_use = max(1, min(KNN_K, int(counts[valid].min()) - 1, len(valid) - 1))

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(sss.split(fv, lv))
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

    knn = KNeighborsClassifier(n_neighbors=k_use, metric='cosine', algorithm='brute')
    knn.fit(fv[tr], lv[tr])
    point = float(knn.score(fv[te], lv[te]))
    preds = knn.predict(fv[te])

    n_te = len(te)
    boot_accs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n_te, size=n_te)
        boot_accs[i] = float(np.mean(preds[idx] == lv[te][idx]))

    lo = float(np.percentile(boot_accs, 100 * alpha / 2))
    hi = float(np.percentile(boot_accs, 100 * (1 - alpha / 2)))
    return point, lo, hi, float(boot_accs.std())


def bootstrap_gallery_query_ci(
    feat_gallery: np.ndarray,
    labels_gallery: np.ndarray,
    feat_query: np.ndarray,
    labels_query: np.ndarray,
    k: int = KNN_K,
    B: int = BOOTSTRAP_B,
    seed: int = RANDOM_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
) -> Tuple[float, float, float, float]:
    """
    Bootstrap CI for gallery-query kNN. Resamples query predictions.

    Returns (point, ci_lo, ci_hi, std).
    """
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(seed)
    k_use = max(1, min(k, len(feat_gallery) - 1))
    knn = KNeighborsClassifier(n_neighbors=k_use, metric='cosine', algorithm='brute')
    knn.fit(feat_gallery, labels_gallery)

    point = float(knn.score(feat_query, labels_query))
    preds = knn.predict(feat_query)
    n = len(labels_query)

    boot_accs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        boot_accs[i] = np.mean(preds[idx] == labels_query[idx])

    lo = float(np.percentile(boot_accs, 100 * alpha / 2))
    hi = float(np.percentile(boot_accs, 100 * (1 - alpha / 2)))
    return point, lo, hi, float(boot_accs.std())


# ── Friedman Test ─────────────────────────────────────────────────────────

def friedman_test(
    accuracy_matrix: np.ndarray,
) -> Tuple[float, float]:
    """
    Friedman test: non-parametric repeated-measures ANOVA.

    Tests whether backbone rankings differ significantly across conditions.

    Parameters
    ----------
    accuracy_matrix : (n_conditions, n_backbones) float
        Each row is one condition (dataset × perturbation), columns are backbones.

    Returns
    -------
    (chi2, p_value)
    """
    n_cond, n_bb = accuracy_matrix.shape
    if n_cond < 2 or n_bb < 3:
        return np.nan, np.nan
    # scipy wants each group as a separate array
    groups = [accuracy_matrix[:, i] for i in range(n_bb)]
    try:
        chi2, p = friedmanchisquare(*groups)
        return float(chi2), float(p)
    except Exception:
        return np.nan, np.nan


# ── Nemenyi Post-Hoc ─────────────────────────────────────────────────────

def nemenyi_posthoc(
    accuracy_matrix: np.ndarray,
    backbone_ids: List[str] = None,
) -> Optional[object]:
    """
    Nemenyi post-hoc test after Friedman.

    Requires scikit-posthocs: pip install scikit-posthocs

    Parameters
    ----------
    accuracy_matrix : (n_conditions, n_backbones) float
    backbone_ids    : list of backbone names for index/columns

    Returns
    -------
    pandas DataFrame of p-values, or None if test fails
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        logger.warning('scikit-posthocs not installed. Run: pip install scikit-posthocs')
        return None

    if backbone_ids is None:
        backbone_ids = BB_ORDER[:accuracy_matrix.shape[1]]

    try:
        result = sp.posthoc_nemenyi_friedman(accuracy_matrix)
        result.index = backbone_ids
        result.columns = backbone_ids
        return result
    except Exception as e:
        logger.warning('Nemenyi test failed: %s', e)
        return None


def nemenyi_cd(
    accuracy_matrix: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Critical Difference for Nemenyi test.

    CD = q_alpha * sqrt(k*(k+1) / (6*n))

    Parameters
    ----------
    accuracy_matrix : (n_conditions, n_backbones)
    alpha : significance level

    Returns
    -------
    float : critical difference value
    """
    n, k = accuracy_matrix.shape
    # Studentized range critical values for Nemenyi (approximate for common k)
    # Source: tables for k groups at alpha=0.05
    q_table_005 = {
        3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_table_010 = {
        3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
        8: 2.780, 9: 2.855, 10: 2.920,
    }
    table = q_table_005 if alpha <= 0.05 else q_table_010
    q = table.get(k, 2.949)  # default to k=7

    return q * np.sqrt(k * (k + 1) / (6 * n))


def compute_mean_ranks(
    accuracy_matrix: np.ndarray,
    backbone_ids: List[str] = None,
) -> Dict[str, float]:
    """
    Compute mean ranks of backbones across conditions (higher accuracy = lower rank).

    Parameters
    ----------
    accuracy_matrix : (n_conditions, n_backbones)

    Returns
    -------
    dict : {backbone_id: mean_rank}
    """
    if backbone_ids is None:
        backbone_ids = [f'bb_{i}' for i in range(accuracy_matrix.shape[1])]

    n_cond, n_bb = accuracy_matrix.shape
    ranks = np.zeros_like(accuracy_matrix)
    for i in range(n_cond):
        # Rank: 1 = best (highest accuracy)
        ranks[i] = n_bb - np.argsort(np.argsort(accuracy_matrix[i]))

    mean_ranks = ranks.mean(axis=0)
    return {bb: float(mr) for bb, mr in zip(backbone_ids, mean_ranks)}


# ── Wilcoxon Signed-Rank + Bonferroni ────────────────────────────────────

def wilcoxon_pairwise(
    accuracy_matrix: np.ndarray,
    backbone_ids: List[str] = None,
    bonferroni: bool = True,
) -> dict:
    """
    Pairwise Wilcoxon signed-rank tests between all backbone pairs.

    Parameters
    ----------
    accuracy_matrix : (n_conditions, n_backbones)
    backbone_ids    : list of backbone names
    bonferroni      : if True, multiply p-values by n_comparisons

    Returns
    -------
    dict with keys: 'pairs' (list of dicts with 'a', 'b', 'stat', 'p', 'p_corrected', 'sig')
    """
    if backbone_ids is None:
        backbone_ids = [f'bb_{i}' for i in range(accuracy_matrix.shape[1])]

    n_bb = len(backbone_ids)
    n_comparisons = n_bb * (n_bb - 1) // 2

    pairs = []
    for i in range(n_bb):
        for j in range(i + 1, n_bb):
            a_vals = accuracy_matrix[:, i]
            b_vals = accuracy_matrix[:, j]
            diff = a_vals - b_vals
            if np.all(diff == 0):
                pairs.append({
                    'a': backbone_ids[i], 'b': backbone_ids[j],
                    'stat': np.nan, 'p': 1.0, 'p_corrected': 1.0, 'sig': 'n.s.',
                })
                continue
            try:
                stat, p = wilcoxon(a_vals, b_vals, alternative='two-sided')
                p_corr = min(p * n_comparisons, 1.0) if bonferroni else p
                pairs.append({
                    'a': backbone_ids[i], 'b': backbone_ids[j],
                    'stat': float(stat), 'p': float(p),
                    'p_corrected': float(p_corr),
                    'sig': sig_str(p_corr),
                })
            except Exception:
                pairs.append({
                    'a': backbone_ids[i], 'b': backbone_ids[j],
                    'stat': np.nan, 'p': np.nan, 'p_corrected': np.nan, 'sig': 'n/a',
                })

    return {'pairs': pairs, 'n_comparisons': n_comparisons, 'bonferroni': bonferroni}


# ── Kendall's W ──────────────────────────────────────────────────────────

def kendalls_w(
    accuracy_matrix: np.ndarray,
) -> float:
    """
    Kendall's W concordance coefficient.

    Measures agreement in backbone rankings across conditions.

    Parameters
    ----------
    accuracy_matrix : (n_conditions, n_backbones)

    Returns
    -------
    float : W in [0, 1], or NaN if insufficient data.
    """
    n, k = accuracy_matrix.shape
    if n < 2 or k < 2:
        return np.nan

    # Rank within each row (condition): higher value = rank 1
    ranks = np.zeros_like(accuracy_matrix)
    for i in range(n):
        ranks[i] = np.argsort(np.argsort(-accuracy_matrix[i])) + 1

    Rj = ranks.sum(axis=0)
    S = np.sum((Rj - Rj.mean()) ** 2)
    W = 12 * S / (n ** 2 * (k ** 3 - k))
    return float(W)


def kendalls_w_from_dict(
    rankings_dict: Dict[str, Dict[str, float]],
    backbone_ids: List[str],
) -> float:
    """
    Kendall's W from nested dict: {condition: {backbone: value}}.

    Parameters
    ----------
    rankings_dict : {condition_key: {backbone_id: accuracy_value}}
    backbone_ids  : ordered list of backbone IDs

    Returns
    -------
    float : W in [0, 1]
    """
    rows = []
    for _, bb_vals in rankings_dict.items():
        rows.append([bb_vals.get(b, 0) for b in backbone_ids])
    return kendalls_w(np.array(rows))


# ── Statistical helpers ──────────────────────────────────────────────────

def cohens_d(a, b) -> float:
    """Paired Cohen's d effect size."""
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    s = diff.std()
    if s < 1e-10:
        return 0.0
    return float(diff.mean() / s)


def safe_pearsonr(x, y) -> Tuple[float, float]:
    """Pearson r ignoring NaN pairs. Returns (r, p), or (NaN, NaN) if N < 3."""
    valid = [(a, b) for a, b in zip(x, y) if not (np.isnan(a) or np.isnan(b))]
    if len(valid) < 3:
        return np.nan, np.nan
    xa, ya = zip(*valid)
    r, p = pearsonr(xa, ya)
    return float(r), float(p)


def sig_str(p) -> str:
    """Return significance stars for a p-value."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return 'n/a'
    if np.isnan(p):
        return 'n/a'
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def accuracy_drop(acc_clean: float, acc_pert: float) -> float:
    """Absolute accuracy drop (positive = degradation)."""
    return acc_clean - acc_pert


def relative_drop(acc_clean: float, acc_pert: float) -> float:
    """Relative accuracy drop as fraction (positive = degradation)."""
    if acc_clean < 1e-10:
        return 0.0
    return (acc_clean - acc_pert) / acc_clean
