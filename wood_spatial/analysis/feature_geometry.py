"""
Feature geometry metrics for multi-level domain gap analysis.
"""
import numpy as np


def _as_2d_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f'Expected 2D feature array, got shape {x.shape}')
    return x


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    features = _as_2d_float(features)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)


def cosine_distance_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Mean paired cosine distance between feature matrices."""
    a = l2_normalize(a)
    b = l2_normalize(b)
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    sim = np.clip(np.sum(a[:n] * b[:n], axis=1), -1.0, 1.0)
    return float(max(0.0, np.mean(1.0 - sim)))


def feature_drift(feat_clean: np.ndarray, feat_shift: np.ndarray) -> float:
    """Per-image clean-to-shift cosine feature drift."""
    return cosine_distance_mean(feat_clean, feat_shift)


def class_centroids(features: np.ndarray, labels: np.ndarray, normalize: bool = True) -> dict:
    """Compute class centroids."""
    features = _as_2d_float(features)
    labels = np.asarray(labels)
    centroids = {}
    for cls in np.unique(labels):
        mask = labels == cls
        if not np.any(mask):
            continue
        centroid = features[mask].mean(axis=0)
        if normalize:
            norm = np.linalg.norm(centroid)
            if norm > 1e-12:
                centroid = centroid / norm
        centroids[cls] = centroid.astype(np.float32)
    return centroids


def intra_class_variance(features: np.ndarray, labels: np.ndarray) -> float:
    """Mean cosine distance from samples to their class centroid."""
    features = l2_normalize(features)
    labels = np.asarray(labels)
    centroids = class_centroids(features, labels, normalize=True)
    values = []
    for cls, centroid in centroids.items():
        mask = labels == cls
        if mask.sum() < 2:
            continue
        sim = features[mask] @ centroid
        values.extend(1.0 - sim)
    if not values:
        return np.nan
    return float(np.mean(values))


def inter_class_distance(features: np.ndarray, labels: np.ndarray) -> float:
    """Mean pairwise cosine distance between class centroids."""
    features = l2_normalize(features)
    centroids = list(class_centroids(features, labels, normalize=True).values())
    if len(centroids) < 2:
        return np.nan
    centroids = np.stack(centroids, axis=0)
    sims = centroids @ centroids.T
    iu = np.triu_indices(len(centroids), k=1)
    return float(np.mean(1.0 - sims[iu]))


def feature_geometry_collapse_score(
    features: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """FGCS = intra-class variance / inter-class distance."""
    intra = intra_class_variance(features, labels)
    inter = inter_class_distance(features, labels)
    if np.isnan(intra) or np.isnan(inter):
        return np.nan
    return float(intra / max(inter, eps))


def fisher_separability_ratio(
    features: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """FSR = inter-class distance / intra-class variance."""
    intra = intra_class_variance(features, labels)
    inter = inter_class_distance(features, labels)
    if np.isnan(intra) or np.isnan(inter):
        return np.nan
    return float(inter / max(intra, eps))


def geometry_summary(features: np.ndarray, labels: np.ndarray) -> dict:
    """Compute the core feature geometry metrics."""
    intra = intra_class_variance(features, labels)
    inter = inter_class_distance(features, labels)
    return {
        'intra': intra,
        'inter': inter,
        'fgcs': feature_geometry_collapse_score(features, labels),
        'fsr': fisher_separability_ratio(features, labels),
    }
