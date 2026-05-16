"""
Wood Spatial — Spatial & CAM Metrics
======================================
Novel metrics for evaluating spatial feature clustering and Grad-CAM overlap.

Spatial metrics:
    CSI  — brightness-ordered Cluster Stability Index
    CSI_H — label-invariant Hungarian-matched Cluster Stability Index
    SGI  — Spatial Granularity Index: connected components density
    ARI  — Adjusted Rand Index: cross-architecture cluster agreement
    Silhouette — cluster quality on spatial feature space

CAM-Cluster metrics:
    cluster_cam_distribution — fraction of CAM activation per cluster
    cam_shift_index          — Jensen-Shannon distance between clean and perturbed CAM distributions
    cam_entropy              — Shannon entropy of CAM distribution over clusters
"""
import logging

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import adjusted_rand_score, silhouette_score

from wood_spatial.config import N_CLUSTERS

logger = logging.getLogger(__name__)


# ── Cluster Stability Index (CSI) ────────────────────────────────────────

def cluster_iou(labels_a: np.ndarray, labels_b: np.ndarray, cluster_id: int) -> float:
    """IoU for a single cluster between two label maps."""
    mask_a = labels_a == cluster_id
    mask_b = labels_b == cluster_id
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_csi(
    labels_clean: np.ndarray,
    labels_perturbed: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> float:
    """
    Brightness-ordered Cluster Stability Index: mean IoU across all clusters.

    CSI = (1/k) * sum_{i=0}^{k-1} IoU(clean_cluster_i, perturbed_cluster_i)

    Both label maps must use brightness-ordered IDs (0=darkest, ..., K-1=lightest).

    Parameters
    ----------
    labels_clean     : (H, W) int — cluster labels from clean image
    labels_perturbed : (H, W) int — cluster labels from perturbed image
    n_clusters       : number of clusters

    Returns
    -------
    float in [0, 1] — 1.0 means perfect stability
    """
    ious = []
    for k in range(n_clusters):
        ious.append(cluster_iou(labels_clean, labels_perturbed, k))
    return float(np.mean(ious))


def compute_iou_matrix(
    labels_clean: np.ndarray,
    labels_perturbed: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> np.ndarray:
    """Pairwise IoU matrix between clean and perturbed cluster masks."""
    mat = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    for i in range(n_clusters):
        mask_i = labels_clean == i
        for j in range(n_clusters):
            mask_j = labels_perturbed == j
            intersection = np.logical_and(mask_i, mask_j).sum()
            union = np.logical_or(mask_i, mask_j).sum()
            mat[i, j] = 1.0 if union == 0 else intersection / union
    return mat


def compute_csi_hungarian(
    labels_clean: np.ndarray,
    labels_perturbed: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> tuple:
    """
    Label-invariant CSI using Hungarian matching over the IoU matrix.

    Returns (score, matching, iou_matrix), where matching maps clean cluster
    IDs to perturbed cluster IDs.
    """
    iou = compute_iou_matrix(labels_clean, labels_perturbed, n_clusters)
    row_ind, col_ind = linear_sum_assignment(-iou)
    score = float(iou[row_ind, col_ind].mean())
    matching = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    return score, matching, iou


def compute_csi_metrics(
    labels_clean: np.ndarray,
    labels_perturbed: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> dict:
    """Compute brightness-ordered and Hungarian-matched CSI variants."""
    csi_bo = compute_csi(labels_clean, labels_perturbed, n_clusters)
    csi_h, matching, _ = compute_csi_hungarian(
        labels_clean, labels_perturbed, n_clusters)
    return {
        'csi': csi_bo,
        'csi_bo': csi_bo,
        'csi_hungarian': csi_h,
        'csi_permutation_gap': csi_h - csi_bo,
        'csi_matching': matching,
    }


def compute_csi_per_cluster(
    labels_clean: np.ndarray,
    labels_perturbed: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> dict:
    """CSI broken down per cluster."""
    return {k: cluster_iou(labels_clean, labels_perturbed, k)
            for k in range(n_clusters)}


# ── Spatial Granularity Index (SGI) ───────────────────────────────────────

def compute_sgi(labels: np.ndarray, n_clusters: int = N_CLUSTERS) -> float:
    """
    Spatial Granularity Index: total connected components normalized by image area.

    SGI = sum_{k=0}^{k-1} n_components(cluster_k) / (H * W)

    Higher SGI = finer spatial granularity (more small components).
    HRNet should produce highest SGI due to 56x56 feature resolution.

    Parameters
    ----------
    labels : (H, W) int — cluster label map

    Returns
    -------
    float — components per pixel (typically 1e-5 to 1e-3)
    """
    H, W = labels.shape
    total_components = 0
    for k in range(n_clusters):
        mask = (labels == k).astype(np.uint8)
        n_labels, _ = cv2.connectedComponents(mask, connectivity=8)
        total_components += (n_labels - 1)  # subtract background label
    return float(total_components / (H * W))


def compute_sgi_per_cluster(labels: np.ndarray, n_clusters: int = N_CLUSTERS) -> dict:
    """SGI broken down per cluster."""
    H, W = labels.shape
    result = {}
    for k in range(n_clusters):
        mask = (labels == k).astype(np.uint8)
        n_labels, _ = cv2.connectedComponents(mask, connectivity=8)
        result[k] = float((n_labels - 1) / (H * W))
    return result


# ── Cross-Architecture ARI ────────────────────────────────────────────────

def compute_ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Adjusted Rand Index between two cluster label maps.

    Measures how similarly two backbones cluster the same image,
    independent of cluster ID assignment.

    Parameters
    ----------
    labels_a : (H, W) int — cluster labels from backbone A
    labels_b : (H, W) int — cluster labels from backbone B

    Returns
    -------
    float in [-1, 1] — 1.0 means perfect agreement
    """
    return float(adjusted_rand_score(labels_a.ravel(), labels_b.ravel()))


# ── Silhouette Score ──────────────────────────────────────────────────────

def compute_silhouette(
    spatial_feat: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 1000,
    random_state: int = 42,
) -> float:
    """
    Silhouette score for spatial clustering quality.

    Subsamples spatial positions for computational efficiency.

    Parameters
    ----------
    spatial_feat : (C, H_f, W_f) float32 — raw spatial features
    labels       : (H_f, W_f) int — cluster labels at feature resolution

    Returns
    -------
    float in [-1, 1] — higher is better clustering
    """
    C, H_f, W_f = spatial_feat.shape
    feat_flat = spatial_feat.transpose(1, 2, 0).reshape(-1, C)
    labels_flat = labels.ravel()

    n_total = feat_flat.shape[0]
    if n_total > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_total, max_samples, replace=False)
        feat_flat = feat_flat[idx]
        labels_flat = labels_flat[idx]

    n_unique = len(np.unique(labels_flat))
    if n_unique < 2:
        return 0.0

    return float(silhouette_score(feat_flat, labels_flat))


def downsample_labels_to_feat(
    labels: np.ndarray,
    h_f: int,
    w_f: int,
) -> np.ndarray:
    """Downsample full-resolution labels to feature map resolution via nearest."""
    return cv2.resize(labels.astype(np.float32), (w_f, h_f),
                      interpolation=cv2.INTER_NEAREST).astype(np.int32)


# ── CAM-Cluster Metrics ──────────────────────────────────────────────────

def cluster_cam_distribution(
    cam: np.ndarray,
    labels: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> np.ndarray:
    """
    Fraction of CAM activation falling in each cluster.

    pct_k = sum(CAM * mask_k) / sum(CAM)

    Parameters
    ----------
    cam    : (H, W) float in [0, 1] — Grad-CAM or Centroid-CAM heatmap
    labels : (H, W) int — cluster labels (same resolution as cam)

    Returns
    -------
    (n_clusters,) float — sums to 1.0
    """
    total = cam.sum()
    if total < 1e-10:
        return np.ones(n_clusters) / n_clusters

    dist = np.zeros(n_clusters)
    for k in range(n_clusters):
        mask = labels == k
        dist[k] = cam[mask].sum() / total
    return dist


def cam_shift_index(dist_clean: np.ndarray, dist_perturbed: np.ndarray) -> float:
    """
    CAM Shift Index: Jensen-Shannon distance between clean and perturbed
    CAM-cluster distributions.

    Parameters
    ----------
    dist_clean     : (k,) float — cluster_cam_distribution from clean
    dist_perturbed : (k,) float — cluster_cam_distribution from perturbed

    Returns
    -------
    float in [0, 1] — 0 means no shift, 1 means maximum distance
    """
    # scipy.spatial.distance.jensenshannon returns JS distance, not divergence.
    eps = 1e-10
    p = np.clip(dist_clean, eps, None)
    q = np.clip(dist_perturbed, eps, None)
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q, base=2))


def cam_js_divergence(dist_clean: np.ndarray, dist_perturbed: np.ndarray) -> float:
    """Jensen-Shannon divergence, equal to squared JS distance with base 2."""
    dist = cam_shift_index(dist_clean, dist_perturbed)
    return float(dist * dist)


def cam_entropy(dist: np.ndarray) -> float:
    """
    Shannon entropy of CAM distribution over clusters.

    Low entropy = CAM concentrated on few clusters.
    Max entropy = log2(k) when uniform.

    Parameters
    ----------
    dist : (k,) float — cluster_cam_distribution

    Returns
    -------
    float in [0, log2(k)]
    """
    eps = 1e-10
    p = np.clip(dist, eps, None)
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


# ── Convenience: Compute all spatial metrics for one image ────────────────

def compute_all_spatial_metrics(
    labels_clean: np.ndarray,
    labels_perturbed: np.ndarray = None,
    spatial_feat: np.ndarray = None,
    n_clusters: int = N_CLUSTERS,
) -> dict:
    """
    Compute all available spatial metrics for a single image.

    Parameters
    ----------
    labels_clean     : (H, W) int — cluster labels from clean image
    labels_perturbed : (H, W) int or None — labels from perturbed image
    spatial_feat     : (C, H_f, W_f) or None — raw spatial features for silhouette

    Returns
    -------
    dict with keys: 'sgi', 'sgi_per_cluster', and optionally 'csi', 'csi_per_cluster', 'silhouette'
    """
    metrics = {
        'sgi': compute_sgi(labels_clean, n_clusters),
        'sgi_per_cluster': compute_sgi_per_cluster(labels_clean, n_clusters),
    }

    if labels_perturbed is not None:
        metrics.update(compute_csi_metrics(labels_clean, labels_perturbed, n_clusters))
        metrics['csi_per_cluster'] = compute_csi_per_cluster(
            labels_clean, labels_perturbed, n_clusters)

    if spatial_feat is not None:
        C, H_f, W_f = spatial_feat.shape
        labels_ds = downsample_labels_to_feat(labels_clean, H_f, W_f)
        metrics['silhouette'] = compute_silhouette(spatial_feat, labels_ds)

    return metrics
