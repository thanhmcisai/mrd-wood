"""
Wood Spatial — Centroid-CAM Module
====================================
Gradient-free class activation mapping using class centroids from kNN features.

Centroid-CAM: cosine similarity between per-pixel spatial features and class centroids.
This avoids the need for backpropagation on frozen models.

Also provides a wrapper for True Grad-CAM via GradCAMExtractor in backbone.py.
"""
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from wood_spatial.config import V4_GRADCAM_CACHE

logger = logging.getLogger(__name__)


# ── Centroid-CAM (gradient-free) ──────────────────────────────────────────

def compute_centroids(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute L2-normalized class centroids from global features.

    Parameters
    ----------
    features : (N, D) float — L2-normalized global features
    labels   : (N,) int — class labels

    Returns
    -------
    (n_classes, D) float — L2-normalized centroids
    """
    classes = np.unique(labels)
    D = features.shape[1]
    centroids = np.zeros((len(classes), D), dtype=np.float32)
    for i, c in enumerate(sorted(classes)):
        mask = labels == c
        centroid = features[mask].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid /= norm
        centroids[i] = centroid
    return centroids


def centroid_cam(
    spatial_feat: np.ndarray,
    centroid: np.ndarray,
    img_h: int = None,
    img_w: int = None,
) -> np.ndarray:
    """
    Compute Centroid-CAM: cosine similarity between each spatial position
    and the target class centroid.

    Parameters
    ----------
    spatial_feat : (C, H_f, W_f) float — spatial features from SpatialExtractor
    centroid     : (D,) float — L2-normalized class centroid from global features
    img_h, img_w : if provided, upsample CAM to this resolution

    Returns
    -------
    (H, W) float32 heatmap in [0, 1]

    Note
    ----
    The spatial feature channels (C) may differ from the global feature dim (D)
    when SpatialExtractor uses an intermediate layer vs GlobalExtractor using
    the final pool. In that case, the cosine similarity still works if both are
    from the same backbone — the centroid should be computed from the same
    spatial features (pooled), not from GlobalExtractor features.
    """
    C, H_f, W_f = spatial_feat.shape

    # Normalize spatial features per-position
    feat_flat = spatial_feat.reshape(C, -1).T  # (H_f*W_f, C)
    norms = np.linalg.norm(feat_flat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    feat_norm = feat_flat / norms

    # Normalize centroid
    c_norm = centroid / max(np.linalg.norm(centroid), 1e-8)

    # Cosine similarity
    sim = feat_norm @ c_norm  # (H_f*W_f,)
    sim = sim.reshape(H_f, W_f)

    # ReLU + normalize to [0, 1]
    cam = np.maximum(sim, 0)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    # Upsample if requested
    if img_h is not None and img_w is not None:
        cam = cv2.resize(cam, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    return cam.astype(np.float32)


def compute_spatial_centroids(
    spatial_features: list,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute class centroids from spatial features (pooled to global).

    This produces centroids in the same feature space as SpatialExtractor output,
    which is what Centroid-CAM needs.

    Parameters
    ----------
    spatial_features : list of (C, H_f, W_f) arrays
    labels           : (N,) int — class labels

    Returns
    -------
    (n_classes, C) float — L2-normalized centroids in spatial feature space
    """
    # Pool spatial features to global
    pooled = []
    for feat in spatial_features:
        pooled.append(feat.mean(axis=(1, 2)))  # (C,) — spatial average
    pooled = np.stack(pooled)  # (N, C)

    # L2-normalize
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    pooled = pooled / norms

    return compute_centroids(pooled, labels)


# ── True Grad-CAM wrapper ────────────────────────────────────────────────

def compute_gradcam_batch(
    backbone_id: str,
    images: list,
    labels: np.ndarray,
    centroids: np.ndarray,
    device: str = 'cpu',
) -> list:
    """
    Compute Grad-CAM for a batch of images using GradCAMExtractor.

    Parameters
    ----------
    backbone_id : str — backbone key
    images      : list of (3, H, W) tensors — normalized image tensors
    labels      : (N,) int — predicted class for each image
    centroids   : (n_classes, D) float — class centroids for classifier init
    device      : 'cpu' or 'cuda'

    Returns
    -------
    list of (H_f, W_f) numpy heatmaps in [0, 1]
    """
    from wood_spatial.core.backbone import GradCAMExtractor

    n_classes = centroids.shape[0]
    extractor = GradCAMExtractor(backbone_id, n_classes).to(device)
    extractor.init_classifier_from_centroids(
        torch.from_numpy(centroids).float().to(device)
    )

    cams = []
    for img_tensor, label in zip(images, labels):
        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor)
        x = img_tensor.unsqueeze(0).float().to(device)
        cam = extractor.compute_gradcam(x, int(label))
        cams.append(cam.cpu().numpy())

    del extractor
    if device == 'cuda':
        torch.cuda.empty_cache()

    return cams


# ── Cache I/O ─────────────────────────────────────────────────────────────

def get_gradcam_cache_path(backbone_id: str, dataset_name: str, tag: str = 'original'):
    from wood_spatial.core.cache import _safe_tag
    safe = _safe_tag(tag)
    return V4_GRADCAM_CACHE / f'{backbone_id}_{dataset_name}_{safe}_cam.npz'


def save_gradcam_cache(
    cams: list,
    labels: np.ndarray,
    paths: np.ndarray,
    backbone_id: str,
    dataset_name: str,
    tag: str = 'original',
):
    """Save Grad-CAM heatmaps to cache."""
    save_path = get_gradcam_cache_path(backbone_id, dataset_name, tag)
    np.savez_compressed(
        save_path,
        cams=np.array(cams, dtype=object),
        labels=labels,
        paths=paths,
    )
    logger.info('Saved Grad-CAM cache: %s (%d images)', save_path.name, len(cams))


def load_gradcam_cache(backbone_id: str, dataset_name: str, tag: str = 'original'):
    """Load Grad-CAM cache. Returns (cams_list, labels, paths)."""
    path = get_gradcam_cache_path(backbone_id, dataset_name, tag)
    if not path.exists():
        raise FileNotFoundError(f'Grad-CAM cache not found: {path}')
    d = np.load(path, allow_pickle=True)
    return list(d['cams']), d['labels'], d['paths']
