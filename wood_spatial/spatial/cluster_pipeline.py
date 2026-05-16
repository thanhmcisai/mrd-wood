"""
Wood Spatial — Spatial Cluster Pipeline
=========================================
K-Means clustering on backbone spatial feature maps with Guided Filter refinement
and Smart Hole Filling post-processing.

Adapted from HRNet_exp/compare_multi_model.py and HRNet_exp/visualize_images.py.

Pipeline:
    1. K-Means k=3 on flattened spatial features (H*W, C)
    2. Soft assignment via exp(-distance)
    3. Upsample to original image resolution
    4. Guided Filter edge-aware refinement
    5. Brightness-ordered cluster labeling (0=darkest, 2=lightest)
    6. Smart Hole Filling: fill dark-cluster contours, morphological cleanup
"""
import logging

import cv2
import numpy as np
from sklearn.cluster import KMeans

from wood_spatial.config import (
    N_CLUSTERS, GUIDED_RADIUS, GUIDED_EPS,
    VESSEL_MIN_AREA, MORPH_KERNEL_SIZE,
)

KMEANS_N_INIT = 10

logger = logging.getLogger(__name__)


# ── Guided Filter ─────────────────────────────────────────────────────────

def guided_filter(guide: np.ndarray, src: np.ndarray,
                  radius: int = GUIDED_RADIUS, eps: float = GUIDED_EPS) -> np.ndarray:
    """
    Edge-preserving guided filter.

    Parameters
    ----------
    guide : (H, W) float32 grayscale guide image
    src   : (H, W) float32 source signal to filter
    radius : int  box filter radius
    eps    : float  regularization

    Returns
    -------
    (H, W) float32 filtered output
    """
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
    return mean_a * guide + mean_b


def correct_illumination(img_rgb: np.ndarray) -> np.ndarray:
    """
    Flatten uneven illumination via LAB background subtraction + CLAHE.

    Parameters
    ----------
    img_rgb : (H, W, 3) uint8 RGB image

    Returns
    -------
    (H, W, 3) uint8 corrected RGB image
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    bg_map = cv2.GaussianBlur(l, (0, 0), sigmaX=100, sigmaY=100)
    l_float = l.astype(np.float32)
    bg_float = bg_map.astype(np.float32)
    mean_light = np.mean(bg_float)
    l_flat = np.clip(l_float - bg_float + mean_light, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_flat)
    return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)


# ── Core Clustering Pipeline ──────────────────────────────────────────────

def cluster_spatial_features(
    spatial_feat: np.ndarray,
    guide_gray: np.ndarray,
    img_h: int,
    img_w: int,
    n_clusters: int = N_CLUSTERS,
    random_state: int = 42,
) -> dict:
    """
    Full clustering pipeline on a single image's spatial feature map.

    Parameters
    ----------
    spatial_feat : (C, H_f, W_f) float32 numpy array — raw spatial features
    guide_gray   : (img_h, img_w) float32 grayscale guide image in [0, 1]
    img_h, img_w : target output resolution
    n_clusters   : number of clusters (default 3)
    random_state : KMeans seed

    Returns
    -------
    dict with keys:
        'labels'       : (img_h, img_w) int32 — brightness-ordered cluster labels
        'labels_raw'   : (img_h, img_w) int32 — labels before hole filling
        'kmeans'       : fitted KMeans object
        'probs'        : (img_h, img_w, k) float32 — refined soft assignments
        'brightness_order' : list[int] — mapping from new_idx to old KMeans idx
    """
    C, H_f, W_f = spatial_feat.shape

    # Reshape to (H_f * W_f, C) for K-Means and normalize each spatial token.
    # This keeps Euclidean K-Means closer to cosine clustering and prevents
    # large raw activation norms from dominating soft assignments.
    feat_flat = spatial_feat.transpose(1, 2, 0).reshape(-1, C)  # (N, C)
    norms = np.linalg.norm(feat_flat, axis=1, keepdims=True)
    feat_flat = feat_flat / np.maximum(norms, 1e-8)

    # K-Means clustering directly in the normalized backbone feature space.
    # We intentionally avoid PCA here so CSI/SGI remain interpretable as
    # properties of the original spatial tokens rather than projected tokens.
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=KMEANS_N_INIT,
    )
    distances = kmeans.fit_transform(feat_flat)  # (N, k)

    # Soft assignment via exp(-distance); clamp to avoid NaN when all distances are huge
    probs_flat = np.exp(-np.clip(distances, 0, 500))
    row_sums = probs_flat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-30, 1.0, row_sums)
    probs_flat /= row_sums
    probs_spatial = probs_flat.reshape(H_f, W_f, n_clusters)

    # Upsample to image resolution
    probs_up = cv2.resize(probs_spatial, (img_w, img_h),
                          interpolation=cv2.INTER_LINEAR)

    # Guided Filter refinement per channel
    refined_probs = np.zeros_like(probs_up)
    for k in range(n_clusters):
        refined_probs[:, :, k] = guided_filter(
            guide_gray, probs_up[:, :, k].astype(np.float32),
            GUIDED_RADIUS, GUIDED_EPS,
        )

    # Hard assignment
    pixel_labels = np.argmax(refined_probs, axis=-1)

    # Brightness ordering: sort clusters by mean guide grayscale
    means = []
    for k in range(n_clusters):
        mask = pixel_labels == k
        means.append(guide_gray[mask].mean() if mask.any() else 0.0)
    sorted_indices = np.argsort(means)  # darkest first

    # Remap labels: 0=darkest, ..., n_clusters-1=lightest (brightness ordering)
    mapped_labels = np.zeros_like(pixel_labels)
    for new_idx, old_idx in enumerate(sorted_indices):
        mapped_labels[pixel_labels == old_idx] = new_idx

    labels_raw = mapped_labels.copy()

    # Smart Hole Filling
    final_labels = _smart_hole_filling(mapped_labels, n_clusters)

    return {
        'labels': final_labels,
        'labels_raw': labels_raw,
        'kmeans': kmeans,
        'probs': refined_probs,
        'brightness_order': sorted_indices.tolist(),
    }


def _smart_hole_filling(
    labels: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> np.ndarray:
    """
    Post-process cluster labels with anatomically-informed hole filling.

    1. Fill vessel (ID=0) contours: holes from tyloses are filled if contour area > threshold
    2. Morphological open on parenchyma (ID=1) and ray (ID=2) to remove noise

    Parameters
    ----------
    labels : (H, W) int32 brightness-ordered cluster labels

    Returns
    -------
    (H, W) int32 post-processed labels
    """
    result = labels.copy()

    # 1. Dark-cluster contour filling (ID=0, darkest pixels)
    vessel_mask = (labels == 0).astype(np.uint8)
    contours, _ = cv2.findContours(vessel_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    vessel_filled = np.zeros_like(vessel_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > VESSEL_MIN_AREA:
            cv2.drawContours(vessel_filled, [cnt], -1, 1, thickness=cv2.FILLED)

    # Overwrite vessel region (fills holes inside large vessel contours)
    result[vessel_filled == 1] = 0

    # 2. Morphological cleanup for parenchyma (1) and ray (2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    fiber_id = n_clusters - 1  # brightest = fiber background
    for tissue_id in [1, 2]:
        if tissue_id >= n_clusters:
            continue
        mask = (result == tissue_id).astype(np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Pixels that were this tissue but got cleaned → reassign to fiber
        result[(result == tissue_id) & (vessel_filled == 0)] = fiber_id
        result[cleaned == 1] = tissue_id

    return result


# ── Batch Processing ──────────────────────────────────────────────────────

def cluster_batch(
    spatial_features: list,
    images_rgb: list,
    img_size: int = 224,
    n_clusters: int = N_CLUSTERS,
) -> list:
    """
    Run clustering pipeline on a batch of images.

    Parameters
    ----------
    spatial_features : list of (C, H_f, W_f) numpy arrays
    images_rgb       : list of (H, W, 3) uint8 RGB images (original resolution)
    img_size         : target resolution for clustering output
    n_clusters       : number of clusters

    Returns
    -------
    list of result dicts from cluster_spatial_features()
    """
    results = []
    for i, (feat, img) in enumerate(zip(spatial_features, images_rgb)):
        # Resize image to target if needed
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img_resized = cv2.resize(img, (img_size, img_size))
        else:
            img_resized = img

        # Correct illumination and compute guide
        img_corrected = correct_illumination(img_resized)
        guide_gray = cv2.cvtColor(img_corrected, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        result = cluster_spatial_features(
            feat, guide_gray, img_size, img_size, n_clusters,
        )
        results.append(result)

    return results


def make_guide_from_tensor(img_tensor, img_size: int = 224) -> np.ndarray:
    """
    Convert a normalized image tensor to a float32 grayscale guide image.

    Parameters
    ----------
    img_tensor : (3, H, W) torch.Tensor — ImageNet-normalized

    Returns
    -------
    (img_size, img_size) float32 in [0, 1]
    """
    from wood_spatial.core.dataset import denormalize
    import torch

    if isinstance(img_tensor, torch.Tensor):
        raw = denormalize(img_tensor).cpu().numpy()  # (3, H, W) in [0, 1]
    else:
        raw = img_tensor

    rgb = (raw.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    corrected = correct_illumination(rgb)
    guide = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    if guide.shape[0] != img_size or guide.shape[1] != img_size:
        guide = cv2.resize(guide, (img_size, img_size))

    return guide


def make_overlay(img_rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    Create a color overlay of cluster labels on the original image.

    Parameters
    ----------
    img_rgb : (H, W, 3) uint8 RGB image
    labels  : (H, W) int32 cluster labels
    alpha   : blend factor for cluster colors

    Returns
    -------
    (H, W, 3) uint8 overlay image
    """
    from wood_spatial.config import CLUSTER_COLOR
    import matplotlib.colors as mcolors

    cmap_img = np.zeros((*labels.shape, 3), dtype=np.float32)
    for k, hex_color in CLUSTER_COLOR.items():
        rgb = np.array(mcolors.hex2color(hex_color)) * 255
        cmap_img[labels == k] = rgb

    overlay = (img_rgb.astype(np.float32) * (1 - alpha) + cmap_img * alpha)
    return overlay.clip(0, 255).astype(np.uint8)
