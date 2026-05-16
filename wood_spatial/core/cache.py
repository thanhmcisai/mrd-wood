"""
Wood Spatial — Feature Cache
==============================
Resume-safe feature extraction and caching for both global and spatial features.

Global cache:  results_v4/feature_cache/{backbone}_{dataset}_{tag}.npz
Spatial cache: results_v4/spatial_cache/{backbone}_{dataset}_{img_id}_{tag}.npz

Also supports loading v2 caches (results_v2/feature_cache/) for reuse.
"""
import logging
from pathlib import Path
import zipfile

import numpy as np
import torch

from wood_spatial.config import (
    BACKBONE_CONFIGS, V2_CACHE_DIR, V4_FEAT_CACHE, V4_SPATIAL_CACHE,
    BATCH_SIZE, NUM_WORKERS, FP16, RANDOM_SEED,
)

logger = logging.getLogger(__name__)


class CorruptCacheError(IOError):
    """Raised when an existing cache file cannot be read as a valid npz."""


def _safe_tag(tag: str, maxlen: int = 60) -> str:
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in tag)[:maxlen]


def _load_npz_checked(path: Path):
    try:
        return np.load(path, allow_pickle=True)
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise CorruptCacheError(f'Corrupt cache file: {path} ({exc})') from exc


def _is_valid_npz(path: Path, required_keys=('features', 'labels', 'paths')) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            return all(key in data.files for key in required_keys)
    except (EOFError, OSError, ValueError, zipfile.BadZipFile):
        return False


# ── Global feature cache ───────────────────────────────────────────────────

def get_cache_path(backbone_id: str, dataset_name: str, tag: str = 'original') -> Path:
    """Return v4 cache path for a (backbone, dataset, tag) triple."""
    safe = _safe_tag(tag)
    return V4_FEAT_CACHE / f'{backbone_id}_{dataset_name}_{safe}.npz'


def get_v2_cache_path(backbone_id: str, dataset_name: str, tag: str = 'original') -> Path:
    """Return v2 cache path (for reuse of existing extractions)."""
    safe = _safe_tag(tag)
    return V2_CACHE_DIR / f'{backbone_id}_{dataset_name}_all_{safe}.npz'


def load_cache(backbone_id: str, dataset_name: str, tag: str = 'original'):
    """
    Load feature cache, trying v4 first, then v2 fallback.

    Returns (features, labels, paths) as numpy arrays.
    """
    # Try v4 first
    v4_path = get_cache_path(backbone_id, dataset_name, tag)
    if v4_path.exists():
        d = _load_npz_checked(v4_path)
        return d['features'], d['labels'], d['paths']

    # Try v2 fallback
    v2_path = get_v2_cache_path(backbone_id, dataset_name, tag)
    if v2_path.exists():
        logger.debug('Using v2 cache: %s', v2_path.name)
        d = _load_npz_checked(v2_path)
        return d['features'], d['labels'], d['paths']

    raise FileNotFoundError(
        f'Cache not found for {backbone_id}/{dataset_name}/{tag}. '
        f'Checked: {v4_path}, {v2_path}'
    )


def cache_exists(backbone_id: str, dataset_name: str, tag: str = 'original') -> bool:
    """Check if cache exists in v4 or v2."""
    return (
        _is_valid_npz(get_cache_path(backbone_id, dataset_name, tag))
        or _is_valid_npz(get_v2_cache_path(backbone_id, dataset_name, tag))
    )


def save_cache(
    features: np.ndarray,
    labels: np.ndarray,
    paths: np.ndarray,
    backbone_id: str,
    dataset_name: str,
    tag: str = 'original',
):
    """Save feature cache to v4 directory."""
    save_path = get_cache_path(backbone_id, dataset_name, tag)
    np.savez_compressed(save_path, features=features, labels=labels, paths=paths)
    logger.info('Saved cache: %s (%d samples, %d-dim)',
                save_path.name, len(features), features.shape[1])


def extract_and_save(
    backbone_id: str,
    dataset_root: str,
    dataset_name: str,
    perturbation=None,
    tag: str = 'original',
    max_images: int = None,
    force: bool = False,
):
    """
    Extract global features and save to v4 cache.
    Skips if cache already exists (v4 or v2) unless force=True.

    Returns (features, labels, paths).
    """
    from tqdm import tqdm
    from wood_spatial.core.dataset import WoodDataset, make_loader
    from wood_spatial.core.backbone import GlobalExtractor

    if not force and cache_exists(backbone_id, dataset_name, tag):
        logger.debug('Cache hit: %s/%s/%s', backbone_id, dataset_name, tag)
        return load_cache(backbone_id, dataset_name, tag)

    logger.info('Extracting: %s | %s | %s', backbone_id, dataset_name, tag[:30])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = BACKBONE_CONFIGS[backbone_id]

    ds = WoodDataset(
        dataset_root, cfg['img_size'],
        random_seed=RANDOM_SEED,
        max_images=max_images,
        perturbation=perturbation,
    )
    bs = cfg.get('batch', BATCH_SIZE)
    loader = make_loader(ds, bs, NUM_WORKERS)
    extractor = GlobalExtractor(backbone_id).to(device)

    all_feats, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc=f'  {backbone_id}|{tag[:18]}', leave=False):
            imgs = imgs.to(device)
            if FP16 and device == 'cuda':
                with torch.amp.autocast('cuda'):
                    feats = extractor(imgs)
            else:
                feats = extractor(imgs)
            all_feats.append(feats.float().cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    features = np.concatenate(all_feats)
    labels_np = np.array(all_labels)
    paths_np = np.array(all_paths)

    save_cache(features, labels_np, paths_np, backbone_id, dataset_name, tag)

    del extractor
    if device == 'cuda':
        torch.cuda.empty_cache()

    return features, labels_np, paths_np


# ── Spatial feature cache ──────────────────────────────────────────────────

def get_spatial_cache_path(backbone_id: str, dataset_name: str, tag: str = 'original') -> Path:
    """Return spatial cache path."""
    safe = _safe_tag(tag)
    return V4_SPATIAL_CACHE / f'{backbone_id}_{dataset_name}_{safe}.npz'


def load_spatial_cache(backbone_id: str, dataset_name: str, tag: str = 'original'):
    """
    Load spatial feature cache.

    Returns dict with keys: 'features' (list of (C,H,W) arrays), 'labels', 'paths'.
    """
    path = get_spatial_cache_path(backbone_id, dataset_name, tag)
    if not path.exists():
        raise FileNotFoundError(f'Spatial cache not found: {path}')
    d = _load_npz_checked(path)
    features = d['features']
    if features.ndim == 4 and features.shape[-1] > features.shape[1]:
        features = features.transpose(0, 3, 1, 2)
    return features, d['labels'], d['paths']


def save_spatial_cache(
    features: list,
    labels: np.ndarray,
    paths: np.ndarray,
    backbone_id: str,
    dataset_name: str,
    tag: str = 'original',
):
    """
    Save spatial features to cache.

    features: list of (C, H_f, W_f) numpy arrays (variable size per backbone).
    """
    save_path = get_spatial_cache_path(backbone_id, dataset_name, tag)
    # Stack into 4D array — all images from same backbone have identical spatial dims
    features_arr = np.stack(features, axis=0)  # (N, C, H_f, W_f)
    np.savez_compressed(
        save_path,
        features=features_arr,
        labels=labels,
        paths=paths,
    )
    logger.info('Saved spatial cache: %s (%d samples, shape=%s)',
                save_path.name, len(features), features_arr.shape)


def extract_spatial_and_save(
    backbone_id: str,
    dataset_root: str,
    dataset_name: str,
    perturbation=None,
    tag: str = 'original',
    max_images: int = None,
    force: bool = False,
):
    """
    Extract spatial features (intermediate feature maps) and save to cache.

    Returns (features_list, labels, paths) where features_list contains
    (C, H_f, W_f) numpy arrays.
    """
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from wood_spatial.core.dataset import WoodDataset
    from wood_spatial.core.backbone import SpatialExtractor
    from wood_spatial.config import SPATIAL_BATCH_SIZE

    cache_path = get_spatial_cache_path(backbone_id, dataset_name, tag)
    if _is_valid_npz(cache_path) and not force:
        logger.debug('Spatial cache hit: %s', cache_path.name)
        return load_spatial_cache(backbone_id, dataset_name, tag)

    logger.info('Extracting spatial: %s | %s | %s', backbone_id, dataset_name, tag[:30])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = BACKBONE_CONFIGS[backbone_id]

    ds = WoodDataset(
        dataset_root, cfg['img_size'],
        random_seed=RANDOM_SEED,
        max_images=max_images,
        perturbation=perturbation,
    )
    loader = DataLoader(ds, batch_size=SPATIAL_BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)
    extractor = SpatialExtractor(backbone_id).to(device)

    all_feats, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc=f'  spatial|{backbone_id}|{tag[:14]}', leave=False):
            imgs = imgs.to(device)
            spatial = extractor(imgs)
            # Store per-image spatial maps
            for i in range(spatial.shape[0]):
                all_feats.append(spatial[i].float().cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    labels_np = np.array(all_labels)
    paths_np = np.array(all_paths)

    save_spatial_cache(all_feats, labels_np, paths_np, backbone_id, dataset_name, tag)

    del extractor
    if device == 'cuda':
        torch.cuda.empty_cache()

    return all_feats, labels_np, paths_np
