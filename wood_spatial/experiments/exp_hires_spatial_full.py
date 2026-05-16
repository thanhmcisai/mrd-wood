#!/usr/bin/env python3
"""
High-resolution no-PCA spatial metrics.

This experiment runs live backbone inference at high resolution and clusters
normalized spatial tokens directly, without PCA. It is intentionally separate
from Exp2/Exp3 cache-based metrics because it is much more expensive and is the
strongest setting for reporting spatial/CAM claims.
"""
import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from scipy.ndimage import label as ndlabel
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans

from wood_spatial.config import (
    ALL_DATASETS, BB_ORDER, DATASET_ROOT, N_CLUSTERS, PERTURB_CONFIGS,
    TIER_A, TIER_B, TIER_C, V4_CSV, V4_DIR, V4_FEAT_CACHE,
)
from wood_spatial.core.perturbations import cache_tag_for


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_W = int(os.environ.get('WOOD_HIRES_IMG_W', '1280'))
IMG_H = int(os.environ.get('WOOD_HIRES_IMG_H', '1024'))
DINO_SIZE = int(os.environ.get('WOOD_HIRES_DINO_SIZE', '518'))
N_FIT = int(os.environ.get('WOOD_HIRES_N_FIT', '2000'))
N_INIT = int(os.environ.get('WOOD_HIRES_N_INIT', '10'))
GF_R = int(os.environ.get('WOOD_HIRES_GF_R', '24'))
GF_EPS = float(os.environ.get('WOOD_HIRES_GF_EPS', '1e-4'))
RANDOM_SEED = 42

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

BB_MODEL_IDS = {
    'resnet50': 'resnet50',
    'efficientnet_b3': 'efficientnet_b3',
    'convnext_tiny': 'convnext_tiny',
    'swin_tiny': 'swin_tiny_patch4_window7_224',
    'dinov2_b': 'vit_base_patch14_dinov2.lvd142m',
    'hrnet32': 'hrnet_w32',
    'mobilenetv3': 'mobilenetv3_large_100',
}

HIRES_FEATURE_CACHE = V4_DIR / 'hires_feature_cache'


def _safe_name(*parts) -> str:
    raw = '__'.join(str(p) for p in parts)
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in raw)


def _checkpoint_path(backbone: str, dataset: str) -> Path:
    d = V4_CSV / 'exp_hires_spatial_checkpoints'
    d.mkdir(parents=True, exist_ok=True)
    return d / f'{_safe_name(backbone, dataset)}.csv'


def _condition_checkpoint_path(backbone: str, dataset: str, condition: str) -> Path:
    d = V4_CSV / 'exp_hires_stream_checkpoints'
    d.mkdir(parents=True, exist_ok=True)
    return d / f'{_safe_name(backbone, dataset, condition)}.csv'


def _hires_cache_path(backbone: str, dataset: str, condition: str) -> Path:
    return HIRES_FEATURE_CACHE / backbone / dataset / f'{condition}.npz'


def _load_hires_cache(backbone: str, dataset: str, condition: str):
    path = _hires_cache_path(backbone, dataset, condition)
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def _cache_maps(cache):
    if cache is None:
        return {}, {}
    paths = [str(p) for p in cache['paths']]
    feats = cache['features']
    pooled = cache['pooled'] if 'pooled' in cache.files else None
    feat_map = {p: feats[i].astype(np.float32) for i, p in enumerate(paths)}
    pooled_map = {}
    if pooled is not None:
        pooled_map = {p: pooled[i].astype(np.float32) for i, p in enumerate(paths)}
    return feat_map, pooled_map


def _centroids_from_cache(cache) -> dict:
    labels = cache['labels'].astype(str)
    pooled = cache['pooled'].astype(np.float32)
    centroids = {}
    for label in np.unique(labels):
        rows = pooled[labels == label]
        cent = rows.mean(axis=0)
        centroids[label] = cent / (np.linalg.norm(cent) + 1e-8)
    return centroids


def correct_illumination(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    bg = cv2.GaussianBlur(l, (0, 0), sigmaX=100, sigmaY=100)
    l_flat = np.clip(l.astype(np.float32) - bg.astype(np.float32) + np.mean(bg), 0, 255)
    l_flat = l_flat.astype(np.uint8)
    l_enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_flat)
    return cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2RGB)


def guided_filter(guide: np.ndarray, src: np.ndarray) -> np.ndarray:
    k = (2 * GF_R + 1, 2 * GF_R + 1)
    mean_i = cv2.boxFilter(guide, cv2.CV_32F, k)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, k)
    mean_ip = cv2.boxFilter(guide * src, cv2.CV_32F, k)
    mean_ii = cv2.boxFilter(guide * guide, cv2.CV_32F, k)
    a = (mean_ip - mean_i * mean_p) / (mean_ii - mean_i * mean_i + GF_EPS)
    b = mean_p - a * mean_i
    return cv2.boxFilter(a, cv2.CV_32F, k) * guide + cv2.boxFilter(b, cv2.CV_32F, k)


def _disk_kernel(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x * x + y * y <= radius * radius
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    kernel[mask] = 1.0
    return kernel / max(kernel.sum(), 1e-8)


def apply_perturbation(img: np.ndarray, name: str, value) -> np.ndarray:
    if name == 'clean':
        return img
    if name == 'gaussian_blur':
        r = int(value)
        return cv2.GaussianBlur(img, (2 * r + 1, 2 * r + 1), sigmaX=r)
    if name == 'defocus_blur':
        return cv2.filter2D(img, -1, _disk_kernel(int(value))).clip(0, 255).astype(np.uint8)
    if name == 'resize':
        factor = float(value)
        h, w = img.shape[:2]
        ch = max(1, int(h / factor))
        cw = max(1, int(w / factor))
        top = (h - ch) // 2
        left = (w - cw) // 2
        return cv2.resize(img[top:top + ch, left:left + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    if name == 'illumination':
        return np.clip(img.astype(np.float32) * float(value), 0, 255).astype(np.uint8)
    if name == 'jpeg':
        _, buf = cv2.imencode(
            '.jpg',
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, int(value)],
        )
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if name in {'color_shift', 'red_channel_shift', 'green_channel_shift', 'blue_channel_shift'}:
        channel = {
            'color_shift': 0,
            'red_channel_shift': 0,
            'green_channel_shift': 1,
            'blue_channel_shift': 2,
        }[name]
        out = img.astype(np.int16).copy()
        out[:, :, channel] = np.clip(out[:, :, channel] + int(value), 0, 255)
        return out.astype(np.uint8)
    if name == 'gaussian_noise':
        noise = np.random.normal(0.0, float(value) * 255.0, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if name == 'shot_noise':
        level = max(float(value), 1.0)
        scaled = img.astype(np.float32) / 255.0
        noisy = np.random.poisson(scaled * level) / level
        return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
    if name == 'impulse_noise':
        p = float(value)
        out = img.copy()
        rnd = np.random.rand(*img.shape[:2])
        out[rnd < p / 2.0] = 0
        out[(rnd >= p / 2.0) & (rnd < p)] = 255
        return out
    if name == 'motion_blur':
        k = int(value)
        if k % 2 == 0:
            k += 1
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        return cv2.filter2D(img, -1, kernel).clip(0, 255).astype(np.uint8)
    if name == 'zoom_blur':
        h, w = img.shape[:2]
        acc = img.astype(np.float32)
        for z in np.linspace(1.0, float(value), 5)[1:]:
            ch = max(1, int(h / z))
            cw = max(1, int(w / z))
            top = (h - ch) // 2
            left = (w - cw) // 2
            acc += cv2.resize(img[top:top + ch, left:left + cw], (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(acc / 5.0, 0, 255).astype(np.uint8)
    if name == 'contrast':
        mean = img.astype(np.float32).mean(axis=(0, 1), keepdims=True)
        return np.clip((img.astype(np.float32) - mean) * float(value) + mean, 0, 255).astype(np.uint8)
    if name == 'pixelate':
        h, w = img.shape[:2]
        ratio = float(value)
        ph = max(1, int(h * ratio))
        pw = max(1, int(w * ratio))
        small = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if name == 'scratch':
        params = {
            'mild': (3, 1),
            'moderate': (6, 2),
            'severe': (12, 3),
        }
        n_lines, thickness = params[value]
        seed = int(abs(float(img[::16, ::16].sum())) * 1000003) % (2**32 - 1)
        rng = np.random.RandomState(seed)
        out = img.copy()
        h, w = out.shape[:2]
        for _ in range(n_lines):
            x1 = int(rng.randint(0, w))
            y1 = int(rng.randint(0, h))
            length = int(rng.randint(max(8, w // 8), max(9, w // 2)))
            angle = float(rng.uniform(-np.pi, np.pi))
            x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
            y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
            gray = int(rng.choice([35, 220]))
            cv2.line(out, (x1, y1), (x2, y2), (gray, gray, gray), thickness, lineType=cv2.LINE_AA)
        return out
    if name == 'rotation':
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(value), 1.0)
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if name == 'compound':
        presets = {
            'mild': [('gaussian_blur', 4), ('illumination', 0.85), ('jpeg', 50)],
            'moderate': [('gaussian_blur', 8), ('illumination', 0.70), ('jpeg', 30)],
            'severe': [('gaussian_blur', 12), ('illumination', 0.70), ('jpeg', 10)],
        }
        out = img
        for sub_name, sub_value in presets[value]:
            out = apply_perturbation(out, sub_name, sub_value)
        return out
    if name in {'compound_optical', 'compound_digital', 'compound_field'}:
        presets = {
            'compound_optical': {
                'mild': [('defocus_blur', 3), ('motion_blur', 5), ('illumination', 0.85)],
                'moderate': [('defocus_blur', 8), ('motion_blur', 9), ('illumination', 0.70)],
                'severe': [('defocus_blur', 15), ('motion_blur', 15), ('illumination', 0.70)],
            },
            'compound_digital': {
                'mild': [('contrast', 0.75), ('pixelate', 0.50), ('jpeg', 70)],
                'moderate': [('contrast', 0.50), ('pixelate', 0.33), ('jpeg', 30)],
                'severe': [('contrast', 1.50), ('pixelate', 0.25), ('jpeg', 10)],
            },
            'compound_field': {
                'mild': [('gaussian_noise', 0.02), ('blue_channel_shift', 30), ('zoom_blur', 1.05), ('jpeg', 70)],
                'moderate': [('gaussian_noise', 0.05), ('green_channel_shift', -30), ('zoom_blur', 1.10), ('jpeg', 30)],
                'severe': [('gaussian_noise', 0.10), ('blue_channel_shift', -45), ('zoom_blur', 1.20), ('jpeg', 10)],
            },
        }
        out = img
        for sub_name, sub_value in presets[name][value]:
            out = apply_perturbation(out, sub_name, sub_value)
        return out
    raise ValueError(f'Unknown perturbation: {name}/{value}')


def condition_specs(mode: str) -> list:
    specs = [('clean', 'clean', 'clean', None)]
    if mode == 'representative':
        values = [
            ('gaussian_blur', 4), ('gaussian_blur', 8), ('gaussian_blur', 12),
            ('defocus_blur', 5), ('defocus_blur', 11),
            ('resize', 2.0), ('jpeg', 10), ('jpeg', 50),
            ('rotation', 45), ('rotation', 180),
            ('red_channel_shift', -45), ('red_channel_shift', 45),
            ('green_channel_shift', -45), ('green_channel_shift', 45),
            ('blue_channel_shift', -45), ('blue_channel_shift', 45),
            ('gaussian_noise', 0.10), ('shot_noise', 15), ('impulse_noise', 0.05),
            ('motion_blur', 15), ('zoom_blur', 1.20),
            ('contrast', 0.50), ('contrast', 1.50), ('pixelate', 0.25),
            ('scratch', 'severe'),
            ('compound', 'mild'), ('compound', 'severe'),
            ('compound_optical', 'severe'),
            ('compound_digital', 'severe'),
            ('compound_field', 'severe'),
        ]
    elif mode == 'full':
        values = [
            (name, value)
            for name, cfg in PERTURB_CONFIGS.items()
            for value in cfg['values']
        ]
    else:
        raise ValueError(f'Unknown condition mode: {mode}')

    for name, value in values:
        tag = _condition_tag(name, value)
        specs.append((tag, name, tag, value))
    return specs


def _condition_tag(name: str, value) -> str:
    return cache_tag_for(name, value)


_MODELS = {}


def get_model(backbone: str):
    if backbone in _MODELS:
        return _MODELS[backbone]
    model_id = BB_MODEL_IDS[backbone]
    if backbone == 'dinov2_b':
        model = timm.create_model(model_id, pretrained=True, img_size=DINO_SIZE).to(DEVICE).eval()
    elif backbone == 'swin_tiny':
        model = timm.create_model(
            model_id, pretrained=True, img_size=(IMG_H, IMG_W),
            features_only=True, out_indices=(2,),
        ).to(DEVICE).eval()
    else:
        model = timm.create_model(
            model_id, pretrained=True, features_only=True, out_indices=(2,),
        ).to(DEVICE).eval()
    _MODELS[backbone] = model
    return model


def extract_spatial(backbone: str, img_rgb: np.ndarray) -> np.ndarray:
    model = get_model(backbone)
    if backbone == 'dinov2_b':
        resized = cv2.resize(img_rgb, (DINO_SIZE, DINO_SIZE))
        tensor = TRANSFORM(resized).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model.forward_features(tensor)[:, 1:, :]
            hw = int(feat.shape[1] ** 0.5)
            return feat[0].reshape(hw, hw, -1).float().cpu().numpy()

    tensor = TRANSFORM(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model(tensor)[0].squeeze(0).float().cpu().numpy()
    if backbone == 'swin_tiny':
        return feat
    return np.transpose(feat, (1, 2, 0))


def cluster_map(feat_hwc: np.ndarray, guide: np.ndarray) -> np.ndarray:
    h_f, w_f, c = feat_hwc.shape
    x = feat_hwc.reshape(-1, c)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    rng = np.random.RandomState(RANDOM_SEED)
    fit_n = min(N_FIT, len(x))
    fit_idx = rng.choice(len(x), fit_n, replace=False)
    km = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        n_init=N_INIT,
        max_iter=100,
    ).fit(x[fit_idx])

    dists = km.transform(x)
    probs = np.exp(-np.clip(dists, 0, 500))
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-30)
    probs_up = cv2.resize(
        probs.reshape(h_f, w_f, N_CLUSTERS),
        (guide.shape[1], guide.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    refined = np.stack([
        guided_filter(guide, probs_up[:, :, k].astype(np.float32))
        for k in range(N_CLUSTERS)
    ], axis=-1)
    labels = np.argmax(refined, axis=-1)

    means = [guide[labels == k].mean() if np.any(labels == k) else 0.0 for k in range(N_CLUSTERS)]
    order = np.argsort(means)
    mapped = np.zeros_like(labels)
    for new_idx, old_idx in enumerate(order):
        mapped[labels == old_idx] = new_idx
    return _smart_hole_filling(mapped)


def _smart_hole_filling(labels: np.ndarray) -> np.ndarray:
    result = labels.copy()
    vessel = (labels == 0).astype(np.uint8)
    contours, _ = cv2.findContours(vessel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(vessel)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(filled, [cnt], -1, 1, thickness=cv2.FILLED)
    result[filled == 1] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fiber_id = N_CLUSTERS - 1
    for tissue_id in range(1, N_CLUSTERS):
        mask = (result == tissue_id).astype(np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        result[(result == tissue_id) & (filled == 0)] = fiber_id
        result[cleaned == 1] = tissue_id
    return result


def sgi(labels: np.ndarray) -> float:
    return float(sum(ndlabel(labels == k)[1] for k in range(N_CLUSTERS)) / labels.size)


def csi(clean_labels: np.ndarray, pert_labels: np.ndarray) -> float:
    vals = []
    for k in range(N_CLUSTERS):
        a = clean_labels == k
        b = pert_labels == k
        union = np.logical_or(a, b).sum()
        vals.append(np.logical_and(a, b).sum() / union if union > 0 else 1.0)
    return float(np.mean(vals))


def cam_distribution(feat_hwc: np.ndarray, centroid: np.ndarray, labels: np.ndarray) -> np.ndarray:
    h_f, w_f, c = feat_hwc.shape
    cent = centroid / (np.linalg.norm(centroid) + 1e-8)
    scores = np.clip(feat_hwc.reshape(-1, c) @ cent, 0, None).reshape(h_f, w_f)
    cam = cv2.resize(scores.astype(np.float32), (labels.shape[1], labels.shape[0]))
    total = cam.sum() + 1e-8
    return np.array([cam[labels == k].sum() / total for k in range(N_CLUSTERS)])


def entropy(dist: np.ndarray) -> float:
    d = dist.astype(np.float64) + 1e-10
    d /= d.sum()
    return float(-np.sum(d * np.log2(d)))


def js_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a + 1e-10
    b = b + 1e-10
    return float(jensenshannon(a / a.sum(), b / b.sum()))


def load_rgb(path: str) -> np.ndarray | None:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (IMG_W, IMG_H))


def sample_paths(backbone: str, dataset: str, n_per_class: int) -> dict:
    cache = V4_FEAT_CACHE / f'{backbone}_{dataset}_original.npz'
    if not cache.exists():
        print(f'  Missing global cache: {cache}')
        return {}
    data = np.load(cache, allow_pickle=True)
    paths = data['paths']
    labels = data['labels']
    by_class = defaultdict(list)
    rng = np.random.RandomState(RANDOM_SEED)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if n_per_class <= 0:
            chosen = idx
        else:
            chosen = rng.choice(idx, min(n_per_class, len(idx)), replace=False)
        by_class[cls].extend(str(paths[i]) for i in chosen)
    return dict(by_class)


def run_pair(
    backbone: str,
    dataset: str,
    specs: list,
    n_per_class: int,
    resume: bool = True,
) -> None:
    clean_cache = _load_hires_cache(backbone, dataset, 'clean')
    cls_paths = sample_paths(backbone, dataset, n_per_class)
    if not cls_paths:
        if clean_cache is None:
            return
        cls_paths = defaultdict(list)
        for label, path in zip(clean_cache['labels'].astype(str), clean_cache['paths']):
            cls_paths[str(label)].append(str(path))

    print(f'  {backbone}/{dataset}: {len(cls_paths)} classes, '
          f'{sum(len(v) for v in cls_paths.values())} images')

    clean_feat_cache, clean_pool_cache = _cache_maps(clean_cache)
    if clean_cache is not None:
        print(f'    using high-res feature cache: clean ({len(clean_feat_cache)} images)')

    samples = []
    centroids = {}
    for cls, paths in cls_paths.items():
        pooled = []
        for path in paths:
            raw = load_rgb(path)
            if raw is None:
                continue
            corr = correct_illumination(raw)
            if str(path) in clean_pool_cache:
                clean_pool = clean_pool_cache[str(path)]
                clean_pool = clean_pool / (np.linalg.norm(clean_pool) + 1e-8)
            else:
                feat = clean_feat_cache.get(str(path))
                if feat is None:
                    feat = extract_spatial(backbone, corr)
                vec = feat.mean(axis=(0, 1))
                clean_pool = vec / (np.linalg.norm(vec) + 1e-8)
            pooled.append(clean_pool)
            samples.append({
                'class': str(cls),
                'path': str(path),
                'image': Path(path).name,
                'clean': corr,
                'clean_pool': clean_pool,
            })
        if pooled:
            cent = np.mean(pooled, axis=0)
            centroids[str(cls)] = cent / (np.linalg.norm(cent) + 1e-8)

    if clean_cache is not None and 'pooled' in clean_cache.files:
        cached_centroids = _centroids_from_cache(clean_cache)
        # Use cached centroids when dimensions match the sampled clean vectors.
        if samples and cached_centroids:
            dim = samples[0]['clean_pool'].shape[0]
            if all(c.shape[0] == dim for c in cached_centroids.values()):
                centroids = cached_centroids

    clean_refs = {}
    for idx, sample in enumerate(samples, start=1):
        cls = sample['class']
        centroid = centroids.get(cls)
        if centroid is None:
            continue
        guide = cv2.cvtColor(sample['clean'], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        feat = clean_feat_cache.get(sample['path'])
        if feat is None:
            feat = extract_spatial(backbone, sample['clean'])
        labels = cluster_map(feat, guide)
        dist = cam_distribution(feat, centroid, labels)
        clean_refs[sample['path']] = {
            'labels': labels,
            'dist': dist,
            'entropy': entropy(dist),
        }
        if idx % 10 == 0 or idx == len(samples):
            print(f'    clean reference clusters: {idx}/{len(samples)}', flush=True)

    for cond_label, pert_name, _tag, value in specs:
        ckpt = _condition_checkpoint_path(backbone, dataset, cond_label)
        if ckpt.exists() and resume:
            print(f'    checkpoint hit: {ckpt.name}')
            continue

        cond_cache = clean_cache if cond_label == 'clean' else _load_hires_cache(backbone, dataset, cond_label)
        cond_feat_cache, cond_pool_cache = _cache_maps(cond_cache)
        if cond_cache is not None:
            print(f'    using high-res feature cache: {cond_label} ({len(cond_feat_cache)} images)')

        rows = []
        t0 = time.time()
        for idx, sample in enumerate(samples, start=1):
            ref = clean_refs.get(sample['path'])
            centroid = centroids.get(sample['class'])
            if ref is None or centroid is None:
                continue

            if cond_label == 'clean':
                labels = ref['labels']
                pooled = sample['clean_pool']
                dist = ref['dist']
            else:
                img = apply_perturbation(sample['clean'], pert_name, value)
                guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                feat = cond_feat_cache.get(sample['path'])
                if feat is None:
                    feat = extract_spatial(backbone, img)
                labels = cluster_map(feat, guide)
                pooled = cond_pool_cache.get(sample['path'])
                if pooled is None:
                    pooled = feat.mean(axis=(0, 1))
                pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
                dist = cam_distribution(feat, centroid, labels)

            h = entropy(dist)
            js = js_distance(ref['dist'], dist)
            rows.append({
                'dataset': dataset,
                'backbone': backbone,
                'class': sample['class'],
                'image': sample['image'],
                'condition': cond_label,
                'perturbation': 'clean' if cond_label == 'clean' else pert_name,
                'severity': '' if cond_label == 'clean' else value,
                'sgi': sgi(labels),
                'csi': csi(ref['labels'], labels),
                'feature_drift': float(1 - np.dot(sample['clean_pool'], pooled)),
                'cam_entropy': h,
                'delta_entropy': h - ref['entropy'],
                'cam_js_distance': js,
                'cam_js_divergence': js ** 2,
            })
            if idx % 10 == 0 or idx == len(samples):
                print(f'    {cond_label}: {idx}/{len(samples)} images', flush=True)

        pd.DataFrame(rows).to_csv(ckpt, index=False)
        print(f'    saved {ckpt.name} ({len(rows)} rows, {(time.time() - t0) / 60:.1f} min)')


def expand_names(values: list[str], kind: str) -> list[str]:
    if not values:
        return []
    expanded = []
    for v in values:
        if kind == 'dataset':
            groups = {
                'all': TIER_A + TIER_B + TIER_C,
                'tier_a': TIER_A,
                'tier_b': TIER_B,
                'tier_c': TIER_C,
                'vn26': TIER_C,
            }
            expanded.extend(groups.get(v, [v]))
        elif kind == 'backbone':
            expanded.extend(BB_ORDER if v == 'all' else [v])
    seen = []
    for item in expanded:
        if item not in seen:
            seen.append(item)
    return seen


def finalize(datasets: list[str], backbones: list[str]):
    frames = []
    for bb in backbones:
        for ds in datasets:
            pattern = f'{_safe_name(bb, ds)}__*.csv'
            for p in sorted((V4_CSV / 'exp_hires_stream_checkpoints').glob(pattern)):
                frames.append(pd.read_csv(p))
            old_p = _checkpoint_path(bb, ds)
            if old_p.exists():
                frames.append(pd.read_csv(old_p))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(V4_CSV / 'exp_hires_spatial_metrics.csv', index=False)
    print(f'\nSaved {len(out)} rows -> {V4_CSV / "exp_hires_spatial_metrics.csv"}')


def main():
    parser = argparse.ArgumentParser(description='Run high-res no-PCA spatial metrics.')
    parser.add_argument('--datasets', nargs='+', default=os.environ.get('WOOD_HIRES_DATASETS', 'tier_a').split(','))
    parser.add_argument('--backbones', nargs='+', default=os.environ.get('WOOD_HIRES_BACKBONES', 'all').split(','))
    parser.add_argument('--conditions', choices=['representative', 'full'],
                        default=os.environ.get('WOOD_HIRES_CONDITIONS', 'full'))
    parser.add_argument('--n-per-class', type=int, default=int(os.environ.get('WOOD_HIRES_N_PER_CLASS', '3')),
                        help='Images per class; use 0 to process every image in each dataset.')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--finalize-only', action='store_true')
    args = parser.parse_args()

    datasets = expand_names(args.datasets, 'dataset')
    backbones = expand_names(args.backbones, 'backbone')
    specs = condition_specs(args.conditions)

    print(f'Device: {DEVICE}')
    print(f'Image size: {IMG_W}x{IMG_H}; DINO size: {DINO_SIZE}; no PCA; k={N_CLUSTERS}')
    print(f'Datasets: {datasets}')
    print(f'Backbones: {backbones}')
    print(f'Conditions: {len(specs)} ({args.conditions})')
    print(f'N_PER_CLASS: {args.n_per_class}')

    if not args.finalize_only:
        for bb in backbones:
            t0 = time.time()
            for ds in datasets:
                run_pair(bb, ds, specs, args.n_per_class, resume=not args.no_resume)
            print(f'Finished {bb} in {(time.time() - t0) / 60:.1f} min')

    finalize(datasets, backbones)


if __name__ == '__main__':
    main()
