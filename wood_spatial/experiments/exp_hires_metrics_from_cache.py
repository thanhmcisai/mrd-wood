#!/usr/bin/env python3
"""
High-resolution no-PCA spatial metrics from cached high-res features.

CPU stage: read features produced by exp_hires_extract.py and run KMeans,
guided filtering, CSI/SGI/CAM metrics without requiring GPU runtime.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from wood_spatial.config import BB_ORDER, N_CLUSTERS, V4_CSV, V4_DIR
from wood_spatial.experiments.exp_hires_spatial_full import (
    IMG_H, IMG_W, apply_perturbation, cam_distribution, cluster_map,
    condition_specs, correct_illumination, csi, entropy, expand_names,
    js_distance, load_rgb, sample_paths, sgi,
)


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)

HIRES_CACHE = V4_DIR / 'hires_feature_cache'
CKPT_DIR = V4_CSV / 'exp_hires_cache_metrics_checkpoints'


def _safe_name(*parts) -> str:
    raw = '__'.join(str(p) for p in parts)
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in raw)


def _cache_path(backbone: str, dataset: str, condition: str) -> Path:
    return HIRES_CACHE / backbone / dataset / f'{condition}.npz'


def _checkpoint_path(backbone: str, dataset: str) -> Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    return CKPT_DIR / f'{_safe_name(backbone, dataset)}.csv'


def _flatten_samples(samples: dict) -> tuple[list[str], list[str]]:
    labels = []
    paths = []
    for cls in sorted(samples.keys(), key=lambda x: str(x)):
        for path in samples[cls]:
            labels.append(str(cls))
            paths.append(str(path))
    return paths, labels


def _load_condition(backbone: str, dataset: str, condition: str):
    path = _cache_path(backbone, dataset, condition)
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def _class_centroids(clean_cache) -> dict:
    labels = clean_cache['labels'].astype(str)
    pooled = clean_cache['pooled'].astype(np.float32)
    centroids = {}
    for label in np.unique(labels):
        rows = pooled[labels == label]
        cent = rows.mean(axis=0)
        centroids[label] = cent / (np.linalg.norm(cent) + 1e-8)
    return centroids


def _load_clean_images(paths: list[str]) -> dict[str, np.ndarray]:
    images = {}
    for path in paths:
        raw = load_rgb(path)
        if raw is None:
            continue
        images[path] = correct_illumination(raw)
    return images


def run_pair(backbone: str, dataset: str, specs: list, n_per_class: int) -> pd.DataFrame:
    clean_cache = _load_condition(backbone, dataset, 'clean')
    if clean_cache is None:
        print(f'  Missing clean cache: {backbone}/{dataset}')
        return pd.DataFrame()

    paths = [str(p) for p in clean_cache['paths']]
    labels = clean_cache['labels'].astype(str)
    print(f'  {backbone}/{dataset}: {len(set(labels))} classes, {len(paths)} images')

    clean_images = _load_clean_images(paths)
    centroids = _class_centroids(clean_cache)
    clean_features = clean_cache['features'].astype(np.float32)
    clean_pooled = clean_cache['pooled'].astype(np.float32)

    clean_labels_by_image = []
    clean_dist_by_image = []
    for i, path in enumerate(paths):
        clean = clean_images.get(path)
        if clean is None:
            clean_labels_by_image.append(None)
            clean_dist_by_image.append(None)
            continue
        guide = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        feat = clean_features[i]
        labels_map = cluster_map(feat, guide)
        clean_labels_by_image.append(labels_map)
        clean_dist_by_image.append(cam_distribution(feat, centroids[labels[i]], labels_map))
        if (i + 1) % 10 == 0 or i + 1 == len(paths):
            print(f'    clean clusters: {i + 1}/{len(paths)}', flush=True)

    rows = []
    for cond_label, pert_name, _tag, value in specs:
        cond_cache = _load_condition(backbone, dataset, cond_label)
        if cond_cache is None:
            print(f'    missing condition cache: {cond_label}')
            continue

        cond_features = cond_cache['features'].astype(np.float32)
        cond_pooled = cond_cache['pooled'].astype(np.float32)
        for i, path in enumerate(paths):
            clean = clean_images.get(path)
            labels_clean = clean_labels_by_image[i]
            dist_clean = clean_dist_by_image[i]
            if clean is None or labels_clean is None or dist_clean is None:
                continue

            img = apply_perturbation(clean, pert_name, value)
            guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            feat = cond_features[i]
            labels_cond = cluster_map(feat, guide)
            dist = cam_distribution(feat, centroids[labels[i]], labels_cond)
            h = entropy(dist)
            h_clean = entropy(dist_clean)

            rows.append({
                'dataset': dataset,
                'backbone': backbone,
                'class': labels[i],
                'image': Path(path).name,
                'condition': cond_label,
                'perturbation': 'clean' if cond_label == 'clean' else pert_name,
                'severity': '' if cond_label == 'clean' else value,
                'sgi': sgi(labels_cond),
                'csi': csi(labels_clean, labels_cond),
                'feature_drift': float(1 - np.dot(clean_pooled[i], cond_pooled[i])),
                'cam_entropy': h,
                'delta_entropy': h - h_clean,
                'cam_js_distance': js_distance(dist_clean, dist),
                'cam_js_divergence': js_distance(dist_clean, dist) ** 2,
            })
            if (i + 1) % 10 == 0 or i + 1 == len(paths):
                print(f'    {cond_label}: {i + 1}/{len(paths)} images', flush=True)

    return pd.DataFrame(rows)


def finalize(datasets: list[str], backbones: list[str]):
    frames = []
    for backbone in backbones:
        for dataset in datasets:
            path = _checkpoint_path(backbone, dataset)
            if path.exists():
                frames.append(pd.read_csv(path))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(V4_CSV / 'exp_hires_spatial_metrics.csv', index=False)
    print(f'\nSaved {len(out)} rows -> {V4_CSV / "exp_hires_spatial_metrics.csv"}')


def main():
    parser = argparse.ArgumentParser(description='CPU stage: high-res metrics from feature cache.')
    parser.add_argument('--datasets', nargs='+', default=os.environ.get('WOOD_HIRES_DATASETS', 'tier_a').split(','))
    parser.add_argument('--backbones', nargs='+', default=os.environ.get('WOOD_HIRES_BACKBONES', 'all').split(','))
    parser.add_argument('--conditions', choices=['representative', 'full'],
                        default=os.environ.get('WOOD_HIRES_CONDITIONS', 'representative'))
    parser.add_argument('--n-per-class', type=int, default=int(os.environ.get('WOOD_HIRES_N_PER_CLASS', '3')))
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--finalize-only', action='store_true')
    args = parser.parse_args()

    datasets = expand_names(args.datasets, 'dataset')
    backbones = expand_names(args.backbones, 'backbone')
    specs = condition_specs(args.conditions)

    print(f'CPU metrics from cache: {HIRES_CACHE}')
    print(f'Image size: {IMG_W}x{IMG_H}; no PCA; k={N_CLUSTERS}')
    print(f'Datasets: {datasets}')
    print(f'Backbones: {backbones}')
    print(f'Conditions: {len(specs)} ({args.conditions})')

    if not args.finalize_only:
        for backbone in backbones:
            if backbone not in BB_ORDER:
                raise ValueError(f'Unknown backbone: {backbone}')
            t0 = time.time()
            for dataset in datasets:
                ckpt = _checkpoint_path(backbone, dataset)
                if ckpt.exists() and not args.no_resume:
                    print(f'  Checkpoint hit: {ckpt.name}')
                    continue
                df = run_pair(backbone, dataset, specs, args.n_per_class)
                df.to_csv(ckpt, index=False)
                print(f'  Saved checkpoint: {ckpt.name} ({len(df)} rows)')
            print(f'Finished {backbone} in {(time.time() - t0) / 60:.1f} min')

    finalize(datasets, backbones)


if __name__ == '__main__':
    main()
