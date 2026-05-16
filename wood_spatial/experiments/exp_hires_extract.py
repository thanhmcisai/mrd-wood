#!/usr/bin/env python3
"""
High-resolution feature extraction cache.

GPU stage: run live backbone inference at high resolution and save spatial
features. CPU-heavy clustering/metrics are intentionally left to
exp_hires_metrics_from_cache.py so GPU runtime is not wasted on KMeans/OpenCV.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from wood_spatial.config import BB_ORDER, V4_DIR
from wood_spatial.experiments.exp_hires_spatial_full import (
    DEVICE, IMG_H, IMG_W, N_CLUSTERS, apply_perturbation, condition_specs,
    correct_illumination, expand_names, extract_spatial, load_rgb, sample_paths,
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


def _cache_path(backbone: str, dataset: str, condition: str) -> Path:
    return HIRES_CACHE / backbone / dataset / f'{condition}.npz'


def _flatten_samples(samples: dict) -> tuple[list[str], list[str]]:
    labels = []
    paths = []
    for cls in sorted(samples.keys(), key=lambda x: str(x)):
        for path in samples[cls]:
            labels.append(str(cls))
            paths.append(str(path))
    return paths, labels


def _save_npz(path: Path, compressed: bool, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def extract_pair(
    backbone: str,
    dataset: str,
    specs: list,
    n_per_class: int,
    resume: bool = True,
    compressed: bool = False,
):
    samples = sample_paths(backbone, dataset, n_per_class)
    paths, labels = _flatten_samples(samples)
    if not paths:
        print(f'  {backbone}/{dataset}: no sampled paths')
        return

    print(f'  {backbone}/{dataset}: {len(set(labels))} classes, {len(paths)} images')

    clean_images = []
    valid_paths = []
    valid_labels = []
    for path, label in zip(paths, labels):
        raw = load_rgb(path)
        if raw is None:
            continue
        clean_images.append(correct_illumination(raw))
        valid_paths.append(path)
        valid_labels.append(label)

    paths_arr = np.asarray(valid_paths, dtype=object)
    labels_arr = np.asarray(valid_labels, dtype=object)

    for cond_label, pert_name, _tag, value in specs:
        out_path = _cache_path(backbone, dataset, cond_label)
        if resume and out_path.exists():
            print(f'    checkpoint hit: {backbone}/{dataset}/{cond_label}')
            continue

        t0 = time.time()
        features = []
        pooled = []
        for i, clean in enumerate(clean_images, start=1):
            img = apply_perturbation(clean, pert_name, value)
            feat = extract_spatial(backbone, img)
            features.append(feat.astype(np.float16))
            vec = feat.mean(axis=(0, 1)).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            pooled.append(vec)
            if i % 10 == 0 or i == len(clean_images):
                print(f'    {cond_label}: {i}/{len(clean_images)} images', flush=True)

        _save_npz(
            out_path,
            compressed,
            features=np.stack(features, axis=0),
            pooled=np.stack(pooled, axis=0),
            paths=paths_arr,
            labels=labels_arr,
            condition=np.asarray(cond_label),
            perturbation=np.asarray(pert_name),
            severity=np.asarray('' if value is None else str(value)),
            backbone=np.asarray(backbone),
            dataset=np.asarray(dataset),
            img_w=np.asarray(IMG_W),
            img_h=np.asarray(IMG_H),
            n_clusters=np.asarray(N_CLUSTERS),
        )
        print(f'    saved {out_path} [{(time.time() - t0) / 60:.1f} min]')


def main():
    parser = argparse.ArgumentParser(description='GPU stage: extract high-res spatial feature cache.')
    parser.add_argument('--datasets', nargs='+', default=os.environ.get('WOOD_HIRES_DATASETS', 'tier_a').split(','))
    parser.add_argument('--backbones', nargs='+', default=os.environ.get('WOOD_HIRES_BACKBONES', 'all').split(','))
    parser.add_argument('--conditions', choices=['representative', 'full'],
                        default=os.environ.get('WOOD_HIRES_CONDITIONS', 'representative'))
    parser.add_argument('--n-per-class', type=int, default=int(os.environ.get('WOOD_HIRES_N_PER_CLASS', '3')))
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--compressed', action='store_true',
                        default=os.environ.get('WOOD_HIRES_COMPRESS', '0') == '1')
    args = parser.parse_args()

    datasets = expand_names(args.datasets, 'dataset')
    backbones = expand_names(args.backbones, 'backbone')
    specs = condition_specs(args.conditions)

    print(f'Device: {DEVICE}')
    print(f'Image size: {IMG_W}x{IMG_H}; no PCA cache; k={N_CLUSTERS}')
    print(f'Datasets: {datasets}')
    print(f'Backbones: {backbones}')
    print(f'Conditions: {len(specs)} ({args.conditions})')
    print(f'N_PER_CLASS: {args.n_per_class}')
    print(f'Cache: {HIRES_CACHE}')

    for backbone in backbones:
        if backbone not in BB_ORDER:
            raise ValueError(f'Unknown backbone: {backbone}')
        t0 = time.time()
        for dataset in datasets:
            extract_pair(
                backbone,
                dataset,
                specs,
                args.n_per_class,
                resume=not args.no_resume,
                compressed=args.compressed,
            )
        print(f'Finished {backbone} in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
