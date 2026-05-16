"""
Wood Spatial — Feature Extraction Orchestrator
=================================================
Extract global and spatial features for all (backbone, dataset, perturbation) combos.

This script should be run first (Phase 2) before any experiment.
It populates results_v4/feature_cache/ and results_v4/spatial_cache/.

Usage:
    python -m wood_spatial.experiments.extract_features --mode global
    python -m wood_spatial.experiments.extract_features --mode spatial
    python -m wood_spatial.experiments.extract_features --mode all
"""
import argparse
import logging
import time

from wood_spatial.config import (
    BACKBONE_CONFIGS, PERTURB_CONFIGS, ALL_DATASETS,
    TIER_A, TIER_B, TIER_C, BB_ORDER,
    SPATIAL_MAX_IMAGES,
)
from wood_spatial.core.cache import (
    extract_and_save, cache_exists,
    extract_spatial_and_save,
)
from wood_spatial.core.perturbations import make_perturbation, cache_tag_for

logger = logging.getLogger(__name__)


def _coerce_value(text: str):
    try:
        if any(ch in text for ch in ('.', 'e', 'E')):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_spatial_perturbations(items):
    if items is None:
        return None
    if len(items) == 1 and str(items[0]).lower() in {'none', 'empty', '[]'}:
        return []
    specs = []
    for item in items:
        if ':' not in str(item):
            raise ValueError(f'Expected spatial perturbation as name:value, got {item!r}')
        name, value = str(item).split(':', 1)
        specs.append((name, _coerce_value(value)))
    return specs


def extract_global_features(
    datasets: list = None,
    backbones: list = None,
    clean_datasets: list = None,
    perturbation_datasets: list = None,
    force: bool = False,
):
    """
    Extract global features for all combos.

    1. Clean features for all datasets × all backbones
    2. Perturbed features for Tier-A datasets × all perturbation × all severity
    """
    if datasets is None:
        datasets = TIER_A + TIER_B + TIER_C
    if backbones is None:
        backbones = BB_ORDER
    if clean_datasets is None:
        clean_datasets = datasets
    if perturbation_datasets is None:
        perturbation_datasets = [d for d in datasets if d in TIER_A]

    total = 0
    skipped = 0

    # Clean features
    logger.info('=== Phase 1: Clean features ===')
    for ds_name in clean_datasets:
        ds_info = ALL_DATASETS[ds_name]
        for bb in backbones:
            if not force and cache_exists(bb, ds_name, 'original'):
                skipped += 1
                continue
            logger.info('Extracting: %s × %s (clean)', bb, ds_name)
            extract_and_save(
                bb, ds_info['root'], ds_name,
                tag='original', force=force,
            )
            total += 1

    # Perturbed features (Tier-A only)
    logger.info('=== Phase 2: Perturbed features ===')
    for ds_name in perturbation_datasets:
        ds_info = ALL_DATASETS[ds_name]
        for pert_name, pcfg in PERTURB_CONFIGS.items():
            for value in pcfg['values']:
                tag = cache_tag_for(pert_name, value)
                for bb in backbones:
                    if not force and cache_exists(bb, ds_name, tag):
                        skipped += 1
                        continue
                    img_size = BACKBONE_CONFIGS[bb].get('img_size')
                    pert = make_perturbation(pcfg, value, img_size=img_size)
                    logger.info('Extracting: %s × %s × %s', bb, ds_name, tag)
                    extract_and_save(
                        bb, ds_info['root'], ds_name,
                        perturbation=pert, tag=tag, force=force,
                    )
                    total += 1

    logger.info('Global extraction done: %d new, %d skipped', total, skipped)


def extract_spatial_features(
    datasets: list = None,
    backbones: list = None,
    perturbation_subset: list = None,
    force: bool = False,
):
    """
    Extract spatial features for clustering analysis.

    Spatial features are larger (C×H×W per image), so we limit to SPATIAL_MAX_IMAGES.

    Parameters
    ----------
    perturbation_subset : list of (pert_name, value) tuples for perturbed spatial features.
        If None, extracts only clean + default perturbation subset.
    """
    if datasets is None:
        datasets = TIER_A
    if backbones is None:
        backbones = BB_ORDER

    # Default perturbation subset for spatial analysis
    if perturbation_subset is None:
        perturbation_subset = [
            ('gaussian_blur', 4), ('gaussian_blur', 8), ('gaussian_blur', 12),
            ('defocus_blur', 5), ('defocus_blur', 11),
            ('resize', 2.00),
            ('jpeg', 10), ('jpeg', 50),
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

    total = 0

    # Clean spatial
    logger.info('=== Spatial: Clean features ===')
    for ds_name in datasets:
        ds_info = ALL_DATASETS[ds_name]
        for bb in backbones:
            logger.info('Spatial: %s × %s (clean)', bb, ds_name)
            extract_spatial_and_save(
                bb, ds_info['root'], ds_name,
                tag='original', max_images=SPATIAL_MAX_IMAGES, force=force,
            )
            total += 1

    # Perturbed spatial
    logger.info('=== Spatial: Perturbed features ===')
    for ds_name in datasets:
        ds_info = ALL_DATASETS[ds_name]
        for pert_name, value in perturbation_subset:
            pcfg = PERTURB_CONFIGS[pert_name]
            tag = cache_tag_for(pert_name, value)
            for bb in backbones:
                img_size = BACKBONE_CONFIGS[bb].get('img_size')
                pert = make_perturbation(pcfg, value, img_size=img_size)
                logger.info('Spatial: %s × %s × %s', bb, ds_name, tag)
                extract_spatial_and_save(
                    bb, ds_info['root'], ds_name,
                    perturbation=pert, tag=tag,
                    max_images=SPATIAL_MAX_IMAGES, force=force,
                )
                total += 1

    logger.info('Spatial extraction done: %d total', total)


def main():
    parser = argparse.ArgumentParser(description='Feature extraction orchestrator')
    parser.add_argument('--mode', choices=['global', 'spatial', 'all'], default='all')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Dataset names (default: all tiers)')
    parser.add_argument('--clean-datasets', nargs='+', default=None,
                        help='Datasets for clean feature extraction (default: --datasets)')
    parser.add_argument('--perturb-datasets', nargs='+', default=None,
                        help='Datasets for perturbed feature extraction (default: Tier-A subset)')
    parser.add_argument('--backbones', nargs='+', default=None,
                        help='Backbone IDs (default: all 7)')
    parser.add_argument('--spatial-perturbations', nargs='*', default=None,
                        help='Spatial perturbations as name:value pairs, or "none".')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

    t0 = time.time()

    if args.mode in ('global', 'all'):
        perturb_datasets = args.perturb_datasets
        if perturb_datasets == ['none']:
            perturb_datasets = []
        extract_global_features(
            datasets=args.datasets,
            backbones=args.backbones,
            clean_datasets=args.clean_datasets,
            perturbation_datasets=perturb_datasets,
            force=args.force,
        )
    if args.mode in ('spatial', 'all'):
        extract_spatial_features(
            args.datasets,
            args.backbones,
            perturbation_subset=parse_spatial_perturbations(args.spatial_perturbations),
            force=args.force,
        )

    logger.info('Total time: %.1f min', (time.time() - t0) / 60)


if __name__ == '__main__':
    main()
