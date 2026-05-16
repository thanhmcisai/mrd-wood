"""
Wood Spatial — Main CLI Entry Point
=====================================
Run experiments from the command line.

Usage:
    python -m wood_spatial.main --exp 1         # Run Exp 1 only
    python -m wood_spatial.main --exp 1,2,3     # Run Exps 1-3
    python -m wood_spatial.main --exp all        # Run all experiments
    python -m wood_spatial.main --extract global # Feature extraction only
    python -m wood_spatial.main --extract all    # All extraction + all experiments
"""
import argparse
import logging
import time
import sys

logger = logging.getLogger('wood_spatial')


def run_extraction(mode: str, args):
    from wood_spatial.experiments.extract_features import (
        extract_global_features, extract_spatial_features,
    )
    if mode in ('global', 'all'):
        extract_global_features(
            datasets=args.datasets,
            backbones=args.backbones,
            clean_datasets=getattr(args, 'clean_datasets', None),
            perturbation_datasets=getattr(args, 'perturb_datasets', None),
            force=args.force,
        )
    if mode in ('spatial', 'all'):
        extract_spatial_features(
            datasets=args.datasets, backbones=args.backbones, force=args.force,
        )


def run_experiments(exp_ids: list, args):
    results = {}

    if '1' in exp_ids:
        logger.info('=' * 60)
        logger.info('EXPERIMENT 1: Perturbation Robustness Benchmark')
        logger.info('=' * 60)
        from wood_spatial.experiments.exp1_robustness import run_exp1
        results['exp1'] = run_exp1(
            datasets=args.datasets, backbones=args.backbones,
        )

    if '1b' in exp_ids:
        logger.info('=' * 60)
        logger.info('EXPERIMENT 1b: Feature Geometry Distortion')
        logger.info('=' * 60)
        from wood_spatial.experiments.exp1b_feature_geometry import run_exp1b
        results['exp1b'] = run_exp1b(
            datasets=args.datasets, backbones=args.backbones,
        )

    if '2' in exp_ids:
        logger.info('=' * 60)
        logger.info('EXPERIMENT 2: Spatial Feature Clustering')
        logger.info('=' * 60)
        from wood_spatial.experiments.exp2_spatial_clustering import run_exp2
        results['exp2'] = run_exp2(
            datasets=args.datasets, backbones=args.backbones,
        )

    if '3' in exp_ids:
        logger.info('=' * 60)
        logger.info('EXPERIMENT 3: Grad-CAM x Cluster Overlap')
        logger.info('=' * 60)
        from wood_spatial.experiments.exp3_gradcam_cluster import run_exp3
        results['exp3'] = run_exp3(
            datasets=args.datasets, backbones=args.backbones,
        )

    if '4' in exp_ids:
        logger.info('=' * 60)
        logger.info('EXPERIMENT 4: Cross-Dataset & Wild Generalization')
        logger.info('=' * 60)
        from wood_spatial.experiments.exp4_cross_dataset import run_exp4
        results['exp4'] = run_exp4(backbones=args.backbones)

    if '5' in exp_ids:
        logger.info('=' * 60)
        logger.info('EXPERIMENT 5: Cross-Magnification Stability')
        logger.info('=' * 60)
        from wood_spatial.experiments.exp5_magnification import run_exp5
        results['exp5'] = run_exp5(backbones=args.backbones)

    if '6' in exp_ids:
        logger.info('=' * 60)
        logger.info('EXPERIMENT 6: Multi-level Failure Analysis')
        logger.info('=' * 60)
        from wood_spatial.experiments.exp6_multilevel_failure import run_exp6
        results['exp6'] = run_exp6()

    if '7' in exp_ids:
        logger.info('=' * 60)
        logger.info('EXPERIMENT 7: Path Analysis & Hierarchical Regression')
        logger.info('=' * 60)
        from wood_spatial.experiments.exp7_path_analysis import run_exp7
        results['exp7'] = run_exp7()

    if '8' in exp_ids:
        logger.info('=' * 60)
        logger.info('EXPERIMENT 8: Unsupervised Failure Detection')
        logger.info('=' * 60)
        from wood_spatial.experiments.exp8_failure_detection import run_exp8
        results['exp8'] = run_exp8()

    if '9' in exp_ids:
        logger.info('=' * 60)
        logger.info('EXPERIMENT 9: Tier-B Cross-Dataset Validation')
        logger.info('=' * 60)
        from wood_spatial.experiments.exp9_tierb_validation import run_exp9
        results['exp9'] = run_exp9()

    # Cross-experiment analysis
    if '1' in exp_ids and '2' in exp_ids and 'exp1' in results and 'exp2' in results:
        logger.info('=' * 60)
        logger.info('SGI vs Robustness Correlation (RQ3)')
        logger.info('=' * 60)
        from wood_spatial.experiments.exp2_spatial_clustering import analyze_sgi_vs_robustness
        corr = analyze_sgi_vs_robustness(results['exp1'], results['exp2'])
        logger.info('SGI-Robustness correlation: r=%.3f, p=%.4g (n=%d)',
                     corr['r'], corr['p'], corr['n'])

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Wood Spatial — Research Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m wood_spatial.main --exp 1
  python -m wood_spatial.main --exp 1,2,3 --backbones resnet50 dinov2_b hrnet32
  python -m wood_spatial.main --extract global --datasets WRD25
  python -m wood_spatial.main --exp all --extract all
        """,
    )
    parser.add_argument('--exp', type=str, default=None,
                        help='Experiment IDs: "1", "1b", "1,1b,6", or "all"')
    parser.add_argument('--extract', type=str, default=None,
                        choices=['global', 'spatial', 'all'],
                        help='Run feature extraction')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Dataset names (default: tier-appropriate)')
    parser.add_argument('--clean-datasets', nargs='+', default=None,
                        help='Datasets for clean global extraction')
    parser.add_argument('--perturb-datasets', nargs='+', default=None,
                        help='Datasets for perturbed global extraction')
    parser.add_argument('--backbones', nargs='+', default=None,
                        help='Backbone IDs (default: all 7)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-extraction even if cache exists')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )

    if args.exp is None and args.extract is None:
        parser.print_help()
        sys.exit(0)

    t0 = time.time()

    # Feature extraction
    if args.extract:
        run_extraction(args.extract, args)

    # Experiments
    if args.exp:
        if args.exp == 'all':
            exp_ids = ['1', '1b', '2', '3', '4', '5', '6', '7', '8', '9']
        else:
            exp_ids = [x.strip() for x in args.exp.split(',')]
        run_experiments(exp_ids, args)

    elapsed = time.time() - t0
    logger.info('Total runtime: %.1f min (%.0f sec)', elapsed / 60, elapsed)


if __name__ == '__main__':
    main()
