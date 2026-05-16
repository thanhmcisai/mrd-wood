"""
Pixel-Space Baseline vs Feature Drift
======================================
Compares pixel-space image quality metrics (PSNR, SSIM) as failure detectors
against feature drift (AUC=0.984).

Motivation: Reviewer may ask "why not just measure image degradation?"
This shows deep feature drift >> pixel metrics for predicting recognition failure.

Note: PSNR/SSIM require the clean reference image — making them strictly EASIER
than feature_drift which only needs feature spaces. Yet feature_drift still wins.
"""
import logging
import time
import warnings

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
try:
    from skimage.metrics import structural_similarity as ssim_fn, peak_signal_noise_ratio as psnr_fn
except ImportError:
    import cv2 as _cv2
    def psnr_fn(a, b, data_range=255):
        mse = float(np.mean((a.astype(float) - b.astype(float))**2))
        return 100.0 if mse < 1e-10 else 10 * np.log10(data_range**2 / mse)
    def ssim_fn(a, b, channel_axis=None, data_range=255):
        gray_a = _cv2.cvtColor(a, _cv2.COLOR_RGB2GRAY).astype(float)
        gray_b = _cv2.cvtColor(b, _cv2.COLOR_RGB2GRAY).astype(float)
        mu_a, mu_b = gray_a.mean(), gray_b.mean()
        sig_a = gray_a.std(); sig_b = gray_b.std()
        sig_ab = float(np.mean((gray_a - mu_a) * (gray_b - mu_b)))
        c1, c2 = (0.01*data_range)**2, (0.03*data_range)**2
        return float((2*mu_a*mu_b+c1)*(2*sig_ab+c2) / ((mu_a**2+mu_b**2+c1)*(sig_a**2+sig_b**2+c2)))
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr

from wood_spatial.config import BB_ORDER, PERTURB_CONFIGS, TIER_A, V4_CSV, ALL_DATASETS, BACKBONE_CONFIGS
from wood_spatial.core.cache import load_cache
from wood_spatial.core.perturbations import make_perturbation, cache_tag_for
from wood_spatial.core.dataset import WoodDataset

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

SAMPLE_BACKBONE = 'resnet50'
SAMPLE_PERTURBATIONS = [
    ('gaussian_blur', 12),
    ('defocus_blur', 11),
    ('resize', 2.0),
    ('jpeg', 10),
    ('rotation', 45),
    ('illumination', 0.7),
    ('red_channel_shift', -45),
    ('green_channel_shift', -45),
    ('blue_channel_shift', -45),
    ('gaussian_noise', 0.10),
    ('shot_noise', 15),
    ('impulse_noise', 0.05),
    ('motion_blur', 15),
    ('zoom_blur', 1.20),
    ('contrast', 0.50),
    ('pixelate', 0.25),
    ('scratch', 'severe'),
    ('compound', 'severe'),
    ('compound_optical', 'severe'),
    ('compound_digital', 'severe'),
    ('compound_field', 'severe'),
]
MAX_IMAGES_PER_PERT = 100
FAILURE_THRESHOLD = 0.20


def _compute_pixel_metrics(clean_img: np.ndarray, pert_img: np.ndarray) -> dict:
    """Compute PSNR and SSIM between clean and perturbed images."""
    if clean_img.shape != pert_img.shape:
        pert_img = cv2.resize(pert_img, (clean_img.shape[1], clean_img.shape[0]))
    psnr = float(psnr_fn(clean_img, pert_img, data_range=255))
    s = float(ssim_fn(clean_img, pert_img, channel_axis=2, data_range=255))
    return {'psnr': psnr, 'ssim': s}


def run_pixel_baseline(save: bool = True) -> dict:
    rows = []

    for ds_name in TIER_A:
        root = ALL_DATASETS[ds_name]['root']
        img_size = BACKBONE_CONFIGS[SAMPLE_BACKBONE]['img_size']

        # Load clean features and accuracy baseline
        try:
            feat_clean, labels_clean, paths_clean = load_cache(SAMPLE_BACKBONE, ds_name, 'original')
        except FileNotFoundError:
            continue

        from wood_spatial.analysis.statistical_tests import knn_accuracy_cv, knn_accuracy_gallery_query
        acc_clean, _ = knn_accuracy_cv(feat_clean, labels_clean)

        for pert_name, value in SAMPLE_PERTURBATIONS:
            tag = cache_tag_for(pert_name, value)
            try:
                feat_pert, labels_pert, paths_pert = load_cache(SAMPLE_BACKBONE, ds_name, tag)
            except FileNotFoundError:
                continue

            # Feature drift
            from wood_spatial.analysis.feature_geometry import feature_drift
            drift = feature_drift(feat_clean, feat_pert)

            # Accuracy drop
            acc_pert = knn_accuracy_gallery_query(feat_clean, labels_clean, feat_pert, labels_pert)
            drop = acc_clean - acc_pert

            # Pixel-space metrics (sample images)
            cfg = PERTURB_CONFIGS[pert_name]
            pert_transform = make_perturbation(cfg, value, img_size)
            n = min(MAX_IMAGES_PER_PERT, len(paths_clean))
            psnr_vals, ssim_vals = [], []

            for path in paths_clean[:n]:
                img_bgr = cv2.imread(str(path))
                if img_bgr is None:
                    continue
                img_bgr = cv2.resize(img_bgr, (img_size, img_size))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                import torch
                import torchvision.transforms.functional as TF
                tensor_clean = TF.to_tensor(img_rgb)
                try:
                    tensor_pert = pert_transform(tensor_clean)
                    img_pert = (tensor_pert.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                except Exception:
                    continue
                m = _compute_pixel_metrics(img_rgb, img_pert)
                psnr_vals.append(m['psnr'])
                ssim_vals.append(m['ssim'])

            if not psnr_vals:
                continue

            rows.append({
                'dataset': ds_name,
                'perturbation': pert_name,
                'severity': value,
                'feature_drift': drift,
                'drop': drop,
                'psnr_mean': float(np.mean(psnr_vals)),
                'ssim_mean': float(np.mean(ssim_vals)),
            })
            logger.info('  %s/%s/%s: drift=%.3f drop=%.3f PSNR=%.1f SSIM=%.3f',
                        ds_name, pert_name, value, drift, drop, np.mean(psnr_vals), np.mean(ssim_vals))

    df = pd.DataFrame(rows)
    if len(df) == 0:
        logger.error('No data computed')
        return {}

    y = (df['drop'] > FAILURE_THRESHOLD).astype(int)

    # Detector comparison
    det_rows = []
    for det_name, scores, higher_is_worse in [
        ('feature_drift', df['feature_drift'], True),
        ('1 - SSIM', 1 - df['ssim_mean'], True),
        ('neg_PSNR', -df['psnr_mean'], True),
    ]:
        if y.sum() < 3 or (1 - y).sum() < 3:
            continue
        try:
            auc = roc_auc_score(y, scores)
            ap = average_precision_score(y, scores)
            r, p = pearsonr(scores.values, df['drop'].values)
            det_rows.append({'detector': det_name, 'auc_roc': auc, 'avg_precision': ap,
                             'r_vs_drop': r, 'p_vs_drop': p, 'n': len(df)})
        except Exception as e:
            logger.warning('%s failed: %s', det_name, e)

    df_det = pd.DataFrame(det_rows).sort_values('auc_roc', ascending=False)
    logger.info('\n=== Pixel-Space vs Feature Drift Comparison ===')
    logger.info('\n%s', df_det.to_string(index=False))

    # Correlation
    logger.info('\nCorrelations with accuracy drop:')
    for col in ['feature_drift', 'ssim_mean', 'psnr_mean']:
        r, p = pearsonr(df[col], df['drop'])
        logger.info('  %-15s: r=%.3f (p=%.2e)', col, r, p)

    results = {'data': df, 'detector_comparison': df_det}

    if save:
        df.to_csv(V4_CSV / 'exp_pixel_baseline.csv', index=False)
        df_det.to_csv(V4_CSV / 'exp_pixel_baseline_auc.csv', index=False)
        logger.info('Saved pixel baseline to %s', V4_CSV)

    return results


def plot_pixel_baseline(results: dict):
    import matplotlib.pyplot as plt

    df_det = results['detector_comparison']
    df = results['data']
    from wood_spatial.config import V4_FIGURES

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: AUC comparison
    ax = axes[0]
    colors = ['firebrick' if 'feature' in d else 'steelblue' for d in df_det['detector']]
    ax.barh(df_det['detector'], df_det['auc_roc'], color=colors, alpha=0.85)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('AUC-ROC')
    ax.set_title('Deep feature drift vs pixel-space metrics\nas unsupervised failure detectors')
    for i, (_, row) in enumerate(df_det.iterrows()):
        ax.text(row['auc_roc'] - 0.01, i, f'{row["auc_roc"]:.4f}',
                va='center', ha='right', fontsize=9, color='white')
    ax.set_xlim(0.4, 1.05)

    # Panel B: Scatter — SSIM vs drop vs feature_drift
    ax2 = axes[1]
    sc1 = ax2.scatter(df['ssim_mean'], df['drop'], c=df['feature_drift'],
                      cmap='Reds', s=40, alpha=0.7, edgecolors='none')
    plt.colorbar(sc1, ax=ax2, label='Feature drift')
    ax2.set_xlabel('SSIM (pixel-space quality)')
    ax2.set_ylabel('Accuracy drop')
    ax2.set_title('SSIM vs drop (color = feature drift)\nFeature drift better predicts failure than SSIM')

    fig.suptitle('Why feature drift > pixel metrics: deep representations capture semantic distortion', fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(V4_FIGURES / f'fig_pixel_baseline.{ext}', bbox_inches='tight',
                    dpi=300 if ext == 'png' else None)
    logger.info('Saved fig_pixel_baseline')
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    t0 = time.time()
    results = run_pixel_baseline()
    if results:
        plot_pixel_baseline(results)
    logger.info('Pixel baseline done in %.1f min', (time.time() - t0) / 60)
    if results:
        print('\n=== Detector Comparison ===')
        print(results['detector_comparison'].to_string(index=False))


if __name__ == '__main__':
    main()
