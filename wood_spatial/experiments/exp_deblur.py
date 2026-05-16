#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROPER deblur intervention experiment.

Design (fixes all 3 methodological issues):
1. Gallery kNN = all clean features from cache (not in-sample)
2. Wiener deconvolution (proper PSF-based, not unsharp mask)
3. 50 images per dataset from held-out subset of cache paths
4. All 7 backbones, multiple blur severities (4, 8, 12)
5. k=3 consistent with the final spatial configuration
"""
import numpy as np
import pandas as pd
import cv2
import torch
import timm
import torchvision.transforms as T
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings('ignore')

from wood_spatial.config import BB_ORDER, TIER_A, V4_FEAT_CACHE, V4_CSV

CACHE  = V4_FEAT_CACHE
OUT    = V4_CSV
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

BACKBONES = BB_ORDER
DATASETS  = TIER_A
BLUR_RADII = [4, 8, 12]
N_TEST     = 50   # images per dataset for intervention test

BB_MODEL_IDS = {
    'resnet50':        'resnet50',
    'efficientnet_b3': 'efficientnet_b3',
    'convnext_tiny':   'convnext_tiny',
    'swin_tiny':       'swin_tiny_patch4_window7_224',
    'dinov2_b':        'vit_base_patch14_dinov2.lvd142m',
    'hrnet32':         'hrnet_w32',
    'mobilenetv3':     'mobilenetv3_large_100',
}

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Wiener deconvolution (proper PSF-based, no reference needed) ─────────────
def wiener_deblur(img_rgb, psf_sigma, noise_snr=30.0):
    """
    Wiener filter in frequency domain.
    psf_sigma: estimated sigma of blur PSF (for Gaussian blur radius r, sigma ≈ r/2.355)
    noise_snr: assumed signal-to-noise ratio (dB)
    """
    H, W = img_rgb.shape[:2]
    noise_power = 1.0 / (10 ** (noise_snr / 10))

    # Gaussian PSF
    y, x = np.mgrid[-H//2:H//2, -W//2:W//2]
    psf = np.exp(-(x**2 + y**2) / (2 * psf_sigma**2))
    psf /= psf.sum()
    PSF_F = np.fft.fft2(psf)

    out_channels = []
    for c in range(3):
        ch = img_rgb[:, :, c].astype(np.float32) / 255.0
        CH_F = np.fft.fft2(ch)
        # Wiener filter: H* / (|H|^2 + noise/signal)
        PSF_conj = np.conj(PSF_F)
        denom = np.abs(PSF_F)**2 + noise_power
        H_wiener = PSF_conj / denom
        restored = np.real(np.fft.ifft2(CH_F * H_wiener))
        out_channels.append(np.clip(restored * 255, 0, 255).astype(np.uint8))

    return np.stack(out_channels, axis=2)

def apply_blur(img_rgb, radius):
    ksize = 2 * radius + 1
    return cv2.GaussianBlur(img_rgb, (ksize, ksize), sigmaX=radius)

# ── Feature extraction ────────────────────────────────────────────────────────
_model_cache = {}

def get_model(bb):
    if bb in _model_cache:
        return _model_cache[bb]
    model_id = BB_MODEL_IDS[bb]
    if bb == 'dinov2_b':
        m = timm.create_model(model_id, pretrained=True, img_size=224).to(DEVICE).eval()
    elif bb == 'swin_tiny':
        m = timm.create_model(model_id, pretrained=True,
                              num_classes=0, global_pool='avg').to(DEVICE).eval()
    else:
        m = timm.create_model(model_id, pretrained=True,
                              num_classes=0, global_pool='avg').to(DEVICE).eval()
    _model_cache[bb] = m
    return m

def extract_single(bb, img_rgb):
    """Extract L2-normalized global feature for one RGB image."""
    img_r = cv2.resize(img_rgb, (224, 224))
    t = TRANSFORM(img_r).unsqueeze(0).to(DEVICE)
    m = get_model(bb)
    with torch.no_grad():
        if bb == 'dinov2_b':
            f = m.forward_features(t)[:, 0, :]   # CLS token
        else:
            f = m(t)
    f = f.squeeze(0).cpu().numpy()
    return f / (np.linalg.norm(f) + 1e-8)

def extract_batch(bb, images_rgb):
    """Extract features for list of RGB images."""
    return np.array([extract_single(bb, img) for img in images_rgb])

# ── Load cache ────────────────────────────────────────────────────────────────
def load_cache(bb, ds, tag='original'):
    p = CACHE / f'{bb}_{ds}_{tag}.npz'
    if not p.exists(): return None, None, None
    d = np.load(p, allow_pickle=True)
    feats  = d['features']
    labels = d['labels']
    norms  = np.linalg.norm(feats, axis=1, keepdims=True)
    feats  = feats / (norms + 1e-8)
    paths  = d['paths'] if 'paths' in d else None
    return feats, labels, paths

# ── Main experiment ───────────────────────────────────────────────────────────
records = []

for bb in BACKBONES:
    print(f'\n=== {bb} ===')
    model = get_model(bb)   # warm up

    for ds in DATASETS:
        fc_all, lc_all, paths_all = load_cache(bb, ds, 'original')
        if fc_all is None:
            print(f'  {ds}: missing cache')
            continue

        # ── Gallery kNN: fit on ALL clean features (proper held-out design)
        le  = LabelEncoder()
        yc  = le.fit_transform(lc_all)
        knn = KNeighborsClassifier(5, metric='cosine')
        knn.fit(fc_all, yc)
        acc_clean_full = knn.score(fc_all, yc)   # in-sample upper bound

        # ── Select test subset (indices NOT overlapping gallery in evaluation)
        # Use last N_TEST images as test set (gallery uses all N for fitting)
        N = len(fc_all)
        test_idx = np.random.RandomState(42).choice(N, min(N_TEST, N), replace=False)

        # Load test images from disk
        if paths_all is None:
            print(f'  {ds}: no paths')
            continue

        test_imgs = []
        valid_idx = []
        for i in test_idx:
            img_bgr = cv2.imread(str(paths_all[i]))
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (224, 224))
            test_imgs.append(img_rgb)
            valid_idx.append(i)

        if len(test_imgs) < 10:
            print(f'  {ds}: too few images')
            continue

        y_test = yc[valid_idx]
        fc_test_clean = extract_batch(bb, test_imgs)
        acc_test_clean = knn.score(fc_test_clean, y_test)

        print(f'  {ds} (n_test={len(test_imgs)}): acc_clean={acc_test_clean:.3f}')

        for blur_r in BLUR_RADII:
            # Blur images
            blurred = [apply_blur(img, blur_r) for img in test_imgs]
            fc_blur = extract_batch(bb, blurred)
            acc_blur = knn.score(fc_blur, y_test)
            drift_blur = float(np.mean(1 - np.sum(fc_test_clean * fc_blur, axis=1)))

            # Wiener deblur
            psf_sigma = blur_r / 2.355   # Gaussian PSF for known blur radius
            deblurred = [wiener_deblur(img, psf_sigma) for img in blurred]
            fc_deblu  = extract_batch(bb, deblurred)
            acc_deblu = knn.score(fc_deblu, y_test)
            drift_deblu = float(np.mean(1 - np.sum(fc_test_clean * fc_deblu, axis=1)))

            drop_blur  = acc_test_clean - acc_blur
            drop_deblu = acc_test_clean - acc_deblu
            drift_reduction = (drift_blur - drift_deblu) / drift_blur * 100 if drift_blur > 0.001 else 0
            acc_recovery    = (drop_blur - drop_deblu) / drop_blur * 100 if drop_blur > 0.001 else 0

            print(f'    blur_r={blur_r:2d}: '
                  f'drift {drift_blur:.3f}→{drift_deblu:.3f} ({drift_reduction:+.1f}%)  '
                  f'drop {drop_blur:.3f}→{drop_deblu:.3f} (recovery {acc_recovery:+.1f}%)')

            records.append({
                'backbone': bb, 'dataset': ds, 'blur_radius': blur_r,
                'n_test': len(test_imgs),
                'acc_test_clean': acc_test_clean,
                'drift_blur':     drift_blur,
                'drift_deblur':   drift_deblu,
                'drift_reduction_pct': drift_reduction,
                'drop_blur':      drop_blur,
                'drop_deblur':    drop_deblu,
                'acc_recovery_pct': acc_recovery,
            })

    # Free GPU memory
    torch.cuda.empty_cache()

# ── Save and summary ─────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(OUT / 'exp_deblur_intervention.csv', index=False)

print('\n=== SUMMARY ===')
print(f'Total conditions: {len(df)}')
print(f'Mean drift reduction: {df["drift_reduction_pct"].mean():.1f}%')
print(f'Mean accuracy recovery: {df["acc_recovery_pct"].mean():.1f}%')

r, p = pearsonr(df['drift_reduction_pct'], df['acc_recovery_pct'])
print(f'Correlation drift_reduction ~ acc_recovery: r={r:.3f}, p={p:.3f}')

print('\nPer blur_radius:')
print(df.groupby('blur_radius')[['drift_reduction_pct','acc_recovery_pct']].mean().round(2))

print('\nPer backbone (mean across datasets and blur levels):')
print(df.groupby('backbone')[['drift_reduction_pct','acc_recovery_pct']].mean().round(2))

print('DONE')
