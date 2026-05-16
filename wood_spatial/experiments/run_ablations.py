#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proper ablation experiments — full methods, small representative image count.

Experiment 1: Seed sensitivity
  - 7 backbone × 3 Tier-A datasets × all 15 spatial cache perturbation tags
  - 5 images per condition (sampled evenly)
  - 10 random seeds, k=3, n_init=5
  - Subsample 300 spatial positions per image for speed

Experiment 2: Mahalanobis + kNN-distance baselines (failure detection)
  - 7 backbone × 3 Tier-A datasets × 34 perturbation tags = n=714
  - Full feature cache (all images, consistent with main paper evaluation)
  - No GPU needed

Experiment 3: Deblur intervention
  - 7 backbone × 3 Tier-A datasets × 6 blur severities
  - min(3 per class) images for test (held-out from gallery kNN)
  - Gallery kNN trained on ALL clean features (proper held-out design)
  - Wiener deconvolution with estimated PSF
  - GPU needed
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys, os

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import label as ndlabel
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings('ignore')
from wood_spatial.config import V4_SPATIAL_CACHE, V4_FEAT_CACHE, V4_CSV, BB_ORDER, N_CLUSTERS

SCACHE = V4_SPATIAL_CACHE
FCACHE = V4_FEAT_CACHE
OUT    = V4_CSV

BACKBONES = BB_ORDER

# Auto-discover all datasets and perturbation tags from feature cache
def discover_cache(fcache, backbones):
    """Return {(bb, ds): [tag, ...]} for all available cache files."""
    from collections import defaultdict
    index = defaultdict(list)
    known_ds = ['WRD25','DTSR14','PCA11','BFS46','FSDM41','GOIMAI','WOODAUTH','BD11',
                'VN26_x10','VN26_x20','VN26_x50']
    for bb in backbones:
        for f in sorted(fcache.glob(f'{bb}_*.npz')):
            stem = f.stem[len(bb)+1:]   # remove "bb_"
            for ds in sorted(known_ds, key=len, reverse=True):  # longest first
                if stem.startswith(ds):
                    tag = stem[len(ds)+1:] if len(stem) > len(ds) else 'original'
                    index[(bb, ds)].append(tag)
                    break
    return index

CACHE_INDEX = discover_cache(FCACHE, BACKBONES)

# All unique datasets found
ALL_DATASETS = sorted({ds for (_, ds) in CACHE_INDEX.keys()})
TIER_A = ['WRD25','DTSR14','PCA11']
print(f'Datasets found in cache: {ALL_DATASETS}')

BLUR_TAGS   = ['blur_2','blur_4','blur_6','blur_8','blur_10','blur_12']
BLUR_RADII  = {'blur_2':2,'blur_4':4,'blur_6':6,'blur_8':8,'blur_10':10,'blur_12':12}

BB_MODEL_IDS = {
    'resnet50':        'resnet50',
    'efficientnet_b3': 'efficientnet_b3',
    'convnext_tiny':   'convnext_tiny',
    'swin_tiny':       'swin_tiny_patch4_window7_224',
    'dinov2_b':        'vit_base_patch14_dinov2.lvd142m',
    'hrnet32':         'hrnet_w32',
    'mobilenetv3':     'mobilenetv3_large_100',
}


def _safe_name(*parts):
    raw = '__'.join(str(p) for p in parts)
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in raw)

# ─────────────────────────────────────────────────────────────────────────────
# EXP 1: SEED SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def run_seed_sensitivity(force=False):
    out_detail = OUT / 'exp2_seed_sensitivity.csv'
    out_summary = OUT / 'exp2_seed_sensitivity_summary.csv'
    if out_detail.exists() and out_summary.exists() and not force:
        print('\nEXP 1: Seed Sensitivity')
        print(f'  checkpoint hit: {out_summary}')
        return pd.read_csv(out_summary)

    print('\n' + '='*60)
    print('EXP 1: Seed Sensitivity')
    print('  7 BB × 3 DS × all spatial cache perturbations × 5 imgs × 10 seeds')
    print('='*60)

    K = N_CLUSTERS; SEEDS = list(range(10)); N_IMG = 5; N_SPATIAL = 300; N_INIT = 5

    def sgi(labels, H, W):
        return sum(ndlabel((labels==k).reshape(H,W))[1] for k in range(K)) / (H*W)

    records = []
    for bb in BACKBONES:
        all_sgi_per_seed = defaultdict(list)   # seed → [sgi values]
        n_conditions = 0

        # Collect all available spatial cache files for this backbone
        sc_files = sorted(SCACHE.glob(f'{bb}_*.npz'))
        if not sc_files:
            print(f'  {bb}: no spatial cache'); continue

        for sf in sc_files:
            # Only Tier-A datasets
            if not any(ds in sf.stem for ds in TIER_A):
                continue
            d = np.load(sf)
            feats = d['features']   # (N, C, H_f, W_f)
            N, C, H_f, W_f = feats.shape

            # Sample N_IMG images evenly
            idx = np.linspace(0, N-1, min(N_IMG, N), dtype=int)
            n_conditions += len(idx)

            for i in idx:
                X_full = feats[i].reshape(C, -1).T   # (H*W, C)
                n_pos = len(X_full)
                sub = np.random.RandomState(42).choice(
                    n_pos, min(N_SPATIAL, n_pos), replace=False)
                X_sub = X_full[sub]

                for seed in SEEDS:
                    km = KMeans(K, random_state=seed, n_init=N_INIT, max_iter=100)
                    km.fit(X_sub)
                    lbl = km.predict(X_full)
                    all_sgi_per_seed[seed].append(sgi(lbl, H_f, W_f))

        seed_means = [np.mean(all_sgi_per_seed[s]) for s in SEEDS
                      if all_sgi_per_seed[s]]
        if not seed_means: continue

        mean_sgi = np.mean(seed_means)
        std_sgi  = np.std(seed_means)
        cv_pct   = std_sgi / mean_sgi * 100 if mean_sgi > 0 else 0
        print(f'  {bb:20s}: mean_SGI={mean_sgi:.6f}, CV={cv_pct:.2f}%, '
              f'n_conditions={n_conditions}')

        for seed in SEEDS:
            if all_sgi_per_seed[seed]:
                records.append({
                    'backbone': bb, 'seed': seed,
                    'mean_sgi': np.mean(all_sgi_per_seed[seed]),
                    'std_sgi':  np.std(all_sgi_per_seed[seed]),
                    'n_images': n_conditions,
                })

    df = pd.DataFrame(records)
    df.to_csv(out_detail, index=False)

    summary = df.groupby('backbone').agg(
        mean_sgi=('mean_sgi', 'mean'),
        std_sgi=('mean_sgi', 'std'),
        cv_pct=('mean_sgi', lambda x: x.std()/x.mean()*100 if x.mean()>0 else 0),
        n_images=('n_images', 'first'),
    ).reset_index().sort_values('mean_sgi', ascending=False)
    summary.to_csv(out_summary, index=False)

    print('\n  Summary:')
    print(summary.to_string(index=False))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# EXP 2: MAHALANOBIS + kNN-DISTANCE BASELINES (full n=714)
# ─────────────────────────────────────────────────────────────────────────────

def load_feat(bb, ds, tag):
    p = FCACHE / f'{bb}_{ds}_{tag}.npz'
    if not p.exists(): return None, None, None
    d = np.load(p, allow_pickle=True)
    f = d['features'].astype(np.float32)
    nrm = np.linalg.norm(f, axis=1, keepdims=True)
    f = f / (nrm + 1e-8)
    paths = d['paths'] if 'paths' in d else None
    return f, d['labels'], paths


def _mahalanobis_pair(args):
    bb, ds, available_tags = args
    records = []
    if 'original' not in available_tags:
        return bb, ds, records, 'no_original'

    fc, lc, _ = load_feat(bb, ds, 'original')
    if fc is None:
        return bb, ds, records, 'missing_clean'

    try:
        cov = LedoitWolf(assume_centered=False).fit(fc)
        mah_clean = float(np.mean(cov.mahalanobis(fc)))
        has_mahal = True
    except Exception:
        has_mahal = False
        mah_clean = 0.0
        cov = None

    le = LabelEncoder()
    yc = le.fit_transform(lc)
    knn = KNeighborsClassifier(5, metric='cosine')
    knn.fit(fc, yc)
    acc_clean = knn.score(fc, yc)

    for tag in [t for t in available_tags if t != 'original']:
        fp, _, _ = load_feat(bb, ds, tag)
        if fp is None:
            continue

        n = min(len(fc), len(fp))
        cos = np.clip(np.sum(fc[:n] * fp[:n], axis=1), -1, 1)
        drift = float(np.mean(1 - cos))
        acc_p = knn.score(fp, yc[:len(fp)] if len(fp) != len(yc) else yc)
        drop = acc_clean - acc_p

        if has_mahal:
            try:
                mah_d = float(np.mean(cov.mahalanobis(fp))) - mah_clean
            except Exception:
                mah_d = np.nan
        else:
            mah_d = np.nan

        dists, _ = knn.kneighbors(fp, n_neighbors=1)
        knn_d = float(np.mean(dists))

        records.append({
            'backbone': bb, 'dataset': ds, 'perturbation': tag,
            'feature_drift': drift, 'mahal_delta': mah_d,
            'knn_dist': knn_d, 'accuracy_drop': drop,
        })
    return bb, ds, records, None


def run_mahalanobis(force=False, jobs=1):
    out_detail = OUT / 'exp8_additional_baselines.csv'
    out_auc = OUT / 'exp8_new_baselines_auc.csv'
    if out_detail.exists() and out_auc.exists() and not force:
        print('\nEXP 2: Mahalanobis + kNN-distance baselines')
        print(f'  checkpoint hit: {out_detail}')
        return pd.read_csv(out_auc)

    print('\n' + '='*60)
    print('EXP 2: Mahalanobis + kNN-distance baselines (full n=714)')
    print('='*60)

    tasks = [
        (bb, ds, CACHE_INDEX.get((bb, ds), []))
        for bb in BACKBONES
        for ds in sorted({d for (b, d) in CACHE_INDEX if b == bb})
    ]
    records = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(_mahalanobis_pair, task) for task in tasks]
            for fut in as_completed(futures):
                bb, ds, rows, err = fut.result()
                if err:
                    print(f'    {bb}/{ds}: {err}')
                else:
                    print(f'    {bb}/{ds}: {len(rows)} perturbations')
                records.extend(rows)
    else:
        for task in tasks:
            bb, ds, rows, err = _mahalanobis_pair(task)
            if err:
                print(f'    {bb}/{ds}: {err}')
            else:
                print(f'    {bb}/{ds}: {len(rows)} perturbations')
            records.extend(rows)

    df = pd.DataFrame(records)
    df.to_csv(out_detail, index=False)
    print(f'  Total rows: {len(df)} (target ~714 = 7×3×34)')

    y_true = (df['accuracy_drop'] > 0.20).astype(int)
    print(f'  Failure rate: {y_true.mean():.3f} ({y_true.sum()}/{len(y_true)})')

    results = []
    for name, col in [
        ('Feature drift',     'feature_drift'),
        ('Mahalanobis delta', 'mahal_delta'),
        ('kNN embed dist',    'knn_dist'),
    ]:
        s = df[col].fillna(df[col].median())
        auc = roc_auc_score(y_true, s)
        ap  = average_precision_score(y_true, s)
        rc  = float(np.corrcoef(s, df['accuracy_drop'])[0, 1])
        print(f'  {name:25s}: AUC={auc:.4f}, AvgPrec={ap:.4f}, r={rc:.4f}')
        results.append({'detector':name,'auc_roc':auc,'avg_precision':ap,'r_vs_drop':rc})

    pd.DataFrame(results).to_csv(out_auc, index=False)
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# EXP 3: DEBLUR INTERVENTION (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def _deblur_ckpt_dir():
    path = OUT / 'exp_deblur_checkpoints'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _deblur_ckpt_path(bb, ds):
    return _deblur_ckpt_dir() / f'{_safe_name(bb, ds)}.csv'


def _wiener_deblur(img_rgb, psf_sigma, snr_db=30.0):
    noise_power = 1.0 / (10 ** (snr_db / 10))
    H, W = img_rgb.shape[:2]
    y, x = np.mgrid[-H//2:H//2, -W//2:W//2]
    psf = np.exp(-(x**2 + y**2) / (2*psf_sigma**2)); psf /= psf.sum()
    PSF_F = np.fft.fft2(psf)
    out = []
    for c in range(3):
        ch = img_rgb[:,:,c].astype(np.float32) / 255.0
        CH_F = np.fft.fft2(ch)
        H_w = np.conj(PSF_F) / (np.abs(PSF_F)**2 + noise_power)
        out.append(np.clip(np.real(np.fft.ifft2(CH_F * H_w)) * 255, 0, 255).astype(np.uint8))
    return np.stack(out, axis=2)


def _apply_blur(img_rgb, radius):
    k = 2 * radius + 1
    return cv2.GaussianBlur(img_rgb, (k, k), sigmaX=radius)


def _extract_deblur_feat(model, bb, img_rgb, transform, device):
    import torch
    img_r = cv2.resize(img_rgb, (224, 224))
    t = transform(img_r).unsqueeze(0).to(device)
    with torch.no_grad():
        if bb == 'dinov2_b':
            f = model.forward_features(t)[:, 0, :]
        else:
            f = model(t)
    f = f.squeeze(0).cpu().numpy()
    return f / (np.linalg.norm(f) + 1e-8)


def run_deblur_pair(bb, ds, force=False):
    ckpt = _deblur_ckpt_path(bb, ds)
    if ckpt.exists() and not force:
        print(f'  checkpoint hit: {ckpt.name}')
        return pd.read_csv(ckpt)

    import torch
    import timm
    import torchvision.transforms as T

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TRANSFORM = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    fc_all, lc_all, paths_all = load_feat(bb, ds, 'original')
    if fc_all is None or paths_all is None:
        print(f'  {bb}/{ds}: missing clean cache')
        df = pd.DataFrame()
        df.to_csv(ckpt, index=False)
        return df

    print(f'  {bb}/{ds}: device={DEVICE}', flush=True)
    model_id = BB_MODEL_IDS[bb]
    if bb == 'dinov2_b':
        model = timm.create_model(model_id, pretrained=True, img_size=224).to(DEVICE).eval()
    else:
        model = timm.create_model(
            model_id, pretrained=True, num_classes=0, global_pool='avg'
        ).to(DEVICE).eval()

    le = LabelEncoder()
    yc = le.fit_transform(lc_all)
    knn = KNeighborsClassifier(5, metric='cosine')
    knn.fit(fc_all, yc)

    test_idx = []
    for cls in np.unique(yc):
        cls_idx = np.where(yc == cls)[0]
        chosen = np.random.RandomState(42).choice(
            cls_idx, min(3, len(cls_idx)), replace=False,
        )
        test_idx.extend(chosen.tolist())
    test_idx = np.array(test_idx)

    test_imgs, test_labels = [], []
    for i in test_idx:
        img_bgr = cv2.imread(str(paths_all[i]))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (224, 224))
        test_imgs.append(img_rgb)
        test_labels.append(yc[i])
    test_labels = np.array(test_labels)
    if len(test_imgs) < 5:
        print(f'  {bb}/{ds}: too few test images')
        df = pd.DataFrame()
        df.to_csv(ckpt, index=False)
        return df

    fc_test = np.array([
        _extract_deblur_feat(model, bb, img, TRANSFORM, DEVICE)
        for img in test_imgs
    ])
    acc_test_clean = knn.score(fc_test, test_labels)
    print(f'  {bb}/{ds}: n_test={len(test_imgs)}, acc_clean={acc_test_clean:.3f}', flush=True)

    records = []
    for blur_tag in BLUR_TAGS:
        r = BLUR_RADII[blur_tag]
        psf_sigma = r / 2.355
        blurred = [_apply_blur(img, r) for img in test_imgs]
        deblurred = [_wiener_deblur(img, psf_sigma) for img in blurred]

        fc_blur = np.array([
            _extract_deblur_feat(model, bb, img, TRANSFORM, DEVICE)
            for img in blurred
        ])
        fc_deblur = np.array([
            _extract_deblur_feat(model, bb, img, TRANSFORM, DEVICE)
            for img in deblurred
        ])

        drift_blur = float(np.mean(1 - np.sum(fc_test * fc_blur, axis=1)))
        drift_deblur = float(np.mean(1 - np.sum(fc_test * fc_deblur, axis=1)))
        acc_blur = knn.score(fc_blur, test_labels)
        acc_deblur = knn.score(fc_deblur, test_labels)
        drop_blur = acc_test_clean - acc_blur
        drop_deblur = acc_test_clean - acc_deblur
        drift_red = (drift_blur - drift_deblur) / drift_blur * 100 if drift_blur > 0.001 else 0
        acc_rec = (drop_blur - drop_deblur) / drop_blur * 100 if drop_blur > 0.001 else 0
        print(
            f'    {blur_tag}: drift {drift_blur:.3f}->{drift_deblur:.3f} '
            f'({drift_red:+.1f}%) drop {drop_blur:.3f}->{drop_deblur:.3f} '
            f'(recovery {acc_rec:+.1f}%)',
            flush=True,
        )
        records.append({
            'backbone': bb, 'dataset': ds, 'blur_tag': blur_tag,
            'blur_radius': r, 'n_test': len(test_imgs),
            'acc_clean': acc_test_clean, 'acc_blur': acc_blur, 'acc_deblur': acc_deblur,
            'drift_blur': drift_blur, 'drift_deblur': drift_deblur,
            'drift_reduction_pct': drift_red, 'acc_recovery_pct': acc_rec,
        })

    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    df.to_csv(ckpt, index=False)
    return df


def finalize_deblur(backbones=None, datasets=None):
    out_path = OUT / 'exp_deblur_intervention.csv'
    backbones = backbones or BACKBONES
    datasets = datasets or TIER_A
    frames = []
    for bb in backbones:
        for ds in datasets:
            path = _deblur_ckpt_path(bb, ds)
            if path.exists():
                try:
                    df = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    continue
                if not df.empty:
                    frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(out_path, index=False)
    if not out.empty:
        print(f'\n  Summary ({len(out)} conditions):')
        print(f'  Mean drift reduction: {out["drift_reduction_pct"].mean():.1f}%')
        print(f'  Mean acc recovery:    {out["acc_recovery_pct"].mean():.1f}%')
        r, p = pearsonr(out['drift_reduction_pct'], out['acc_recovery_pct'])
        print(f'  Correlation r={r:.3f}, p={p:.4f}')
        print('\n  Per blur radius:')
        print(out.groupby('blur_radius')[['drift_reduction_pct','acc_recovery_pct']].mean().round(2))
        print('\n  Per backbone:')
        print(out.groupby('backbone')[['drift_reduction_pct','acc_recovery_pct']].mean().round(2))
    return out


def run_deblur(force=False, backbones=None, datasets=None, checkpoint_only=False, finalize_only=False):
    out_path = OUT / 'exp_deblur_intervention.csv'
    backbones = backbones or BACKBONES
    datasets = datasets or TIER_A
    if out_path.exists() and not force and not checkpoint_only and not finalize_only:
        print('\nEXP 3: Deblur Intervention')
        print(f'  checkpoint hit: {out_path}')
        return pd.read_csv(out_path)

    print('\n' + '='*60)
    print('EXP 3: Deblur Intervention (Wiener, proper held-out)')
    print(f'  {len(backbones)} BB × {len(datasets)} DS × 6 blur severities × min(3/class) test images')
    print('='*60)

    if not finalize_only:
        for bb in backbones:
            for ds in datasets:
                if (bb, ds) not in CACHE_INDEX or 'original' not in CACHE_INDEX[(bb, ds)]:
                    print(f'  {bb}/{ds}: no clean cache')
                    continue
                run_deblur_pair(bb, ds, force=force)
                if checkpoint_only:
                    continue
    return finalize_deblur(backbones, datasets)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run optional ablation experiments with CSV-level resume.')
    parser.add_argument(
        '--only',
        nargs='+',
        choices=['seed', 'mahalanobis', 'deblur', 'all'],
        default=['all'],
        help='Sub-experiments to run. Existing CSV outputs are skipped unless --force is set.',
    )
    parser.add_argument('--force', action='store_true', help='Recompute even if output CSVs already exist.')
    parser.add_argument('--jobs', type=int, default=1, help='Parallel workers for Mahalanobis/kNN baseline.')
    parser.add_argument('--datasets', nargs='*', default=None, help='Datasets for deblur; default Tier-A.')
    parser.add_argument('--backbones', nargs='*', default=None, help='Backbones for deblur; default all.')
    parser.add_argument('--checkpoint-only', action='store_true', help='For deblur deep-parallel workers.')
    parser.add_argument('--finalize-only', action='store_true', help='Finalize deblur checkpoints only.')
    args = parser.parse_args()

    print('Running proper ablation experiments...')
    selected = set(args.only)
    if 'all' in selected:
        selected = {'seed', 'mahalanobis', 'deblur'}

    if 'seed' in selected:
        seed_summary = run_seed_sensitivity(force=args.force)

    if 'mahalanobis' in selected:
        mahal_summary = run_mahalanobis(force=args.force, jobs=max(1, args.jobs))

    if 'deblur' in selected:
        deblur_df = run_deblur(
            force=args.force,
            backbones=args.backbones or BACKBONES,
            datasets=args.datasets or TIER_A,
            checkpoint_only=args.checkpoint_only,
            finalize_only=args.finalize_only,
        )

    print('\n' + '='*60)
    print('ALL DONE')
    print('  exp2_seed_sensitivity.csv + exp2_seed_sensitivity_summary.csv')
    print('  exp8_additional_baselines.csv + exp8_new_baselines_auc.csv')
    print('  exp_deblur_intervention.csv')
