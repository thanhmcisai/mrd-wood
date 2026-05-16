#!/usr/bin/env python3
"""
Compute SSIM and PSNR for all (dataset, perturbation) pairs in exp8_additional_baselines.csv.
SSIM/PSNR are backbone-independent → compute once per (dataset, perturbation), join to all 7 backbones.
"""
import numpy as np
import pandas as pd
import cv2
import torch
from skimage.metrics import structural_similarity as ssim_fn, peak_signal_noise_ratio as psnr_fn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
import warnings; warnings.filterwarnings('ignore')
import sys as _sys
from wood_spatial.config import PERTURB_CONFIGS, V4_FEAT_CACHE, V4_CSV
from wood_spatial.core.perturbations import cache_tag_for, make_perturbation
# Force unbuffered output
class Unbuffered:
    def __init__(self, stream): self.stream = stream
    def write(self, data): self.stream.write(data); self.stream.flush()
    def __getattr__(self, attr): return getattr(self.stream, attr)
_sys.stdout = Unbuffered(_sys.stdout)

FCACHE = V4_FEAT_CACHE
OUT    = V4_CSV
N_SAMPLE = 200   # images per (dataset, perturbation) for SSIM/PSNR estimation
PIXEL_SIZE = 224

TAG_TO_SPEC = {
    cache_tag_for(pert_name, value): (pert_name, value)
    for pert_name, pcfg in PERTURB_CONFIGS.items()
    for value in pcfg['values']
}

# ── Perturbation functions ────────────────────────────────────────────────────
def apply_perturbation(img_bgr, tag):
    """Apply a configured perturbation tag to a BGR image, return 224x224 BGR."""
    spec = TAG_TO_SPEC.get(str(tag))
    if spec is None:
        return cv2.resize(img_bgr.copy(), (PIXEL_SIZE, PIXEL_SIZE))

    pert_name, value = spec
    transform = make_perturbation(PERTURB_CONFIGS[pert_name], value, img_size=PIXEL_SIZE)
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (PIXEL_SIZE, PIXEL_SIZE)), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0)
    try:
        with torch.no_grad():
            out = transform(tensor)
    except Exception:
        return img_bgr.copy()
    out_rgb = out.permute(1, 2, 0).cpu().numpy()
    out_rgb = np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

# ── Load image paths per dataset ─────────────────────────────────────────────
def get_paths(ds):
    f = FCACHE / f'resnet50_{ds}_original.npz'
    if not f.exists(): return []
    d = np.load(f, allow_pickle=True)
    return list(d['paths'])

# ── Compute SSIM/PSNR for one (dataset, perturbation) ────────────────────────
def compute_pair(paths, tag, n_sample=N_SAMPLE):
    rng = np.random.RandomState(42)
    idx = rng.choice(len(paths), min(n_sample, len(paths)), replace=False)
    ssim_vals, psnr_vals = [], []
    for i in idx:
        p = str(paths[i])
        img = cv2.imread(p)
        if img is None: continue
        pert = apply_perturbation(img, tag)
        # Resize to 224×224 for efficiency (consistent with the frozen-feature baseline)
        orig_s = cv2.resize(img,  (PIXEL_SIZE, PIXEL_SIZE))
        pert_s = cv2.resize(pert, (PIXEL_SIZE, PIXEL_SIZE))
        try:
            s = ssim_fn(orig_s, pert_s, channel_axis=2, data_range=255)
            p_val = psnr_fn(orig_s, pert_s, data_range=255)
            ssim_vals.append(s)
            psnr_vals.append(p_val if np.isfinite(p_val) else 60.0)
        except Exception:
            pass
    if not ssim_vals: return np.nan, np.nan
    return float(np.mean(ssim_vals)), float(np.mean(psnr_vals))

# ── Main ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(OUT / 'exp8_additional_baselines.csv')
unique_pairs = df[['dataset','perturbation']].drop_duplicates()
print(f"Computing SSIM/PSNR for {len(unique_pairs)} unique (dataset, perturbation) pairs...")
print(f"Total exp8 rows: {len(df)}")

ds_paths = {}
for ds in df.dataset.unique():
    ds_paths[ds] = get_paths(ds)
    print(f"  {ds}: {len(ds_paths[ds])} images")

results = {}
for i, (_, row) in enumerate(unique_pairs.iterrows()):
    ds, tag = row['dataset'], row['perturbation']
    paths = ds_paths.get(ds, [])
    if not paths:
        results[(ds, tag)] = (np.nan, np.nan)
        continue
    s, p = compute_pair(paths, tag)
    results[(ds, tag)] = (s, p)
    if (i+1) % 20 == 0 or (i+1) == len(unique_pairs):
        print(f"  [{i+1}/{len(unique_pairs)}] {ds} × {tag}: SSIM={s:.4f}, PSNR={p:.2f}dB")

# Join back to full dataframe
df['ssim'] = df.apply(lambda r: results.get((r['dataset'],r['perturbation']),(np.nan,np.nan))[0], axis=1)
df['psnr'] = df.apply(lambda r: results.get((r['dataset'],r['perturbation']),(np.nan,np.nan))[1], axis=1)
df.to_csv(OUT / 'exp8_baselines_with_pixel.csv', index=False)
print(f"\nSaved: exp8_baselines_with_pixel.csv ({len(df)} rows)")
print(f"SSIM NaN: {df['ssim'].isna().sum()}, PSNR NaN: {df['psnr'].isna().sum()}")

# ── Compute all metrics ────────────────────────────────────────────────────────
y_true = (df['accuracy_drop'] > 0.20).astype(int)
print(f"\nFailure rate: {y_true.mean():.3f} ({y_true.sum()}/{len(y_true)})")

all_metrics = []
for name, col, higher_is_worse in [
    ('Feature drift',       'feature_drift', True),
    ('Mahalanobis delta',   'mahal_delta',   True),
    ('kNN embed dist',      'knn_dist',      True),
    ('1 − SSIM',            'ssim',          False),  # lower SSIM = worse = failure
    ('−PSNR',               'psnr',          False),  # lower PSNR = worse = failure
]:
    s = df[col].fillna(df[col].median())
    score = s if higher_is_worse else -s   # flip so higher = more likely failure
    try:
        auc = roc_auc_score(y_true, score)
        ap  = average_precision_score(y_true, score)
        precision, recall, thresholds = precision_recall_curve(y_true, score)
        f1s = 2*precision*recall/(precision+recall+1e-10)
        best_f1 = float(np.max(f1s))
        best_thr = thresholds[np.argmax(f1s)] if np.argmax(f1s) < len(thresholds) else thresholds[-1]
        print(f"{name:25s}: AUC={auc:.4f}  AvgPrec={ap:.4f}  BestF1={best_f1:.4f}  τ={best_thr:.4f}")
        all_metrics.append({'detector': name, 'auc_roc': auc, 'avg_precision': ap,
                             'best_f1': best_f1, 'best_threshold': best_thr, 'n': len(df)})
    except Exception as ex:
        print(f"{name}: ERROR {ex}")

pd.DataFrame(all_metrics).to_csv(OUT / 'exp8_all_detectors_full.csv', index=False)
print(f"\nSaved: exp8_all_detectors_full.csv")
print("DONE")
