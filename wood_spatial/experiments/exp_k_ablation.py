#!/usr/bin/env python3
"""k-ablation v3 — L2-norm + Swin-T HWC cache fix."""
import numpy as np, pandas as pd
from scipy.ndimage import label as ndlabel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import sys as _sys, warnings; warnings.filterwarnings('ignore')
from wood_spatial.config import BB_ORDER, TIER_A, V4_SPATIAL_CACHE, V4_CSV
class Unbuffered:
    def __init__(self,s): self.s=s
    def write(self,d): self.s.write(d); self.s.flush()
    def __getattr__(self,a): return getattr(self.s,a)
_sys.stdout = Unbuffered(_sys.stdout)

SCACHE = V4_SPATIAL_CACHE
OUT    = V4_CSV
BACKBONES = BB_ORDER
DATASETS  = TIER_A
K_VALUES  = [2,3,4,5,6,7,8]; N_IMG=100; N_INIT=5; SEED=42

def sgi_fn(labels, k, H, W):
    return sum(ndlabel((labels==ki).reshape(H,W))[1] for ki in range(k)) / (H*W)

def load_cache(bb, ds):
    sf = SCACHE / f'{bb}_{ds}_original.npz'
    if not sf.exists(): return None, None, None, None, None
    d = np.load(sf, allow_pickle=True)
    feats = d['features']; labels = d['labels']
    if feats.shape[-1] > feats.shape[1]:  # HWC
        N, H_f, W_f, C = feats.shape
    else:  # CHW → convert to HWC
        N, C, H_f, W_f = feats.shape
        feats = feats.transpose(0, 2, 3, 1)
    return feats, labels, N, H_f, W_f

records = []
print(f'Scope: {len(BACKBONES)} BB x {len(DATASETS)} DS x k={K_VALUES}')
print(f'Fixes: L2-norm, Swin-T HWC detection')

for bb in BACKBONES:
    print(f'\n=== {bb} ===')
    for ds in DATASETS:
        feats_hwc, labels_sc, N, H_f, W_f = load_cache(bb, ds)
        if feats_hwc is None: print(f'  {ds}: no cache'); continue
        C = feats_hwc.shape[-1]
        le = LabelEncoder(); yc = le.fit_transform(labels_sc)
        n_cls = len(np.unique(yc)); per_cls = max(1, N_IMG//n_cls)
        selected = []
        for cls in np.unique(yc):
            idx = np.where(yc==cls)[0]
            c = np.random.RandomState(SEED).choice(idx, min(per_cls,len(idx)), replace=False)
            selected.extend(c.tolist())
        selected = sorted(selected[:N_IMG])
        n_sel = len(selected)
        print(f'  {ds} ({n_sel} imgs, {H_f}x{W_f}x{C}): ', end='')
        for k in K_VALUES:
            sgi_vals, sil_vals = [], []
            for i in selected:
                X = feats_hwc[i].reshape(-1, C)
                X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)  # L2-norm
                km = KMeans(k, random_state=SEED, n_init=N_INIT, max_iter=100).fit(X)
                lbl = km.labels_
                sgi_vals.append(sgi_fn(lbl, k, H_f, W_f))
                sub = np.random.RandomState(SEED).choice(len(X), min(500,len(X)), replace=False)
                if len(np.unique(lbl[sub])) > 1:
                    sil_vals.append(silhouette_score(X[sub], lbl[sub]))
            print(f'k{k}', end=' ', flush=True)
            records.append({'backbone':bb,'dataset':ds,'k':k,'n_images':n_sel,
                'sgi_mean':float(np.mean(sgi_vals)),'sgi_std':float(np.std(sgi_vals)),
                'sil_mean':float(np.mean(sil_vals)) if sil_vals else float('nan'),
                'sil_std': float(np.std(sil_vals))  if sil_vals else float('nan')})
        print()

df = pd.DataFrame(records)
df.to_csv(OUT/'exp_k_ablation.csv', index=False)
summary = df.groupby(['backbone','k']).agg(
    sgi_mean_all=('sgi_mean','mean'), sgi_std_all=('sgi_mean','std'),
    sil_mean_all=('sil_mean','mean')).reset_index()
summary.to_csv(OUT/'exp_k_ablation_summary.csv', index=False)
print('\n=== SGI by k (mean 3 datasets) ===')
print(summary.pivot(index='backbone',columns='k',values='sgi_mean_all').round(5).to_string())
print('\n=== Silhouette by k ===')
print(summary.pivot(index='backbone',columns='k',values='sil_mean_all').round(4).to_string())
print('\nDONE')
