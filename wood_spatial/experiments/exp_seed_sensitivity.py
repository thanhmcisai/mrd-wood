#!/usr/bin/env python3
"""Seed sensitivity v3 — L2-norm + Swin-T HWC cache fix."""
import numpy as np, pandas as pd
from collections import defaultdict
from scipy.ndimage import label as ndlabel
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import sys as _sys, warnings; warnings.filterwarnings('ignore')
from wood_spatial.config import BB_ORDER, TIER_A, V4_SPATIAL_CACHE, V4_CSV, N_CLUSTERS
class Unbuffered:
    def __init__(self,s): self.s=s
    def write(self,d): self.s.write(d); self.s.flush()
    def __getattr__(self,a): return getattr(self.s,a)
_sys.stdout = Unbuffered(_sys.stdout)

SCACHE = V4_SPATIAL_CACHE
OUT    = V4_CSV
BACKBONES = BB_ORDER
K=N_CLUSTERS; SEEDS=list(range(10)); N_PER_CLASS=3; N_INIT=5; SEED=42

def sgi_fn(labels, H, W):
    return sum(ndlabel((labels==k).reshape(H,W))[1] for k in range(K)) / (H*W)

def load_cache(bb, ds):
    """Load spatial cache, handling Swin-T HWC format."""
    sf = SCACHE / f'{bb}_{ds}_original.npz'
    if not sf.exists(): return None, None, None, None
    d = np.load(sf, allow_pickle=True)
    feats = d['features']; labels = d['labels']
    # Detect HWC (Swin-T) vs CHW (CNN): if last dim > dim 1 → HWC
    if feats.shape[-1] > feats.shape[1]:  # (N, H, W, C) HWC format
        N, H_f, W_f, C = feats.shape
    else:  # (N, C, H_f, W_f) CHW format
        N, C, H_f, W_f = feats.shape
        feats = feats.transpose(0, 2, 3, 1)  # → (N, H_f, W_f, C)
    return feats, labels, H_f, W_f

records = []
print(f'Scope: {len(BACKBONES)} BB x {len(TIER_A)} DS x N_PER_CLASS={N_PER_CLASS} x {len(SEEDS)} seeds')
print(f'Fixes: L2-norm before KMeans, Swin-T HWC cache detection')

for bb in BACKBONES:
    print(f'\n=== {bb} ===')
    all_sgi_per_seed = defaultdict(list)
    n_unique = 0
    for ds in TIER_A:
        feats_hwc, labels_sc, H_f, W_f = load_cache(bb, ds)
        if feats_hwc is None: print(f'  {ds}: no cache'); continue
        N, _, _, C = feats_hwc.shape
        le = LabelEncoder(); yc = le.fit_transform(labels_sc)
        selected = []
        for cls in np.unique(yc):
            idx = np.where(yc==cls)[0]
            chosen = np.random.RandomState(SEED).choice(idx, min(N_PER_CLASS,len(idx)), replace=False)
            selected.extend(chosen.tolist())
        selected = sorted(selected)
        n_unique += len(selected)
        print(f'  {ds}: {len(selected)} imgs ({len(np.unique(yc))} classes), H={H_f} W={W_f} C={C}', flush=True)
        for i in selected:
            X = feats_hwc[i].reshape(-1, C)  # (H*W, C)
            # L2-normalize
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            for seed in SEEDS:
                km = KMeans(K, random_state=seed, n_init=N_INIT, max_iter=100)
                km.fit(X)
                lbl = km.predict(X)
                all_sgi_per_seed[seed].append(sgi_fn(lbl, H_f, W_f))
    seed_means = [np.mean(all_sgi_per_seed[s]) for s in SEEDS if all_sgi_per_seed[s]]
    if not seed_means: continue
    mean_sgi = np.mean(seed_means); std_sgi = np.std(seed_means)
    cv_pct = std_sgi/mean_sgi*100 if mean_sgi>0 else 0
    ci_95  = 1.96*std_sgi/np.sqrt(len(seed_means))
    print(f'  Summary: mean={mean_sgi:.6f}, CV={cv_pct:.2f}%, n_unique={n_unique}')
    for seed in SEEDS:
        if all_sgi_per_seed[seed]:
            records.append({'backbone':bb,'seed':seed,'mean_sgi':np.mean(all_sgi_per_seed[seed]),
                'std_sgi':np.std(all_sgi_per_seed[seed]),'n_unique_images':n_unique,
                'n_sgi_measurements':len(all_sgi_per_seed[seed])})

df = pd.DataFrame(records)
df.to_csv(OUT/'exp2_seed_sensitivity.csv', index=False)
summary = df.groupby('backbone').agg(
    mean_sgi=('mean_sgi','mean'), std_across_seeds=('mean_sgi','std'),
    cv_pct=('mean_sgi', lambda x: x.std()/x.mean()*100 if x.mean()>0 else 0),
    ci_95=('mean_sgi', lambda x: 1.96*x.std()/np.sqrt(len(x))),
    n_unique_images=('n_unique_images','first')).reset_index().sort_values('mean_sgi', ascending=False)
summary.to_csv(OUT/'exp2_seed_sensitivity_summary.csv', index=False)
print('\n=== SUMMARY ===')
print(summary[['backbone','mean_sgi','cv_pct','ci_95','n_unique_images']].to_string(index=False))
print('\nDONE')
