#!/usr/bin/env python3
"""Rerun Swin-T ANOVA with correct HWC cache reading + L2-norm."""
import numpy as np, pandas as pd
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import sys as _sys, warnings; warnings.filterwarnings('ignore')
from wood_spatial.config import PERTURB_CONFIGS, V4_SPATIAL_CACHE, V4_CSV, TIER_A, N_CLUSTERS
from wood_spatial.core.perturbations import cache_tag_for
class Unbuffered:
    def __init__(self,s): self.s=s
    def write(self,d): self.s.write(d); self.s.flush()
    def __getattr__(self,a): return getattr(self.s,a)
_sys.stdout = Unbuffered(_sys.stdout)

SCACHE = V4_SPATIAL_CACHE
OUT    = V4_CSV
K=N_CLUSTERS; N_INIT=5; SEED=42; N_PER_CLASS=3

# Map every configured cache tag to its perturbation family. Missing spatial
# caches are skipped below, so this remains compatible with representative
# spatial subsets as well as the expanded full perturbation set.
PERTURB_GROUPS = {
    cache_tag_for(pert_name, value): pert_name
    for pert_name, pcfg in PERTURB_CONFIGS.items()
    for value in pcfg['values']
}

def load_swint_cache(ds, tag):
    """Load Swin-T cache with correct HWC format."""
    f = SCACHE / f'swin_tiny_{ds}_{tag}.npz'
    if not f.exists(): return None, None
    d = np.load(f, allow_pickle=True)
    feats = d['features']  # (N, 14, 14, 384) HWC
    labels = d['labels']
    # HWC detected: last dim (384) > dim 1 (14)
    if feats.shape[-1] > feats.shape[1]:
        N, H_f, W_f, C = feats.shape
    else:
        N, C, H_f, W_f = feats.shape
        feats = feats.transpose(0,2,3,1)
    return feats, labels  # (N, H_f, W_f, C)

def get_sample_indices(labels_sc, n_per_class=N_PER_CLASS):
    le = LabelEncoder(); yc = le.fit_transform(labels_sc)
    selected = []
    for cls in np.unique(yc):
        idx = np.where(yc==cls)[0]
        chosen = np.random.RandomState(SEED).choice(idx, min(n_per_class,len(idx)), replace=False)
        selected.extend(chosen.tolist())
    return sorted(selected)

def compute_cluster_labels(feat_hwc_img):
    """Single image (H_f, W_f, C) → cluster label map (H_f*W_f,)."""
    H_f, W_f, C = feat_hwc_img.shape
    X = feat_hwc_img.reshape(-1, C)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    km = KMeans(K, random_state=SEED, n_init=N_INIT, max_iter=100).fit(X)
    return km.labels_  # (H_f*W_f,)

def compute_csi(lc, lp):
    ious = []
    for k in range(K):
        A,B = lc==k, lp==k; u=(A|B).sum()
        ious.append((A&B).sum()/u if u>0 else 1.0)
    return float(np.mean(ious))

print("Recomputing Swin-T CSI with correct HWC features + L2-norm...")
records = []

for ds in TIER_A:
    # Load original (clean) cache
    # Load original and find min cache size across all perturbations
    feats_clean, labels_sc = load_swint_cache(ds, "original")
    if feats_clean is None: print(f'{ds}: no original cache'); continue
    # Find min cache size across perturbation tags
    min_n = feats_clean.shape[0]
    for tag in PERTURB_GROUPS:
        f2 = SCACHE / f'swin_tiny_{ds}_{tag}.npz'
        if f2.exists():
            d2 = np.load(f2, allow_pickle=True)
            min_n = min(min_n, d2['features'].shape[0])
    selected = get_sample_indices(labels_sc[:min_n])
    print(f'{ds}: {len(selected)} imgs (min_n={min_n})', flush=True)
    
    # Compute clean cluster labels for selected images
    clean_labels = {}
    for i in selected:
        clean_labels[i] = compute_cluster_labels(feats_clean[i])
    
    # For each perturbation tag
    for tag, ptype in PERTURB_GROUPS.items():
        feats_pert, _ = load_swint_cache(ds, tag)
        if feats_pert is None: continue
        
        for i in selected:
            lc = clean_labels[i]
            lp = compute_cluster_labels(feats_pert[i])
            records.append({
                'backbone':'swin_tiny','dataset':ds,
                'perturbation':ptype,'tag':tag,
                'image_idx':i,'csi':compute_csi(lc,lp)
            })

df = pd.DataFrame(records)
print(f"\nTotal records: {len(df)}")

# ANOVA: does perturbation TYPE significantly affect CSI?
print("\n=== Swin-T CSI per perturbation type ===")
by_type = df.groupby('perturbation')['csi'].agg(['mean','std','count'])
print(by_type.round(4))

groups = [df[df.perturbation==pt]['csi'].values for pt in df.perturbation.unique()]
F_stat, p_val = f_oneway(*groups)
sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))
print(f"\nANOVA: F={F_stat:.4f}, p={p_val:.4e} → {sig}")

# Update the anova CSV
anova_df = pd.read_csv(OUT/'exp_swint_csi_anova.csv')
swint_row_idx = anova_df[anova_df.backbone=='swin_tiny'].index[0]
anova_df.loc[swint_row_idx,'F'] = F_stat
anova_df.loc[swint_row_idx,'p'] = p_val
anova_df.loc[swint_row_idx,'sig'] = sig
anova_df.to_csv(OUT/'exp_swint_csi_anova.csv', index=False)
print(f"\nUpdated exp_swint_csi_anova.csv:")
print(anova_df.to_string())

# Also save detailed Swin-T CSI for reference
df.groupby(['dataset','perturbation'])['csi'].mean().round(4).to_csv(
    OUT/'exp_swint_csi_corrected.csv')
print(f"\nSaved: exp_swint_csi_corrected.csv")
print("DONE")
