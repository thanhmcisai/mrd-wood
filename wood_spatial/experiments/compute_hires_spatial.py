#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-resolution spatial metrics v4 — N_PER_CLASS=3:
- 3 images per species, proper class centroid (mean of 3 pooled features)
- Each image evaluated individually → within-class variance captured
- All 7 backbones × 3 Tier-A datasets × 3 imgs/class × k=3 × 5 conditions

Deprecated:
    This script is kept only for provenance of the older v4 high-resolution
    prototype. The maintained high-resolution pipeline is split into
    exp_hires_extract.py and exp_hires_metrics_from_cache.py and uses the
    expanded perturbation configuration.
"""
raise SystemExit(
    'compute_hires_spatial.py is deprecated. Use '
    'python -m wood_spatial.experiments.exp_hires_extract followed by '
    'python -m wood_spatial.experiments.exp_hires_metrics_from_cache.'
)
import numpy as np
import pandas as pd
import cv2
import torch
import timm
import torchvision.transforms as T
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import label as ndlabel
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import time
import warnings; warnings.filterwarnings('ignore')
import sys as _sys

class Unbuffered:
    def __init__(self, s): self.s = s
    def write(self, d): self.s.write(d); self.s.flush()
    def __getattr__(self, a): return getattr(self.s, a)
_sys.stdout = Unbuffered(_sys.stdout)

from wood_spatial.config import BB_ORDER, TIER_A, V4_FEAT_CACHE, V4_CSV

FCACHE = V4_FEAT_CACHE
OUT    = V4_CSV
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

BACKBONES = BB_ORDER
DATASETS  = TIER_A
IMG_W, IMG_H = 1280, 1024
DINO_SIZE    = 518
K = 3; N_INIT = 3; GF_R = 16; GF_EPS = 1e-4
N_FIT = 2000
N_PER_CLASS = 3   # 3 images per class → proper within-class variance

BB_MODEL_IDS = {
    'resnet50':        'resnet50',
    'efficientnet_b3': 'efficientnet_b3',
    'convnext_tiny':   'convnext_tiny',
    'swin_tiny':       'swin_tiny_patch4_window7_224',
    'dinov2_b':        'vit_base_patch14_dinov2.lvd142m',
    'hrnet32':         'hrnet_w32',
    'mobilenetv3':     'mobilenetv3_large_100',
}
TRANSFORM = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# ── Image helpers ──────────────────────────────────────────────────────────────
def correct_illumination(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    h, w = l.shape
    small = cv2.resize(l.astype(np.float32), (max(w//8,8), max(h//8,8)))
    bg    = cv2.resize(cv2.GaussianBlur(small,(0,0),sigmaX=12.5), (w,h))
    lf    = np.clip(l.astype(np.float32)-bg+np.mean(bg), 0, 255).astype(np.uint8)
    l2    = cv2.createCLAHE(2.0,(8,8)).apply(lf)
    return cv2.cvtColor(cv2.merge((l2,a,b)), cv2.COLOR_LAB2RGB)

def guided_filter(I, p):
    k = (2*GF_R+1, 2*GF_R+1)
    mI = cv2.boxFilter(I, cv2.CV_32F, k)
    mp = cv2.boxFilter(p, cv2.CV_32F, k)
    a  = (cv2.boxFilter(I*p,cv2.CV_32F,k)-mI*mp)/(cv2.boxFilter(I*I,cv2.CV_32F,k)-mI*mI+GF_EPS)
    b_ = mp-a*mI
    return cv2.boxFilter(a,cv2.CV_32F,k)*I + cv2.boxFilter(b_,cv2.CV_32F,k)

def apply_blur(img, r): return cv2.GaussianBlur(img,(2*r+1,2*r+1),sigmaX=r)
def apply_jpeg(img, q=10):
    _, buf = cv2.imencode('.jpg',cv2.cvtColor(img,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,q])
    return cv2.cvtColor(cv2.imdecode(buf,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
def compound_severe(img): return apply_jpeg(apply_blur(img,12))

CONDITIONS = {
    'clean':    lambda x: x,
    'blur_r4':  lambda x: apply_blur(x,4),
    'blur_r8':  lambda x: apply_blur(x,8),
    'blur_r12': lambda x: apply_blur(x,12),
    'compound': compound_severe,
}

# ── Model ──────────────────────────────────────────────────────────────────────
_models = {}
def get_model(bb):
    if bb not in _models:
        mid = BB_MODEL_IDS[bb]
        if bb == 'dinov2_b':
            m = timm.create_model(mid,pretrained=True,img_size=DINO_SIZE).to(DEVICE).eval()
        elif bb == 'swin_tiny':
            m = timm.create_model(mid,pretrained=True,img_size=(IMG_H,IMG_W),
                                  features_only=True,out_indices=(2,)).to(DEVICE).eval()
        else:
            m = timm.create_model(mid,pretrained=True,
                                  features_only=True,out_indices=(2,)).to(DEVICE).eval()
        _models[bb] = m
    return _models[bb]

def get_size(bb): return (DINO_SIZE,DINO_SIZE) if bb=='dinov2_b' else (IMG_W,IMG_H)

def extract_spatial(bb, rgb):
    m = get_model(bb); w,h = get_size(bb)
    t = TRANSFORM(cv2.resize(rgb,(w,h))).unsqueeze(0).to(DEVICE)
    if bb == 'dinov2_b':
        with torch.no_grad():
            f = m.forward_features(t)[:,1:,:]
            hw = int(f.shape[1]**0.5)
            return f[0].reshape(hw,hw,-1).cpu().numpy()
    if bb == 'swin_tiny':
        # Swin-T returns (N, H, W, C) — no permute needed
        with torch.no_grad():
            return m(t)[0].squeeze(0).cpu().numpy()  # (H_f, W_f, C)
    with torch.no_grad():
        return m(t)[0].squeeze(0).permute(1,2,0).cpu().numpy()  # CNN: (C,H,W) → (H,W,C)

# ── Spatial metrics ────────────────────────────────────────────────────────────
def cluster_map(feat_np, guide):
    H_f,W_f,C = feat_np.shape; X = feat_np.reshape(-1,C)
    # L2-normalize: raw distances ~50-65 → exp(-50)≈0 (collapse to hard assignment)
    # Normalized distances ~0.4-1.3 → exp(-0.7)≈0.5 (meaningful soft probs)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    idx = np.random.RandomState(42).choice(len(X),min(N_FIT,len(X)),replace=False)
    km = KMeans(K,random_state=42,n_init=N_INIT,max_iter=100).fit(X[idx])
    dists = km.transform(X)
    probs = np.exp(-dists); probs /= probs.sum(axis=1,keepdims=True)
    out_w,out_h = guide.shape[1],guide.shape[0]
    probs_up = cv2.resize(probs.reshape(H_f,W_f,K),(out_w,out_h),interpolation=cv2.INTER_LINEAR)
    refined  = np.stack([guided_filter(guide,probs_up[:,:,k].astype(np.float32)) for k in range(K)],axis=-1)
    labels   = np.argmax(refined,axis=-1)
    means    = [guide[labels==k].mean() if (labels==k).any() else 0 for k in range(K)]
    order    = np.argsort(means)
    mapped   = np.zeros_like(labels)
    for ni,oi in enumerate(order): mapped[labels==oi]=ni
    return mapped

def sgi(labels): return sum(ndlabel((labels==k))[1] for k in range(K))/labels.size
def csi(lc,lp):
    ious=[]
    for k in range(K):
        A,B=lc==k,lp==k; u=(A|B).sum()
        ious.append((A&B).sum()/u if u>0 else 1.0)
    return float(np.mean(ious))

def cam_dist(feat_np, centroid, labels):
    H_f,W_f,C = feat_np.shape
    cent_n = centroid/(np.linalg.norm(centroid)+1e-8)
    scores = np.clip(feat_np.reshape(-1,C)@cent_n,0,None).reshape(H_f,W_f)
    out_h,out_w = labels.shape
    cam = cv2.resize(scores.astype(np.float32),(out_w,out_h),interpolation=cv2.INTER_LINEAR)
    total = cam.sum()+1e-8
    return np.array([cam[labels==k].sum()/total for k in range(K)])

def entropy(d): d=d+1e-10; d/=d.sum(); return float(-np.sum(d*np.log2(d)))
def jsd_sym(p,q):
    p=p+1e-10; q=q+1e-10; p/=p.sum(); q/=q.sum()
    return float(jensenshannon(p,q))

# ── Sample N_PER_CLASS images per class ───────────────────────────────────────
def sample(bb, ds):
    f = FCACHE/f'{bb}_{ds}_original.npz'
    if not f.exists(): return {}   # cls → [path1, path2, ...]
    d = np.load(f,allow_pickle=True)
    paths,labels = d['paths'],d['labels']
    le=LabelEncoder(); yc=le.fit_transform(labels)
    cls_imgs = defaultdict(list)
    for cls in np.unique(yc):
        idx = np.where(yc==cls)[0]
        chosen = np.random.RandomState(42).choice(idx,min(N_PER_CLASS,len(idx)),replace=False)
        for i in chosen:
            cls_imgs[cls].append(paths[i])
    return cls_imgs   # {cls: [path, ...]}

# ── Main ────────────────────────────────────────────────────────────────────────
records = []
n_imgs_per_bb = sum([25,14,11])*N_PER_CLASS
print(f'Scope: {len(BACKBONES)} BB × {len(DATASETS)} DS × {n_imgs_per_bb} imgs × {len(CONDITIONS)} conditions')
print(f'N_PER_CLASS={N_PER_CLASS} → class centroid averaged over {N_PER_CLASS} images')

for bb in BACKBONES:
    t_bb = time.time()
    print(f'\n=== {bb} ===')
    for ds in DATASETS:
        cls_imgs = sample(bb, ds)
        if not cls_imgs: print(f'  {ds}: no data'); continue
        n_classes = len(cls_imgs)
        n_imgs    = sum(len(v) for v in cls_imgs.values())
        print(f'  {ds} ({n_classes} classes, {n_imgs} imgs): ', end='')

        # ── Pass 1: build per-class centroid (mean of N_PER_CLASS pooled features) ──
        class_centroids = {}
        for cls, paths in cls_imgs.items():
            pooled_list = []
            for path in paths:
                img_bgr = cv2.imread(str(path))
                if img_bgr is None: continue
                raw  = cv2.resize(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB),(IMG_W,IMG_H))
                corr = correct_illumination(raw)
                feat = extract_spatial(bb, corr)
                p = feat.mean(axis=(0,1)); p /= np.linalg.norm(p)+1e-8
                pooled_list.append(p)
            if pooled_list:
                cent = np.mean(pooled_list, axis=0)
                cent /= np.linalg.norm(cent)+1e-8
                class_centroids[cls] = cent

        # ── Pass 2: compute metrics for each image × each condition ──
        n_done = 0
        for cls, paths in cls_imgs.items():
            cent = class_centroids.get(cls)
            if cent is None: continue

            for path in paths:
                img_bgr = cv2.imread(str(path))
                if img_bgr is None: continue
                raw  = cv2.resize(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB),(IMG_W,IMG_H))
                corr = correct_illumination(raw)

                feats_by_cond  = {}
                labels_by_cond = {}
                for cond_name, perturb_fn in CONDITIONS.items():
                    perturbed = perturb_fn(corr)
                    guide = cv2.cvtColor(perturbed,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
                    feat  = extract_spatial(bb, perturbed)
                    lbl   = cluster_map(feat, guide)
                    feats_by_cond[cond_name]  = feat
                    labels_by_cond[cond_name] = lbl

                lc = labels_by_cond['clean']
                fc = feats_by_cond['clean']
                dist_c = cam_dist(fc, cent, lc)
                H_c    = entropy(dist_c)
                gc_pool = fc.mean(axis=(0,1)); gc_pool /= np.linalg.norm(gc_pool)+1e-8

                for cond_name in CONDITIONS:
                    lp = labels_by_cond[cond_name]
                    fp = feats_by_cond[cond_name]
                    dist_p = cam_dist(fp, cent, lp)
                    H_p    = entropy(dist_p)
                    gp_pool = fp.mean(axis=(0,1)); gp_pool /= np.linalg.norm(gp_pool)+1e-8

                    records.append({
                        'backbone':bb, 'dataset':ds, 'class':cls,
                        'image':Path(path).name, 'condition':cond_name,
                        'sgi':           sgi(lp),
                        'csi':           csi(lc, lp),
                        'feature_drift': float(1-np.dot(gc_pool,gp_pool)),
                        'cam_entropy':   H_p,
                        'delta_entropy': H_p - H_c,
                        'cam_jsd':       jsd_sym(dist_c, dist_p),
                    })
                n_done += 1

        elapsed = time.time()-t_bb
        print(f'{n_done} imgs done  [{elapsed:.0f}s]')

df = pd.DataFrame(records)
df.to_csv(OUT/'exp_hires_spatial_metrics.csv', index=False)
print(f'\nSaved: {len(df)} rows → exp_hires_spatial_metrics.csv')

print('\n=== CLEAN — SGI + CAM Entropy ===')
print(df[df.condition=='clean'].groupby('backbone').agg(
    SGI=('sgi','mean'), SGI_std=('sgi','std'), CAM_H=('cam_entropy','mean')
).sort_values('SGI',ascending=False).round(6))

print('\n=== BLUR — CSI + CAM_JSD + Drift ===')
print(df[df.condition.str.startswith('blur')].groupby('backbone').agg(
    CSI=('csi','mean'), CSI_std=('csi','std'), CAM_JSD=('cam_jsd','mean'), Drift=('feature_drift','mean')
).sort_values('CSI',ascending=False).round(4))

print('\nDONE')
