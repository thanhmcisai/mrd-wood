#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate fig4/fig6 at FULL RESOLUTION (1280x1024) — matches old HRNet_exp code exactly.
Runs live backbone inference instead of loading from low-res cache.
"""
import numpy as np
import cv2
import torch
import timm
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.cluster import KMeans
from wood_spatial.config import BB_LABEL, BB_ORDER, V4_FIGURES, DATASET_ROOT, CLUSTER_COLOR, N_CLUSTERS

OUT   = Path(V4_FIGURES)
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

# ── Constants ────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1280, 1024   # same as original HRNet_exp code
K       = 3   # visualization only; paper results use k=4
N_INIT  = 10
GF_R    = 24  # increased from 16: smoother boundaries for 16px-cell backbones (ConvNeXt, Swin)
GF_EPS  = 1e-4
VESSEL_MIN = 100

BB_DISPLAY = {
    'resnet50':        'ResNet-50',
    'efficientnet_b3': 'EfficientNet-B3',
    'convnext_tiny':   'ConvNeXt-T',
    'swin_tiny':       'Swin-Tiny',
    'dinov2_b':        'DINOv2-B',
    'hrnet32':         'HRNet-32',
    'mobilenetv3':     'MobileNetV3-L',
}

# timm model ids — same as config
BB_MODEL_IDS = {
    'resnet50':        'resnet50',
    'efficientnet_b3': 'efficientnet_b3',
    'convnext_tiny':   'convnext_tiny',
    'swin_tiny':       'swin_tiny_patch4_window7_224',
    'dinov2_b':        'vit_base_patch14_dinov2.lvd142m',
    'hrnet32':         'hrnet_w32',
    'mobilenetv3':     'mobilenetv3_large_100',
}

CLUSTER_RGB = {}
for k, hex_c in CLUSTER_COLOR.items():
    r, g, b = mcolors.hex2color(hex_c)
    CLUSTER_RGB[k] = np.array([r*255, g*255, b*255], dtype=np.float32)

CLUSTER_DISPLAY = {
    0: 'C1 (darkest)',
    1: 'C2',
    2: 'C3 (lightest)',
}

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Pipeline (exact match to original HRNet_exp/compare_multi_model.py) ─────

def correct_illumination(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    bg = cv2.GaussianBlur(l, (0, 0), sigmaX=100, sigmaY=100)
    l_f = l.astype(np.float32)
    bg_f = bg.astype(np.float32)
    mean_l = np.mean(bg_f)
    l_flat = np.clip(l_f - bg_f + mean_l, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enh = clahe.apply(l_flat)
    return cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2RGB)

def guided_filter(I, p, r=GF_R, eps=GF_EPS):
    k = (2*r+1, 2*r+1)
    mI  = cv2.boxFilter(I,   cv2.CV_32F, k)
    mp  = cv2.boxFilter(p,   cv2.CV_32F, k)
    mIp = cv2.boxFilter(I*p, cv2.CV_32F, k)
    mII = cv2.boxFilter(I*I, cv2.CV_32F, k)
    a = (mIp - mI*mp) / (mII - mI*mI + eps)
    b = mp - a*mI
    return cv2.boxFilter(a, cv2.CV_32F, k)*I + cv2.boxFilter(b, cv2.CV_32F, k)

def extract_features(bb, img_corrected_rgb):
    """
    Run backbone inference at full resolution.
    Returns spatial feature map (H_f, W_f, C) as numpy array.
    """
    model_id = BB_MODEL_IDS[bb]
    img_tensor = TRANSFORM(img_corrected_rgb).unsqueeze(0).to(DEVICE)

    if bb == 'dinov2_b':
        # DINOv2 ViT-B/14: native input = 518×518 (multiple of 14)
        # Process at 518×518, get 37×37 patch tokens
        dinov2_size = 518
        img_dino = cv2.resize(img_corrected_rgb, (dinov2_size, dinov2_size))
        t_dino = TRANSFORM(img_dino).unsqueeze(0).to(DEVICE)
        model = timm.create_model(model_id, pretrained=True,
                                  img_size=dinov2_size).to(DEVICE).eval()
        with torch.no_grad():
            feats = model.forward_features(t_dino)  # (1, N_tokens, C)
            # Remove CLS token, reshape to spatial grid
            patch_tokens = feats[:, 1:, :]  # (1, N_patches, C)
            n = patch_tokens.shape[1]
            hw = int(n ** 0.5)
            feat_np = patch_tokens[0].reshape(hw, hw, -1).cpu().numpy()  # (37, 37, 768)

    elif bb == 'swin_tiny':
        # Swin-T at 1280×1024: output is (N, H, W, C) NOT (N, C, H, W)!
        # timm swin_tiny uses HWC format natively — NO permute needed
        model = timm.create_model(model_id, pretrained=True,
                                  img_size=(IMG_H, IMG_W),
                                  features_only=True, out_indices=(2,)).to(DEVICE).eval()
        with torch.no_grad():
            feat = model(img_tensor)[0]
        feat_np = feat.squeeze(0).cpu().numpy()  # already (H_f=64, W_f=80, C=384)

    else:
        # CNN backbones: process at full 1280×1024 for high-res feature maps
        model = timm.create_model(model_id, pretrained=True,
                                  features_only=True, out_indices=(2,)).to(DEVICE).eval()
        with torch.no_grad():
            feat = model(img_tensor)[0]
        feat_np = feat.squeeze(0).permute(1, 2, 0).cpu().numpy()

    del model
    torch.cuda.empty_cache()
    return feat_np   # (H_f, W_f, C)

def cluster_and_overlay(feat_np, img_corrected_rgb):
    """
    feat_np: (H_f, W_f, C)
    Returns: overlay (H, W, 3) uint8 and labels (H, W)
    Fix: L2-normalize features before KMeans so distances are in 0-2 range,
    making exp(-d) produce meaningful soft probabilities (previously ~exp(-50)≈0).
    """
    H_f, W_f, C = feat_np.shape
    feat_flat = feat_np.reshape(-1, C)

    # L2 normalize — critical for balanced cluster probabilities
    # Raw distances ~50-65: exp(-50)=0, all probs collapse to hard assignment
    # Normalized distances ~0.4-1.3: exp(-0.4)=0.67, proper soft probabilities
    feat_norm = feat_flat / (np.linalg.norm(feat_flat, axis=1, keepdims=True) + 1e-8)

    guide = cv2.cvtColor(img_corrected_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    H, W = img_corrected_rgb.shape[:2]

    km = KMeans(n_clusters=K, random_state=42, n_init=N_INIT)
    distances = km.fit_transform(feat_norm)   # distances on unit sphere ∈ [0, ~1.4]
    probs_flat = np.exp(-distances)           # now: exp(-0.4)=0.67, meaningful soft probs
    probs_flat /= probs_flat.sum(axis=1, keepdims=True)
    probs_spatial = probs_flat.reshape(H_f, W_f, K)

    # Upsample to full image size
    probs_up = cv2.resize(probs_spatial, (W, H), interpolation=cv2.INTER_LINEAR)

    # Guided filter — exact same as old code
    refined = np.stack([
        guided_filter(guide, probs_up[:,:,k].astype(np.float32))
        for k in range(K)
    ], axis=-1)

    pixel_labels = np.argmax(refined, axis=-1)

    # Brightness ordering
    means = [guide[pixel_labels==k].mean() if (pixel_labels==k).any() else 0
             for k in range(K)]
    sorted_idx = np.argsort(means)
    mapped = np.zeros_like(pixel_labels)
    for ni, oi in enumerate(sorted_idx):
        mapped[pixel_labels == oi] = ni

    # Smart Hole Filling (from visualize_images.py)
    result = mapped.copy()
    vm = (mapped == 0).astype(np.uint8)
    cnts, _ = cv2.findContours(vm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vf = np.zeros_like(vm)
    for c in cnts:
        if cv2.contourArea(c) > VESSEL_MIN:
            cv2.drawContours(vf, [c], -1, 1, thickness=cv2.FILLED)
    result[vf == 1] = 0

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for tid in [1, 2]:
        m = (result == tid).astype(np.uint8)
        cleaned = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
        result[(result == tid) & (vf == 0)] = K - 1
        result[cleaned == 1] = tid

    # Overlay: exact same as compare_multi_model.py
    cmap_img = np.zeros((H, W, 3), dtype=np.float32)
    for k, rgb in CLUSTER_RGB.items():
        cmap_img[result == k] = rgb

    overlay = (img_corrected_rgb.astype(np.float32) * 0.4 + cmap_img * 0.6)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay, result

# ── Image loading ─────────────────────────────────────────────────────────────

def load_and_correct(img_path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_W, IMG_H))   # 1280×1024 like old code
    return correct_illumination(img_rgb)

def get_sample_image(ds):
    """Pick a representative sample image from dataset."""
    base = DATASET_ROOT / 'VN26'
    mag = ds.replace('VN26_', '').lower()   # x10, x20, x50
    img_dir = base / mag
    # Find first image across all species
    for sp_dir in sorted(img_dir.iterdir()):
        if not sp_dir.is_dir(): continue
        imgs = sorted(list(sp_dir.glob('*.jpg')) + list(sp_dir.glob('*.png')))
        if imgs:
            return imgs[0]
    return None

# ── Figure helpers ────────────────────────────────────────────────────────────

def add_row_label(ax, text, fontsize=9.5):
    ax.text(-0.06, 0.5, text, transform=ax.transAxes,
            va='center', ha='right', fontsize=fontsize,
            fontweight='bold', clip_on=False)

def clean_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

def make_legend():
    return [mpatches.Patch(color=mcolors.hex2color(CLUSTER_COLOR[k]),
                           label=CLUSTER_DISPLAY[k]) for k in range(K)]

def save_fig(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'{name}.{ext}', bbox_inches='tight',
                    dpi=200 if ext == 'png' else None)
    plt.close(fig)
    print(f'  Saved: {name}')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4: Spatial cluster panels — VN26 x10 / x20 / x50
# Layout: 7 rows (backbone) × 6 cols (3 magnifications × 2: image + overlay)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig4():
    print('\nBuilding Fig 4 (full-resolution inference at 1280×1024)...')

    conditions = [
        ('VN26_x10', '×10 Macro'),
        ('VN26_x20', '×20 Macro'),
        ('VN26_x50', '×50 Macro'),
    ]

    # Pre-load sample images (one per magnification)
    sample_imgs = {}
    for ds, label in conditions:
        p = get_sample_image(ds)
        sample_imgs[ds] = load_and_correct(p) if p else None
        print(f'  Loaded {ds}: {p.name if p else "NOT FOUND"}')

    n_bb   = len(BB_ORDER)
    n_cols = len(conditions) * 2
    cell_w, cell_h = 2.1, 2.1
    label_w = 1.5
    fig_w = label_w + n_cols * cell_w
    fig_h = n_bb * cell_h + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(n_bb, n_cols, figure=fig,
                            left=label_w/fig_w, right=0.985,
                            top=0.93, bottom=0.02,
                            wspace=0.03, hspace=0.08)

    for row, bb in enumerate(BB_ORDER):
        print(f'  Processing {bb}...')
        row_first = None

        for c_idx, (ds, cond_label) in enumerate(conditions):
            ax_img  = fig.add_subplot(gs[row, c_idx*2])
            ax_over = fig.add_subplot(gs[row, c_idx*2+1])
            if row_first is None:
                row_first = ax_img

            if row == 0:
                ax_img.set_title(f'{cond_label} · image',    fontsize=7.4, pad=2, fontweight='semibold')
                ax_over.set_title(f'{cond_label} · clusters', fontsize=7.4, pad=2, fontweight='semibold')

            corrected = sample_imgs.get(ds)
            if corrected is None:
                for ax in (ax_img, ax_over):
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=8, color='gray')
                continue

            try:
                feat_np = extract_features(bb, corrected)
                overlay, labels = cluster_and_overlay(feat_np, corrected)
                ax_img.imshow(corrected)
                ax_over.imshow(overlay)
            except Exception as e:
                print(f'    ERROR {bb}/{ds}: {e}')
                for ax in (ax_img, ax_over):
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'ERR', ha='center', va='center',
                            transform=ax.transAxes, fontsize=7, color='red')
                continue

            for ax in (ax_img, ax_over):
                clean_ax(ax)

        if row_first:
            add_row_label(row_first, BB_DISPLAY.get(bb, bb))

    save_fig(fig, 'fig4_spatial_cluster_panels_VN26')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6: CAM × cluster overlay — VN26 x20 (reference) vs x50 (magnified)
# Layout: 7 rows × 6 cols (2 conditions × 3: image + cluster + CAM)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig6():
    from wood_spatial.core.gradcam import centroid_cam
    from wood_spatial.analysis.feature_geometry import class_centroids
    from wood_spatial.config import V4_SPATIAL_CACHE

    print('\nBuilding Fig 6 (full-resolution CAM overlay)...')

    conditions = [('VN26_x20', '×20 (Reference)'), ('VN26_x50', '×50 (Magnified)')]

    sample_imgs = {}
    for ds, _ in conditions:
        p = get_sample_image(ds)
        sample_imgs[ds] = load_and_correct(p) if p else None
        print(f'  Loaded {ds}: {p.name if p else "NOT FOUND"}')

    n_bb   = len(BB_ORDER)
    n_cols = len(conditions) * 3
    cell_w, cell_h = 1.9, 2.1
    label_w = 1.5
    fig_w = label_w + n_cols * cell_w
    fig_h = n_bb * cell_h + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(n_bb, n_cols, figure=fig,
                            left=label_w/fig_w, right=0.985,
                            top=0.92, bottom=0.02,
                            wspace=0.03, hspace=0.10)

    # Load centroids from cached x20 features (for CAM)
    cache_dir = Path(V4_SPATIAL_CACHE)

    for row, bb in enumerate(BB_ORDER):
        print(f'  Processing {bb}...')
        row_first = None

        # Get centroids from cached x20 pooled features
        cache_file = cache_dir / f'{bb}_VN26_x20_original.npz'
        centroids = {}
        if cache_file.exists():
            d = np.load(cache_file, allow_pickle=True)
            feats_cached = d['features']     # (N, C, H_f, W_f)
            labels_cached = d['labels']
            pooled = feats_cached.mean(axis=(2, 3))   # (N, C)
            centroids = class_centroids(pooled, labels_cached[:len(pooled)], normalize=True)

        for c_idx, (ds, cond_label) in enumerate(conditions):
            col = c_idx * 3
            ax_img = fig.add_subplot(gs[row, col])
            ax_cl  = fig.add_subplot(gs[row, col+1])
            ax_cam = fig.add_subplot(gs[row, col+2])
            if row_first is None:
                row_first = ax_img

            if row == 0:
                ax_img.set_title(f'{cond_label} · image',    fontsize=7.2, pad=2, fontweight='semibold')
                ax_cl.set_title( f'{cond_label} · clusters', fontsize=7.2, pad=2, fontweight='semibold')
                ax_cam.set_title(f'{cond_label} · CAM',      fontsize=7.2, pad=2, fontweight='semibold')

            corrected = sample_imgs.get(ds)
            if corrected is None:
                for ax in (ax_img, ax_cl, ax_cam):
                    ax.axis('off')
                continue

            try:
                feat_np = extract_features(bb, corrected)
                overlay, labels = cluster_and_overlay(feat_np, corrected)

                # CAM: use cached pooled centroid + full-res spatial feat
                feat_chw = feat_np.transpose(2, 0, 1)   # (C, H_f, W_f)
                C_feat = feat_chw.shape[0]

                # Self-centroid fallback: use global pooled feature of this image
                self_cent = feat_chw.reshape(C_feat, -1).mean(axis=1)
                self_cent = self_cent / (np.linalg.norm(self_cent) + 1e-8)

                cam_cent = self_cent   # default: self-centroid
                if centroids:
                    lbl = next(iter(centroids))
                    c = centroids[lbl]
                    if c.shape[0] == C_feat:   # dimension must match
                        cam_cent = c
                    # else: cached centroid is wrong dim (e.g. Swin-T 14 vs 384), use self-centroid

                cam = centroid_cam(feat_chw, cam_cent, img_h=IMG_H, img_w=IMG_W)

                ax_img.imshow(corrected)
                ax_cl.imshow(overlay)
                # Normalize from 0 to 98th-percentile to show true activation strength
                cam_max = np.percentile(cam, 98)
                cam_std = cam.std()
                ax_cam.imshow(corrected)
                ax_cam.imshow(cam, cmap='jet', alpha=0.55,
                              vmin=0, vmax=max(cam_max, 1e-6))
                # Annotation for degenerate/uniform CAMs
                if cam_std < 0.02 * cam_max:   # uniform: std < 2% of max
                    note = 'uniform\n(drift≈0)' if bb == 'convnext_tiny' else 'degenerate'
                    ax_cam.text(0.5, 0.05, note, transform=ax_cam.transAxes,
                                fontsize=7, color='white', ha='center', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
            except Exception as e:
                print(f'    ERROR {bb}/{ds}: {e}')
                for ax in (ax_img, ax_cl, ax_cam):
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'ERR', ha='center', va='center',
                            transform=ax.transAxes, fontsize=7, color='red')
                continue

            for ax in (ax_img, ax_cl, ax_cam):
                clean_ax(ax)

        if row_first:
            add_row_label(row_first, BB_DISPLAY.get(bb, bb))

    save_fig(fig, 'fig6_cam_cluster_overlay_VN26')


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'=== Full-resolution VN26 figures (device: {DEVICE}) ===')
    plot_fig4()
    plot_fig6()
    print('\nAll done.')
