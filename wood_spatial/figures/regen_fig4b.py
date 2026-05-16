#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate fig4b: Spatial cluster panels — VN26 x20 with perturbations
Conditions: Clean | Gaussian Blur r=12 | Scratch severe | Field compound severe
k=3 clusters, no legend, full-res 1280x1024 inference.
"""
import numpy as np
import cv2
import io
import torch
import timm
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.cluster import KMeans
from wood_spatial.config import BB_ORDER, V4_FIGURES, DATASET_ROOT, CLUSTER_COLOR

OUT    = Path(V4_FIGURES)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_W, IMG_H = 1280, 1024
K      = 3
N_INIT = 10
GF_R   = 16
GF_EPS = 1e-4
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
for k_i, hex_c in CLUSTER_COLOR.items():
    if k_i >= K: continue
    r, g, b = mcolors.hex2color(hex_c)
    CLUSTER_RGB[k_i] = np.array([r*255, g*255, b*255], dtype=np.float32)

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Perturbation functions ───────────────────────────────────────────────────

def apply_blur(img_rgb, radius=12):
    ksize = 2 * radius + 1
    return cv2.GaussianBlur(img_rgb, (ksize, ksize), sigmaX=radius)

def apply_jpeg(img_rgb, quality=10):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), encode_param)
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def apply_illumination(img_rgb, factor=1.3):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def apply_compound_severe(img_rgb):
    """Blur r=12 + Illumination 1.3x + JPEG q=10 (same as paper definition)."""
    img = apply_blur(img_rgb, radius=12)
    img = apply_illumination(img, factor=1.3)
    img = apply_jpeg(img, quality=10)
    return img

def apply_scratch(img_rgb, severity='severe'):
    params = {
        'mild': (3, 1),
        'moderate': (6, 2),
        'severe': (12, 3),
    }
    n_lines, thickness = params[severity]
    seed = int(abs(float(img_rgb[::16, ::16].sum())) * 1000003) % (2**32 - 1)
    rng = np.random.RandomState(seed)
    out = img_rgb.copy()
    h, w = out.shape[:2]
    for _ in range(n_lines):
        x1 = int(rng.randint(0, w))
        y1 = int(rng.randint(0, h))
        length = int(rng.randint(max(8, w // 8), max(9, w // 2)))
        angle = float(rng.uniform(-np.pi, np.pi))
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
        gray = int(rng.choice([35, 220]))
        cv2.line(out, (x1, y1), (x2, y2), (gray, gray, gray), thickness, lineType=cv2.LINE_AA)
    return out

def apply_blue_shift(img_rgb, delta=-45):
    out = img_rgb.astype(np.int16).copy()
    out[:, :, 2] = np.clip(out[:, :, 2] + int(delta), 0, 255)
    return out.astype(np.uint8)

def apply_gaussian_noise(img_rgb, std=0.10):
    seed = int(abs(float(img_rgb[::16, ::16].sum())) * 1000033) % (2**32 - 1)
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, std * 255.0, img_rgb.shape)
    return np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def apply_zoom_blur(img_rgb, max_zoom=1.20):
    h, w = img_rgb.shape[:2]
    acc = img_rgb.astype(np.float32)
    for z in np.linspace(1.0, float(max_zoom), 5)[1:]:
        ch = max(1, int(h / z))
        cw = max(1, int(w / z))
        top = (h - ch) // 2
        left = (w - cw) // 2
        acc += cv2.resize(img_rgb[top:top + ch, left:left + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(acc / 5.0, 0, 255).astype(np.uint8)

def apply_compound_field_severe(img_rgb):
    img = apply_gaussian_noise(img_rgb, std=0.10)
    img = apply_blue_shift(img, delta=-45)
    img = apply_zoom_blur(img, max_zoom=1.20)
    img = apply_jpeg(img, quality=10)
    return img

# ── Pipeline (same as regen_figs_v3.py) ──────────────────────────────────────

def correct_illumination(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    bg = cv2.GaussianBlur(l, (0, 0), sigmaX=100, sigmaY=100)
    l_f = l.astype(np.float32)
    l_flat = np.clip(l_f - bg.astype(np.float32) + np.mean(bg.astype(np.float32)), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enh = clahe.apply(l_flat)
    return cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2RGB)

def guided_filter(I, p):
    k = (2*GF_R+1, 2*GF_R+1)
    mI = cv2.boxFilter(I, cv2.CV_32F, k)
    mp = cv2.boxFilter(p, cv2.CV_32F, k)
    a  = (cv2.boxFilter(I*p, cv2.CV_32F, k) - mI*mp) / \
         (cv2.boxFilter(I*I, cv2.CV_32F, k) - mI*mI + GF_EPS)
    b  = mp - a*mI
    return cv2.boxFilter(a, cv2.CV_32F, k)*I + cv2.boxFilter(b, cv2.CV_32F, k)

def extract_features(bb, img_corrected_rgb):
    model_id = BB_MODEL_IDS[bb]
    t = TRANSFORM(img_corrected_rgb).unsqueeze(0).to(DEVICE)
    if bb == 'dinov2_b':
        dino_size = 518
        img_d = cv2.resize(img_corrected_rgb, (dino_size, dino_size))
        t_d = TRANSFORM(img_d).unsqueeze(0).to(DEVICE)
        model = timm.create_model(model_id, pretrained=True,
                                  img_size=dino_size).to(DEVICE).eval()
        with torch.no_grad():
            feats = model.forward_features(t_d)
            patch = feats[:, 1:, :]
            hw = int(patch.shape[1]**0.5)
            feat_np = patch[0].reshape(hw, hw, -1).cpu().numpy()
    elif bb == 'swin_tiny':
        # Fix: use 1280×1024 (not 224px), and NO permute — Swin output is (N,H,W,C)
        model = timm.create_model(model_id, pretrained=True,
                                  img_size=(IMG_H, IMG_W),
                                  features_only=True, out_indices=(2,)).to(DEVICE).eval()
        with torch.no_grad():
            feat_np = model(t)[0].squeeze(0).cpu().numpy()  # (H=64,W=80,C=384)
    else:
        model = timm.create_model(model_id, pretrained=True,
                                  features_only=True, out_indices=(2,)).to(DEVICE).eval()
        with torch.no_grad():
            feat_np = model(t)[0].squeeze(0).permute(1,2,0).cpu().numpy()
    del model; torch.cuda.empty_cache()
    return feat_np

def cluster_and_overlay(feat_np, img_corrected_rgb):
    H_f, W_f, C = feat_np.shape
    feat_flat = feat_np.reshape(-1, C)
    # L2-normalize: raw distances ~50-65 → exp(-50)≈0 (collapse). Normalized → 0.4-1.3 → proper soft probs
    feat_norm = feat_flat / (np.linalg.norm(feat_flat, axis=1, keepdims=True) + 1e-8)
    guide = cv2.cvtColor(img_corrected_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    H, W = img_corrected_rgb.shape[:2]

    km = KMeans(n_clusters=K, random_state=42, n_init=N_INIT)
    distances = km.fit_transform(feat_norm)
    probs = np.exp(-distances)
    row_sums = probs.sum(axis=1, keepdims=True)
    probs /= np.where(row_sums < 1e-30, 1.0, row_sums)
    probs_up = cv2.resize(probs.reshape(H_f, W_f, K), (W, H), interpolation=cv2.INTER_LINEAR)

    refined = np.stack([guided_filter(guide, probs_up[:,:,k].astype(np.float32))
                        for k in range(K)], axis=-1)
    pixel_labels = np.argmax(refined, axis=-1)

    means = [guide[pixel_labels==k].mean() if (pixel_labels==k).any() else 0 for k in range(K)]
    sorted_idx = np.argsort(means)
    mapped = np.zeros_like(pixel_labels)
    for ni, oi in enumerate(sorted_idx):
        mapped[pixel_labels == oi] = ni

    # Smart Hole Filling
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
        if tid >= K: continue
        m = (result == tid).astype(np.uint8)
        cleaned = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
        result[(result == tid) & (vf == 0)] = K - 1
        result[cleaned == 1] = tid

    cmap_img = np.zeros((H, W, 3), dtype=np.float32)
    for k_i, rgb in CLUSTER_RGB.items():
        if k_i < K:
            cmap_img[result == k_i] = rgb

    overlay = np.clip(img_corrected_rgb.astype(np.float32)*0.4 + cmap_img*0.6, 0, 255).astype(np.uint8)
    return overlay

# ── Main figure ───────────────────────────────────────────────────────────────

def plot_fig4b_perturbation():
    print('Building Fig 4b — VN26 x20 perturbation visualization...')

    # Load one representative sample image
    img_dir = DATASET_ROOT / 'VN26' / 'x20'
    sample_path = None
    for sp in sorted(img_dir.iterdir()):
        if not sp.is_dir(): continue
        imgs = sorted(list(sp.glob('*.jpg')) + list(sp.glob('*.png')))
        if imgs:
            sample_path = imgs[0]
            break

    assert sample_path, 'No VN26 x20 image found'
    print(f'  Sample: {sample_path.name}')

    img_bgr = cv2.imread(str(sample_path))
    img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_orig = cv2.resize(img_rgb_orig, (IMG_W, IMG_H))

    # Define conditions: (label, perturbed_image_rgb)
    conditions = [
        ('Clean',                 img_rgb_orig.copy()),
        ('Gaussian Blur r=12',    apply_blur(img_rgb_orig, radius=12)),
        ('Scratch severe',        apply_scratch(img_rgb_orig, severity='severe')),
        ('Field compound severe', apply_compound_field_severe(img_rgb_orig)),
    ]

    n_bb   = len(BB_ORDER)
    n_cond = len(conditions)
    n_cols = n_cond * 2   # image + overlay per condition

    cell_w, cell_h = 2.0, 2.0
    label_w = 1.5
    fig_w = label_w + n_cols * cell_w
    fig_h = n_bb * cell_h + 0.5   # tight top, no legend bottom

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(n_bb, n_cols, figure=fig,
                            left=label_w/fig_w, right=0.985,
                            top=0.985, bottom=0.02,
                            wspace=0.03, hspace=0.08)

    for row, bb in enumerate(BB_ORDER):
        print(f'  {bb}...', end=' ', flush=True)
        row_first = None

        for c_idx, (cond_label, img_perturbed) in enumerate(conditions):
            ax_img  = fig.add_subplot(gs[row, c_idx*2])
            ax_over = fig.add_subplot(gs[row, c_idx*2+1])
            if row_first is None:
                row_first = ax_img

            if row == 0:
                ax_img.set_title(f'{cond_label} · image',    fontsize=7.4, pad=2, fontweight='semibold')
                ax_over.set_title(f'{cond_label} · clusters', fontsize=7.4, pad=2, fontweight='semibold')

            try:
                corrected = correct_illumination(img_perturbed)
                feat_np   = extract_features(bb, corrected)
                overlay   = cluster_and_overlay(feat_np, corrected)
                ax_img.imshow(corrected)
                ax_over.imshow(overlay)
            except Exception as e:
                print(f'ERR({e})', end=' ')
                for ax in (ax_img, ax_over):
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'ERR', ha='center', va='center',
                            transform=ax.transAxes, fontsize=7, color='red')
                continue

            for ax in (ax_img, ax_over):
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values(): s.set_visible(False)

        if row_first:
            row_first.text(-0.06, 0.5, BB_DISPLAY.get(bb, bb),
                           transform=row_first.transAxes,
                           va='center', ha='right',
                           fontsize=9, fontweight='bold', clip_on=False)
        print('done')

    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'fig4b_perturbation_VN26x20.{ext}',
                    bbox_inches='tight', dpi=200 if ext == 'png' else None)
    plt.close(fig)
    print('  Saved: fig4b_perturbation_VN26x20')

if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    plot_fig4b_perturbation()
    print('Done.')
