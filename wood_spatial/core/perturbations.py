"""
Wood Spatial — Perturbation Transforms
========================================
All perturbation classes operate on denormalized [0, 1] float tensors (C x H x W).

Existing (adapted from v2):
    ChannelShift, ResizeZoom, GaussianBlurPerturb, RotationPerturb, BrightnessContrast

New:
    DefocusBlurPerturb  — disk PSF convolution (realistic optical defocus)
    JPEGCompression     — encode/decode JPEG at given quality
    CompoundPerturb     — sequential composition of multiple perturbations

Factory:
    make_perturbation(pcfg, value, img_size) — construct from config entry
    cache_tag_for(pert_name, value)          — generate cache filename tag
"""
import io
import logging

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

from wood_spatial.config import COMPOUND_PRESETS, IMG_SIZE

logger = logging.getLogger(__name__)


# ── Existing perturbation classes (from v2) ─────────────────────────────────

class ChannelShift:
    """Shift a single RGB channel by delta/255 (clipped to [0, 1])."""

    def __init__(self, channel: int, delta: float):
        self.channel = channel
        self.delta = delta

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[self.channel] = (x[self.channel] + self.delta / 255.0).clamp(0, 1)
        return x

    def __repr__(self):
        ch = {0: 'R', 1: 'G', 2: 'B'}.get(self.channel, str(self.channel))
        return f'ChannelShift({ch},{self.delta:+.0f})'


class ResizeZoom:
    """Zoom into center by cropping to 1/factor and resizing back."""

    def __init__(self, factor: float, output_size: int = IMG_SIZE):
        self.factor = factor
        self.out = output_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        ch = max(1, int(h / self.factor))
        cw = max(1, int(w / self.factor))
        t = (h - ch) // 2
        l = (w - cw) // 2
        return TF.resize(
            x[:, t:t + ch, l:l + cw],
            [self.out, self.out],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def __repr__(self):
        return f'ResizeZoom({self.factor:.2f}x)'


class GaussianBlurPerturb:
    """Apply PIL GaussianBlur of given radius."""

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pil = TF.to_pil_image(x)
        return TF.to_tensor(pil.filter(ImageFilter.GaussianBlur(radius=self.radius)))

    def __repr__(self):
        return f'GaussianBlurPerturb(r={self.radius})'


class RotationPerturb:
    """Rotate image by a fixed angle (degrees)."""

    def __init__(self, angle: float):
        self.angle = angle

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return TF.rotate(x, self.angle)

    def __repr__(self):
        return f'RotationPerturb({self.angle}deg)'


class BrightnessContrast:
    """Multiply all channels by a scalar factor (clipped to [0, 1])."""

    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.factor).clamp(0, 1)

    def __repr__(self):
        return f'BrightnessContrast(x{self.factor:.2f})'


# ── NEW perturbation classes ────────────────────────────────────────────────

class GaussianNoisePerturb:
    """Add zero-mean Gaussian noise with standard deviation in [0, 1]."""

    def __init__(self, std: float):
        self.std = float(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x + torch.randn_like(x) * self.std).clamp(0, 1)

    def __repr__(self):
        return f'GaussianNoisePerturb(std={self.std:.3f})'


class ShotNoisePerturb:
    """Apply signal-dependent Poisson noise; lower level means stronger noise."""

    def __init__(self, level: float):
        self.level = float(level)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        level = max(self.level, 1.0)
        return (torch.poisson(x.clamp(0, 1) * level) / level).clamp(0, 1)

    def __repr__(self):
        return f'ShotNoisePerturb(level={self.level:.0f})'


class ImpulseNoisePerturb:
    """Apply salt-and-pepper impulse noise with probability p per pixel."""

    def __init__(self, probability: float):
        self.probability = float(probability)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p = min(max(self.probability, 0.0), 1.0)
        if p <= 0:
            return x
        out = x.clone()
        rnd = torch.rand((1, x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype)
        pepper = rnd < (p / 2.0)
        salt = (rnd >= (p / 2.0)) & (rnd < p)
        out = torch.where(pepper.expand_as(out), torch.zeros_like(out), out)
        out = torch.where(salt.expand_as(out), torch.ones_like(out), out)
        return out.clamp(0, 1)

    def __repr__(self):
        return f'ImpulseNoisePerturb(p={self.probability:.3f})'


class MotionBlurPerturb:
    """Apply a horizontal motion-blur kernel."""

    def __init__(self, kernel_size: int):
        self.kernel_size = int(kernel_size)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = max(3, self.kernel_size)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        img_np = x.permute(1, 2, 0).cpu().numpy()
        blurred = cv2.filter2D(img_np, -1, kernel)
        return torch.from_numpy(blurred).permute(2, 0, 1).to(x.device).clamp(0, 1)

    def __repr__(self):
        return f'MotionBlurPerturb(k={self.kernel_size})'


class ZoomBlurPerturb:
    """Average the image with progressively center-zoomed copies."""

    def __init__(self, max_zoom: float, output_size: int = IMG_SIZE):
        self.max_zoom = float(max_zoom)
        self.out = output_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        zooms = np.linspace(1.0, self.max_zoom, 5)[1:]
        _, h, w = x.shape
        out = x.clone()
        for z in zooms:
            ch = max(1, int(h / float(z)))
            cw = max(1, int(w / float(z)))
            top = (h - ch) // 2
            left = (w - cw) // 2
            zoomed = TF.resize(
                x[:, top:top + ch, left:left + cw],
                [h, w],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            out = out + zoomed
        return (out / (len(zooms) + 1)).clamp(0, 1)

    def __repr__(self):
        return f'ZoomBlurPerturb(max_zoom={self.max_zoom:.2f})'


class ContrastPerturb:
    """Adjust contrast around the per-channel image mean."""

    def __init__(self, factor: float):
        self.factor = float(factor)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return ((x - mean) * self.factor + mean).clamp(0, 1)

    def __repr__(self):
        return f'ContrastPerturb(x{self.factor:.2f})'


class PixelatePerturb:
    """Downsample by ratio and resize back with nearest-neighbor interpolation."""

    def __init__(self, ratio: float):
        self.ratio = float(ratio)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        ph = max(1, int(h * self.ratio))
        pw = max(1, int(w * self.ratio))
        small = TF.resize(
            x, [ph, pw],
            interpolation=TF.InterpolationMode.NEAREST,
            antialias=False,
        )
        return TF.resize(
            small, [h, w],
            interpolation=TF.InterpolationMode.NEAREST,
            antialias=False,
        ).clamp(0, 1)

    def __repr__(self):
        return f'PixelatePerturb(ratio={self.ratio:.2f})'


class ScratchPerturb:
    """Draw deterministic bright/dark line scratches to mimic surface artifacts."""

    PARAMS = {
        'mild': (3, 1),
        'moderate': (6, 2),
        'severe': (12, 3),
    }

    def __init__(self, severity: str):
        if severity not in self.PARAMS:
            raise ValueError(f'Unknown scratch severity: {severity!r}')
        self.severity = severity
        self.n_lines, self.thickness = self.PARAMS[severity]

    @staticmethod
    def _seed_from_tensor(x: torch.Tensor) -> int:
        # Stable enough for cached perturbations while avoiding global RNG state.
        v = float(x[:, ::16, ::16].sum().item())
        return int(abs(v) * 1_000_003) % (2**32 - 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        img = (x.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        h, w = img.shape[:2]
        rng = np.random.RandomState(self._seed_from_tensor(x))
        out = img.copy()
        for _ in range(self.n_lines):
            x1 = int(rng.randint(0, w))
            y1 = int(rng.randint(0, h))
            length = int(rng.randint(max(8, w // 8), max(9, w // 2)))
            angle = float(rng.uniform(-np.pi, np.pi))
            x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
            y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
            gray = int(rng.choice([35, 220]))
            color = (gray, gray, gray)
            cv2.line(out, (x1, y1), (x2, y2), color, self.thickness, lineType=cv2.LINE_AA)
        return torch.from_numpy(out).permute(2, 0, 1).float().div(255.0).to(device).clamp(0, 1)

    def __repr__(self):
        return f'ScratchPerturb({self.severity})'


class DefocusBlurPerturb:
    """
    Simulate optical defocus using a disk-shaped PSF (pillbox kernel).

    More physically realistic than Gaussian blur for out-of-focus imaging.
    """

    def __init__(self, radius: int):
        self.radius = radius
        self.kernel = self._make_disk_kernel(radius)

    @staticmethod
    def _make_disk_kernel(radius: int) -> np.ndarray:
        size = 2 * radius + 1
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = x ** 2 + y ** 2 <= radius ** 2
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[mask] = 1.0
        kernel /= kernel.sum()
        return kernel

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        img_np = x.permute(1, 2, 0).numpy()  # (H, W, C)
        blurred = cv2.filter2D(img_np, -1, self.kernel)
        return torch.from_numpy(blurred).permute(2, 0, 1).clamp(0, 1)

    def __repr__(self):
        return f'DefocusBlurPerturb(r={self.radius})'


class JPEGCompression:
    """
    Simulate JPEG compression artifacts at a given quality level.

    Lower quality = more artifacts. Typical range: 10 (heavy) to 95 (light).
    """

    def __init__(self, quality: int):
        self.quality = quality

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pil = TF.to_pil_image(x)
        buffer = io.BytesIO()
        pil.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert('RGB')
        return TF.to_tensor(compressed)

    def __repr__(self):
        return f'JPEGCompression(q={self.quality})'


class CompoundPerturb:
    """
    Apply multiple perturbations sequentially.

    Simulates realistic deployment conditions where multiple degradations
    co-occur (e.g., out-of-focus + poor lighting + JPEG compression).
    """

    def __init__(self, perturbations: list):
        self.perturbations = perturbations

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for p in self.perturbations:
            x = p(x)
        return x

    def __repr__(self):
        return f'CompoundPerturb({self.perturbations})'


# ── Factory ─────────────────────────────────────────────────────────────────

def make_perturbation(pcfg: dict, value, img_size: int = IMG_SIZE):
    """Construct a perturbation transform from a PERTURB_CONFIGS entry."""
    t = pcfg['type']
    if t == 'channel':
        return ChannelShift(pcfg['channel'], value)
    elif t == 'gaussian_noise':
        return GaussianNoisePerturb(value)
    elif t == 'shot_noise':
        return ShotNoisePerturb(value)
    elif t == 'impulse_noise':
        return ImpulseNoisePerturb(value)
    elif t == 'motion_blur':
        return MotionBlurPerturb(value)
    elif t == 'zoom_blur':
        return ZoomBlurPerturb(value, img_size)
    elif t == 'contrast':
        return ContrastPerturb(value)
    elif t == 'pixelate':
        return PixelatePerturb(value)
    elif t == 'scratch':
        return ScratchPerturb(value)
    elif t == 'resize':
        return ResizeZoom(value, img_size)
    elif t == 'blur':
        return GaussianBlurPerturb(value)
    elif t == 'rotation':
        return RotationPerturb(value)
    elif t == 'brightness':
        return BrightnessContrast(value)
    elif t == 'defocus':
        return DefocusBlurPerturb(value)
    elif t == 'jpeg':
        return JPEGCompression(value)
    elif t == 'compound':
        prefix = pcfg.get('preset_prefix')
        preset_key = f'{prefix}_{value}' if prefix else value
        preset = COMPOUND_PRESETS[preset_key]
        perturbs = []
        for sub_type, sub_val in preset:
            if sub_type == 'blur':
                perturbs.append(GaussianBlurPerturb(sub_val))
            elif sub_type == 'defocus':
                perturbs.append(DefocusBlurPerturb(sub_val))
            elif sub_type == 'brightness':
                perturbs.append(BrightnessContrast(sub_val))
            elif sub_type == 'jpeg':
                perturbs.append(JPEGCompression(sub_val))
            elif sub_type == 'motion_blur':
                perturbs.append(MotionBlurPerturb(sub_val))
            elif sub_type == 'zoom_blur':
                perturbs.append(ZoomBlurPerturb(sub_val, img_size))
            elif sub_type == 'contrast':
                perturbs.append(ContrastPerturb(sub_val))
            elif sub_type == 'pixelate':
                perturbs.append(PixelatePerturb(sub_val))
            elif sub_type == 'gaussian_noise':
                perturbs.append(GaussianNoisePerturb(sub_val))
            elif sub_type == 'shot_noise':
                perturbs.append(ShotNoisePerturb(sub_val))
            elif sub_type == 'impulse_noise':
                perturbs.append(ImpulseNoisePerturb(sub_val))
            elif sub_type == 'red_channel_shift':
                perturbs.append(ChannelShift(0, sub_val))
            elif sub_type == 'green_channel_shift':
                perturbs.append(ChannelShift(1, sub_val))
            elif sub_type == 'blue_channel_shift':
                perturbs.append(ChannelShift(2, sub_val))
            else:
                raise ValueError(f'Unknown compound sub-perturbation: {sub_type!r}')
        return CompoundPerturb(perturbs)
    else:
        raise ValueError(f'Unknown perturbation type: {t!r}')


def cache_tag_for(pert_name: str, value) -> str:
    """
    Generate cache filename tag for a (perturbation, value) pair.

    Examples:
        cache_tag_for('gaussian_blur', 12)  -> 'blur_12'
        cache_tag_for('defocus_blur', 5)    -> 'defocus_5'
        cache_tag_for('jpeg', 30)           -> 'jpeg_30'
        cache_tag_for('compound', 'mild')   -> 'compound_mild'
        cache_tag_for('resize', 2.0)        -> 'resize_2_0'
    """
    if pert_name == 'gaussian_blur':
        return f'blur_{int(value)}'
    elif pert_name == 'defocus_blur':
        return f'defocus_{int(value)}'
    elif pert_name == 'jpeg':
        return f'jpeg_{int(value)}'
    elif pert_name == 'compound':
        return f'compound_{value}'
    elif pert_name in ('compound_optical', 'compound_digital', 'compound_field'):
        return f'{pert_name}_{value}'
    elif pert_name == 'color_shift':
        return f'color_shift_{int(value)}'
    elif pert_name == 'red_channel_shift':
        return f'color_shift_{int(value)}'
    elif pert_name in ('green_channel_shift', 'blue_channel_shift'):
        return f'{pert_name}_{int(value)}'
    elif pert_name in ('gaussian_noise', 'impulse_noise', 'zoom_blur', 'contrast', 'pixelate'):
        return f'{pert_name}_{str(value).replace(".", "_")}'
    elif pert_name in ('shot_noise', 'motion_blur'):
        return f'{pert_name}_{int(value)}'
    elif pert_name == 'scratch':
        return f'scratch_{value}'
    elif pert_name == 'illumination':
        return f'brightness_{str(value).replace(".", "_")}'
    elif pert_name == 'resize':
        return f'resize_{str(value).replace(".", "_")}'
    elif pert_name == 'rotation':
        return f'rotation_{int(value)}'
    return f'{pert_name}_{value}'


def v2_cache_tag_for(pert_name: str, value) -> str:
    """
    Map v4 perturbation names to v2 cache tags for reusing existing caches.

    Returns None if no v2 equivalent exists.
    """
    mapping = {
        'gaussian_blur': lambda v: f'blur_{int(v)}',
        'resize': lambda v: f'resize_{str(v).replace(".", "_")}',
        'rotation': lambda v: f'rotation_{int(v)}',
        'illumination': lambda v: f'brightness_{str(v).replace(".", "_")}',
        'color_shift': lambda v: f'red_shift_{int(v)}',
        'red_channel_shift': lambda v: f'red_shift_{int(v)}',
    }
    if pert_name in mapping:
        return mapping[pert_name](value)
    return None
