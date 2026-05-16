"""
Wood Spatial — Dataset Module
===============================
WoodDataset: loads images from <root>/<species>/<image files>.
Adapted from github_package_v2/wood_benchmark/core/dataset.py.
"""
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from wood_spatial.config import (
    IMAGENET_MEAN, IMAGENET_STD, IMG_EXTS,
    RANDOM_SEED, BATCH_SIZE, NUM_WORKERS,
)

logger = logging.getLogger(__name__)


def get_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization to [0, 1] range."""
    mean = torch.tensor(IMAGENET_MEAN, device=t.device)
    std = torch.tensor(IMAGENET_STD, device=t.device)
    if t.dim() == 3:
        mean, std = mean.view(3, 1, 1), std.view(3, 1, 1)
    else:
        mean, std = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)
    return (t * std + mean).clamp(0, 1)


def renormalize(t: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a [0, 1] tensor."""
    mean = torch.tensor(IMAGENET_MEAN, device=t.device)
    std = torch.tensor(IMAGENET_STD, device=t.device)
    if t.dim() == 3:
        mean, std = mean.view(3, 1, 1), std.view(3, 1, 1)
    else:
        mean, std = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)
    return (t - mean) / std


class WoodDataset(Dataset):
    """
    Load images from folder structure: <root>/<species_label>/<image files>.

    Perturbation is applied AFTER denormalization and BEFORE renormalization.
    """

    def __init__(
        self,
        root,
        img_size: int = 224,
        split: str = 'all',
        val_split: float = 0.2,
        random_seed: int = RANDOM_SEED,
        min_samples: int = 2,
        max_images: int = None,
        perturbation=None,
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.perturbation = perturbation
        self.transform = get_transform(img_size)
        self.samples, self.class_to_idx = self._load_folder(min_samples)
        if not self.samples:
            raise ValueError(f'No images found in {root}')
        if split != 'all':
            self.samples = self._split(split, val_split, random_seed)
        if max_images and len(self.samples) > max_images:
            rng = np.random.default_rng(random_seed)
            idx = rng.choice(len(self.samples), max_images, replace=False)
            self.samples = [self.samples[i] for i in sorted(idx)]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def _load_folder(self, min_s: int):
        c2i, samples = {}, []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or d.name.startswith('.'):
                continue
            imgs = [p for p in sorted(d.iterdir()) if p.suffix in IMG_EXTS]
            if len(imgs) < min_s:
                continue
            lbl = len(c2i)
            c2i[d.name] = lbl
            samples.extend([(p, lbl) for p in imgs])
        return samples, c2i

    def _split(self, split: str, val_split: float, seed: int):
        rng = np.random.default_rng(seed)
        by_cls = defaultdict(list)
        for s in self.samples:
            by_cls[s[1]].append(s)
        train, val = [], []
        for cls_s in by_cls.values():
            rng.shuffle(cls_s)
            n = max(1, int(len(cls_s) * val_split))
            val.extend(cls_s[:n])
            train.extend(cls_s[n:])
        return train if split == 'train' else val

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        tensor = self.transform(img)
        if self.perturbation is not None:
            raw = denormalize(tensor)
            raw = self.perturbation(raw)
            tensor = renormalize(raw)
        return tensor, label, str(path)

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def labels(self):
        return np.array([s[1] for s in self.samples])


def make_loader(
    ds: WoodDataset,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """Create a non-shuffled DataLoader for feature extraction."""
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
