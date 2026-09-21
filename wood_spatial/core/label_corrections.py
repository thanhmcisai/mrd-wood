"""Dataset label-correction utilities.

The feature caches are treated as immutable feature stores. Dataset-specific
label fixes are applied when caches are loaded, so large cache files do not need
to be rewritten in-place on Google Drive.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from wood_spatial.config import BASE


CORRECTION_MARKER_KEY = "wood_label_correction_applied"
FSDM41_MARKER_VALUE = "dataset_label_corrections:FSDM41:v1"


def _corrections_path() -> Path:
    return Path(os.environ.get("WOOD_LABEL_CORRECTIONS", BASE / "dataset_label_corrections.json"))


def _load_fsdm41_overrides() -> dict[str, str]:
    path = _corrections_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    overrides = payload.get("FSDM41", {}).get("overrides", {})
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError(f"No FSDM41.overrides mapping found in {path}")
    return {str(k): str(v) for k, v in overrides.items()}


def _parent_name(path_text: Any) -> str:
    return Path(str(path_text)).parent.name


def _label_to_name(labels: np.ndarray, paths: np.ndarray) -> dict[int, str]:
    label_to_names: dict[int, set[str]] = {}
    for label, path in zip(labels, paths):
        label_to_names.setdefault(int(label), set()).add(_parent_name(path))

    ambiguous = {
        label: sorted(names)
        for label, names in label_to_names.items()
        if len(names) != 1
    }
    if ambiguous:
        examples = "; ".join(f"{label}:{names}" for label, names in list(ambiguous.items())[:5])
        raise ValueError(f"Cannot apply FSDM41 correction: label maps to multiple folders ({examples})")
    return {label: next(iter(names)) for label, names in label_to_names.items()}


def _replace_parent(path_text: Any, new_parent: str) -> str:
    path = Path(str(path_text))
    parts = list(path.parts)
    if len(parts) >= 2:
        parts[-2] = new_parent
        return str(Path(*parts))
    return str(path)


def apply_cache_label_correction(
    dataset_name: str,
    labels: np.ndarray,
    paths: np.ndarray,
    cache_keys: set[str] | None = None,
    *,
    rewrite_paths: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return corrected labels/paths for known dataset label issues.

    Corrections are enabled by default and can be disabled globally with
    ``WOOD_DISABLE_LABEL_CORRECTIONS=1``. Caches carrying
    ``wood_label_correction_applied`` are considered already normalized.
    """
    if os.environ.get("WOOD_DISABLE_LABEL_CORRECTIONS", "").lower() in {"1", "true", "yes"}:
        return labels, paths
    if dataset_name != "FSDM41":
        return labels, paths
    if cache_keys and CORRECTION_MARKER_KEY in cache_keys:
        return labels, paths

    labels = np.asarray(labels)
    paths = np.asarray(paths)
    overrides = _load_fsdm41_overrides()

    label_to_name = _label_to_name(labels, paths)
    old_by_row = np.asarray([label_to_name[int(label)] for label in labels], dtype=object)
    corrected_by_row = np.asarray([overrides.get(name, name) for name in old_by_row], dtype=object)

    corrected_classes = sorted(set(corrected_by_row.tolist()))
    class_to_idx = {name: idx for idx, name in enumerate(corrected_classes)}
    corrected_labels = np.asarray([class_to_idx[name] for name in corrected_by_row], dtype=labels.dtype)

    if not rewrite_paths:
        return corrected_labels, paths

    corrected_path_strings = [
        _replace_parent(path, corrected) for path, corrected in zip(paths, corrected_by_row)
    ]
    max_len = max((len(p) for p in corrected_path_strings), default=1)
    corrected_paths = np.asarray(corrected_path_strings, dtype=f"<U{max_len}")
    return corrected_labels, corrected_paths


def correction_marker_for_dataset(dataset_name: str) -> tuple[str, np.ndarray] | None:
    if dataset_name == "FSDM41":
        return CORRECTION_MARKER_KEY, np.asarray(FSDM41_MARKER_VALUE)
    return None
