#!/usr/bin/env python3
"""Export the exact timm pretrained configurations used by the paper."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import timm
import torch
import torchvision

from wood_spatial.config import BACKBONE_CONFIGS, BB_ORDER
from wood_spatial.result_io import csv_dir


def _serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return list(value)
    return str(value)


def build_manifest() -> pd.DataFrame:
    rows = []
    for backbone in BB_ORDER:
        config = BACKBONE_CONFIGS[backbone]
        model_id = config["model_id"]
        pretrained_cfg = timm.get_pretrained_cfg(model_id)
        cfg = (
            pretrained_cfg.to_dict()
            if hasattr(pretrained_cfg, "to_dict")
            else dict(pretrained_cfg)
        )
        rows.append({
            "backbone": backbone,
            "timm_model_id": model_id,
            "pretrained_tag": cfg.get("tag", ""),
            "architecture": cfg.get("architecture", model_id),
            "hf_hub_id": cfg.get("hf_hub_id", ""),
            "checkpoint_url": cfg.get("url", ""),
            "input_size": json.dumps(
                _serializable(cfg.get("input_size", ""))
            ),
            "mean": json.dumps(_serializable(cfg.get("mean", ""))),
            "std": json.dumps(_serializable(cfg.get("std", ""))),
            "paper_img_size": config["img_size"],
            "timm_version": timm.__version__,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "python_version": platform.python_version(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=csv_dir() / "backbone_pretrained_manifest.csv",
    )
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(manifest.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
