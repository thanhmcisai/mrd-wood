"""Strict result I/O and reproducibility helpers."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from wood_spatial.config import BASE, V4_CSV, V4_FIGURES


def csv_dir() -> Path:
    V4_CSV.mkdir(parents=True, exist_ok=True)
    return V4_CSV


def figure_dir() -> Path:
    V4_FIGURES.mkdir(parents=True, exist_ok=True)
    return V4_FIGURES


def require_csv(name: str) -> Path:
    path = csv_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"Required canonical result is missing: {path}. "
            "Set WOOD_RESULTS_DIR to the intended run before importing wood_spatial."
        )
    return path


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def write_provenance(
    experiment: str,
    outputs: Iterable[Path],
    *,
    protocol: str,
    parameters: dict[str, object],
    inputs: Iterable[Path] = (),
) -> Path:
    output_paths = [Path(path).resolve() for path in outputs]
    input_paths = [Path(path).resolve() for path in inputs]
    payload = {
        "experiment": experiment,
        "protocol": protocol,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(V4_CSV.parent.resolve()),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "parameters": parameters,
        "inputs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in input_paths
            if path.exists()
        ],
        "outputs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in output_paths
            if path.exists()
        ],
        "environment": {
            "WOOD_BASE": os.environ.get("WOOD_BASE", ""),
            "WOOD_RESULTS_DIR": os.environ.get("WOOD_RESULTS_DIR", ""),
            "WOOD_DATASETS_DIR": os.environ.get("WOOD_DATASETS_DIR", ""),
        },
    }
    path = csv_dir() / f"{experiment}.provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
