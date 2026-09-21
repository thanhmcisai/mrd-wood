#!/usr/bin/env python3
"""Audit and optionally relabel cached FSDM41 feature files.

FSDM41 was distributed with a class-name permutation in some copies. This tool
uses dataset_label_corrections.json to remap cached labels from the old folder
names to corrected species names without re-extracting feature vectors.

Default mode is read-only. Use --write to rewrite labels/paths in cache files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
import zipfile

import numpy as np
from numpy.lib import format as npy_format

from wood_spatial.core.label_corrections import CORRECTION_MARKER_KEY, FSDM41_MARKER_VALUE


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORRECTIONS = ROOT / "dataset_label_corrections.json"
LEGACY_RELABEL_MARKER_KEY = "fsdm41_relabel_applied"


def _default_cache_dirs() -> list[Path]:
    dirs: list[Path] = []
    env_results = os.environ.get("WOOD_RESULTS_DIR")
    if env_results:
        dirs.append(Path(env_results) / "feature_cache")
        dirs.append(Path(env_results) / "spatial_cache")
    for rel in (
        "results/feature_cache",
        "results/spatial_cache",
        "results_v4/feature_cache",
        "results_v4/spatial_cache",
        "results_v2/feature_cache",
    ):
        dirs.append(ROOT / rel)
    out: list[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        d = d.expanduser().resolve()
        if d.exists() and d not in seen:
            out.append(d)
            seen.add(d)
    return out


def _load_overrides(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    overrides = payload.get("FSDM41", {}).get("overrides")
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError(f"No FSDM41.overrides mapping found in {path}")
    return {str(k): str(v) for k, v in overrides.items()}


def _iter_cache_files(cache_dirs: list[Path], dataset: str) -> list[Path]:
    files: list[Path] = []
    for d in cache_dirs:
        if d.is_file() and d.suffix == ".npz":
            if dataset in d.name:
                files.append(d)
            continue
        if not d.exists():
            continue
        files.extend(sorted(d.glob(f"*_{dataset}_*.npz")))
        files.extend(sorted(d.glob(f"*_{dataset}_all_*.npz")))
    return sorted(set(files))


def _path_parent_name(path_text: Any) -> str:
    return Path(str(path_text)).parent.name


def _replace_parent(path_text: Any, corrected_parent: str, dataset_root: Path | None) -> str:
    p = Path(str(path_text))
    if dataset_root is not None:
        return str(dataset_root / corrected_parent / p.name)
    parts = list(p.parts)
    if len(parts) >= 2:
        parts[-2] = corrected_parent
        return str(Path(*parts))
    return str(p)


def _label_to_names(labels: np.ndarray, paths: np.ndarray) -> tuple[dict[int, str], list[str]]:
    label_to_names: dict[int, set[str]] = {}
    for lab, path in zip(labels, paths):
        try:
            key = int(lab)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Labels must be numeric for cache remapping; got {lab!r}") from exc
        label_to_names.setdefault(key, set()).add(_path_parent_name(path))
    ambiguous = [
        f"{lab}:{sorted(names)}"
        for lab, names in sorted(label_to_names.items())
        if len(names) != 1
    ]
    resolved = {lab: next(iter(names)) for lab, names in label_to_names.items() if len(names) == 1}
    return resolved, ambiguous


def _remap_arrays(
    labels: np.ndarray,
    paths: np.ndarray,
    overrides: dict[str, str],
    dataset_root: Path | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    label_to_name, ambiguous = _label_to_names(labels, paths)
    if ambiguous:
        raise ValueError(
            "A numeric label maps to multiple folder names; cannot safely remap: "
            + "; ".join(ambiguous[:8])
        )

    old_names_by_row = np.asarray([label_to_name[int(lab)] for lab in labels], dtype=object)
    corrected_by_row = np.asarray([overrides.get(name, name) for name in old_names_by_row], dtype=object)
    corrected_classes = sorted(set(corrected_by_row.tolist()))
    class_to_idx = {name: idx for idx, name in enumerate(corrected_classes)}
    new_labels = np.asarray([class_to_idx[name] for name in corrected_by_row], dtype=labels.dtype)
    new_path_strings = [
        _replace_parent(path, corrected, dataset_root) for path, corrected in zip(paths, corrected_by_row)
    ]
    max_path_len = max((len(p) for p in new_path_strings), default=1)
    new_paths = np.asarray(new_path_strings, dtype=f"<U{max_path_len}")

    changed_name_rows = int(np.sum(old_names_by_row != corrected_by_row))
    changed_label_rows = int(np.sum(labels != new_labels))
    changed_path_rows = int(np.sum(paths.astype(str) != new_paths.astype(str)))
    old_classes = sorted(set(old_names_by_row.tolist()))
    remapped_classes = sorted(name for name in old_classes if name in overrides)
    already_correct_names = sorted(set(old_classes).intersection(set(overrides.values())))

    info = {
        "n_rows": int(len(labels)),
        "n_old_classes": int(len(old_classes)),
        "n_corrected_classes": int(len(corrected_classes)),
        "n_remapped_classes": int(len(remapped_classes)),
        "n_rows_species_name_changed": changed_name_rows,
        "n_rows_numeric_label_changed": changed_label_rows,
        "n_rows_path_changed": changed_path_rows,
        "old_classes": old_classes,
        "corrected_classes": corrected_classes,
        "remapped_classes": remapped_classes,
        "already_correct_name_overlap": already_correct_names,
    }
    return new_labels, new_paths, info


def _path_exists_sample(paths: np.ndarray, max_check: int = 200) -> tuple[int, int]:
    if len(paths) == 0:
        return 0, 0
    step = max(1, len(paths) // max_check)
    sample = paths[::step][:max_check]
    ok = sum(1 for p in sample if Path(str(p)).exists())
    return ok, len(sample)


def _npz_member_shape(path: Path, key: str) -> tuple[int, ...]:
    """Read an .npz member header without loading the array payload."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{key}.npy") as handle:
            version = npy_format.read_magic(handle)
            shape, _fortran_order, _dtype = npy_format._read_array_header(handle, version)
            return tuple(int(x) for x in shape)


def _write_npz_atomic(path: Path, arrays: dict[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _backup(path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / (path.name + ".before_fsdm41_relabel")
    if backup.exists():
        return backup
    try:
        os.link(path, backup)
    except OSError:
        shutil.copy2(path, backup)
    return backup


def process_file(
    path: Path,
    overrides: dict[str, str],
    dataset_root: Path | None,
    write: bool,
    backup_dir: Path | None,
    force: bool = False,
    check_paths: bool = False,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.files)
        missing = [k for k in ("features", "labels", "paths") if k not in keys]
        if missing:
            return {"path": str(path), "ok": False, "reason": f"missing_keys={missing}"}
        if (CORRECTION_MARKER_KEY in keys or LEGACY_RELABEL_MARKER_KEY in keys) and not force:
            return {
                "path": str(path),
                "ok": True,
                "action": "already_rewritten",
                "keys": keys,
                "feature_shape": _npz_member_shape(path, "features"),
                "label_dtype": str(data["labels"].dtype),
                "path_dtype": str(data["paths"].dtype),
                "sample_paths_exist_before": "skipped",
                "sample_paths_exist_after": "skipped",
                "n_old_classes": 0,
                "n_corrected_classes": 0,
                "n_remapped_classes": 0,
                "n_rows_species_name_changed": 0,
                "n_rows_numeric_label_changed": 0,
                "n_rows_path_changed": 0,
                "already_correct_name_overlap": [],
            }
        labels = np.asarray(data["labels"])
        paths = np.asarray(data["paths"])

    feature_shape = _npz_member_shape(path, "features")
    if not feature_shape or feature_shape[0] != len(labels) or feature_shape[0] != len(paths):
        return {
            "path": str(path),
            "ok": False,
            "reason": f"length_mismatch features={feature_shape} labels={len(labels)} paths={len(paths)}",
        }

    before_exists: tuple[int, int] | None = _path_exists_sample(paths) if check_paths else None
    try:
        new_labels, new_paths, info = _remap_arrays(labels, paths, overrides, dataset_root)
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "ok": False, "reason": str(exc)}
    after_exists: tuple[int, int] | None = _path_exists_sample(new_paths) if check_paths else None

    action = "audit"
    if write:
        with np.load(path, allow_pickle=True) as data:
            arrays = {k: data[k] for k in data.files}
        if backup_dir is not None:
            _backup(path, backup_dir)
        arrays["labels"] = new_labels
        arrays["paths"] = new_paths
        arrays[CORRECTION_MARKER_KEY] = np.asarray(FSDM41_MARKER_VALUE)
        _write_npz_atomic(path, arrays)
        action = "rewritten"

    return {
        "path": str(path),
        "ok": True,
        "action": action,
        "keys": keys,
        "feature_shape": feature_shape,
        "label_dtype": str(labels.dtype),
        "path_dtype": str(paths.dtype),
        "sample_paths_exist_before": (
            f"{before_exists[0]}/{before_exists[1]}" if before_exists is not None else "not_checked"
        ),
        "sample_paths_exist_after": (
            f"{after_exists[0]}/{after_exists[1]}" if after_exists is not None else "not_checked"
        ),
        **info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and optionally relabel FSDM41 caches without feature extraction."
    )
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--dataset", default="FSDM41")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        action="append",
        help="Cache directory or specific .npz file. May be repeated. Defaults to local/WOOD_RESULTS_DIR caches.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Corrected FSDM41 folder root. If provided, cached paths are rewritten to this root.",
    )
    parser.add_argument("--write", action="store_true", help="Rewrite labels/paths in cache files.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess files even if the FSDM41 relabel marker is already present.",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N matched cache files.")
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Sample filesystem path existence before/after remapping. Slow on Google Drive; off by default.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="Directory for hardlink/copy backups before --write. Strongly recommended for in-place writes.",
    )
    parser.add_argument("--json-report", type=Path, help="Optional path to save the full audit report JSON.")
    args = parser.parse_args()

    overrides = _load_overrides(args.corrections)
    cache_dirs = [p.expanduser().resolve() for p in args.cache_dir] if args.cache_dir else _default_cache_dirs()
    files = _iter_cache_files(cache_dirs, args.dataset)
    if args.limit is not None:
        files = files[: args.limit]

    print("=== FSDM41 cache relabel audit ===")
    print(f"Corrections: {args.corrections}")
    print(f"Overrides:   {len(overrides)} class-name mappings")
    print("Cache dirs:")
    for d in cache_dirs:
        print(f"  - {d}")
    if args.dataset_root:
        print(f"Dataset root for path rewrite: {args.dataset_root}")
    print(f"Mode: {'WRITE' if args.write else 'check-only'}")
    print(f"Cache files found: {len(files)}")

    if not files:
        print("\nNo FSDM41 cache files found. On Colab, pass e.g.:")
        print("  --cache-dir /content/drive/MyDrive/NCS/results_v5_k3_full/feature_cache")
        print("  --cache-dir /content/drive/MyDrive/NCS/results_v5_k3_full/spatial_cache")
        return 2

    reports = []
    for path in files:
        try:
            report = process_file(
                path,
                overrides,
                args.dataset_root,
                args.write,
                args.backup_dir,
                force=args.force,
                check_paths=args.check_paths,
            )
        except Exception as exc:  # noqa: BLE001
            report = {"path": str(path), "ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        reports.append(report)
        status = "OK" if report.get("ok") else "FAIL"
        print(f"\n[{status}] {path}")
        if not report.get("ok"):
            print(f"  reason: {report.get('reason')}")
            continue
        print(f"  action: {report['action']}")
        print(f"  shape:  {report['feature_shape']} labels={report['label_dtype']} paths={report['path_dtype']}")
        if report["action"] != "already_rewritten":
            print(
                "  classes: "
                f"old={report['n_old_classes']} corrected={report['n_corrected_classes']} "
                f"mapped={report['n_remapped_classes']}"
            )
            print(
                "  rows: "
                f"species_name_changed={report['n_rows_species_name_changed']} "
                f"numeric_label_changed={report['n_rows_numeric_label_changed']} "
                f"path_changed={report['n_rows_path_changed']}"
            )
            print(
                "  path existence sample: "
                f"before={report['sample_paths_exist_before']} after={report['sample_paths_exist_after']}"
            )
        if report["already_correct_name_overlap"]:
            names = ", ".join(report["already_correct_name_overlap"][:8])
            print(f"  note: cache already contains some corrected names: {names}")

    n_ok = sum(1 for r in reports if r.get("ok"))
    n_fail = len(reports) - n_ok
    n_changed = sum(1 for r in reports if r.get("ok") and r.get("n_rows_species_name_changed", 0) > 0)
    print("\n=== Summary ===")
    print(f"OK={n_ok} FAIL={n_fail} files_with_species_remap={n_changed}")
    if args.write:
        print("Caches were rewritten in place. Feature arrays were preserved; labels/paths were remapped.")
    else:
        print("No files were modified. Re-run with --write to apply remapping.")

    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"Report saved: {args.json_report}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
