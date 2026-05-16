#!/usr/bin/env python3
"""Check Wood Research feature/spatial cache integrity.

The script scans .npz cache files, verifies that they are readable, non-empty,
and contain the expected arrays. By default it only reports problems. Use
--delete-bad to remove corrupted caches so extraction can regenerate them.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np


DEFAULT_REQUIRED_KEYS = ("features", "labels", "paths")


def _format_size(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{n_bytes} B"


def check_npz(path: Path, required_keys: tuple[str, ...], deep: bool = False) -> tuple[str, bool, str, int]:
    if not path.exists():
        return str(path), False, "missing", -1
    try:
        size = path.stat().st_size
    except OSError as exc:
        return str(path), False, f"stat_failed: {exc}", -1
    if size == 0:
        return str(path), False, "empty_file", size

    try:
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in required_keys if key not in data.files]
            if missing:
                return str(path), False, f"missing_keys={','.join(missing)}", size
            if deep:
                # Materialize arrays. This is slower but catches more rare data
                # corruption than zip/header checks alone.
                features = data["features"]
                labels = data["labels"]
                paths = data["paths"]
            else:
                # Header-level shape checks avoid reading large feature arrays.
                features = data["features"]
                labels = data["labels"]
                paths = data["paths"]
            if len(features) == 0:
                return str(path), False, "empty_features", size
            if len(labels) != len(features):
                return str(path), False, f"label_count_mismatch features={len(features)} labels={len(labels)}", size
            if len(paths) != len(features):
                return str(path), False, f"path_count_mismatch features={len(features)} paths={len(paths)}", size
    except Exception as exc:
        return str(path), False, f"{type(exc).__name__}: {exc}", size

    return str(path), True, "ok", size


def check_zip(path: Path) -> tuple[str, bool, str, int]:
    """Fast zip-level check for .npz files without loading arrays."""
    if not path.exists():
        return str(path), False, "missing", -1
    try:
        size = path.stat().st_size
    except OSError as exc:
        return str(path), False, f"stat_failed: {exc}", -1
    if size == 0:
        return str(path), False, "empty_file", size
    try:
        with zipfile.ZipFile(path) as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                return str(path), False, f"bad_zip_member={bad_member}", size
            names = zf.namelist()
            if not names:
                return str(path), False, "empty_zip", size
    except Exception as exc:
        return str(path), False, f"{type(exc).__name__}: {exc}", size
    return str(path), True, "ok", size


def _worker(args):
    path_text, required_keys, mode = args
    path = Path(path_text)
    if mode == "zip":
        return check_zip(path)
    return check_npz(path, tuple(required_keys), deep=(mode == "deep"))


def iter_npz(paths: list[Path], recursive: bool):
    for root in paths:
        if root.is_file():
            if root.suffix == ".npz":
                yield root
            continue
        if not root.exists():
            print(f"WARN missing directory: {root}", file=sys.stderr)
            continue
        pattern = "**/*.npz" if recursive else "*.npz"
        yield from root.glob(pattern)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check .npz cache integrity.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Cache directories or .npz files. Defaults to results/feature_cache and results/spatial_cache when present.",
    )
    parser.add_argument(
        "--required-keys",
        default=",".join(DEFAULT_REQUIRED_KEYS),
        help="Comma-separated keys required in each npz file.",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories recursively.")
    parser.add_argument("--delete-bad", action="store_true", help="Delete unreadable/invalid cache files.")
    parser.add_argument("--show-ok", action="store_true", help="Print valid files too.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of bad files printed; 0 means no limit.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Parallel worker processes. Default: min(8, CPU count).",
    )
    parser.add_argument(
        "--mode",
        choices=["shape", "zip", "deep"],
        default="shape",
        help=(
            "shape: open npz and check required arrays/shapes without forcing full reads; "
            "zip: fastest zip CRC check only; "
            "deep: materialize arrays to catch deeper corruption."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N files; 0 disables progress.",
    )
    args = parser.parse_args()

    paths = args.paths
    if not paths:
        candidates = [
            Path("results/feature_cache"),
            Path("results/spatial_cache"),
            Path("results_v4/feature_cache"),
            Path("results_v4/spatial_cache"),
        ]
        paths = [p for p in candidates if p.exists()]
        if not paths:
            parser.error("No paths provided and no default cache directories found.")

    required_keys = tuple(key.strip() for key in args.required_keys.split(",") if key.strip())
    files = sorted(set(iter_npz(paths, args.recursive)))

    ok_count = 0
    bad = []
    t0 = time.time()
    tasks = [(str(path), required_keys, args.mode) for path in files]
    print(f"Scanning {len(files)} files with {args.jobs} workers (mode={args.mode})...", flush=True)

    if args.jobs <= 1:
        iterator = map(_worker, tasks)
        for i, (path_text, ok, reason, size) in enumerate(iterator, start=1):
            path = Path(path_text)
            if ok:
                ok_count += 1
                if args.show_ok:
                    print(f"OK  {_format_size(size):>10s}  {path}", flush=True)
            else:
                bad.append((path, reason, size))
            if args.progress_every and i % args.progress_every == 0:
                elapsed = time.time() - t0
                print(f"  checked {i}/{len(files)} | bad={len(bad)} | {elapsed:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(_worker, task) for task in tasks]
            for i, fut in enumerate(as_completed(futures), start=1):
                path_text, ok, reason, size = fut.result()
                path = Path(path_text)
                if ok:
                    ok_count += 1
                    if args.show_ok:
                        print(f"OK  {_format_size(size):>10s}  {path}", flush=True)
                else:
                    bad.append((path, reason, size))
                if args.progress_every and i % args.progress_every == 0:
                    elapsed = time.time() - t0
                    print(f"  checked {i}/{len(files)} | bad={len(bad)} | {elapsed:.1f}s", flush=True)

    print("\nCache integrity summary")
    print(f"  scanned: {len(files)}")
    print(f"  ok:      {ok_count}")
    print(f"  bad:     {len(bad)}")
    print(f"  elapsed: {time.time() - t0:.1f}s")

    shown = 0
    for path, reason, size in sorted(bad, key=lambda x: str(x[0])):
        if args.limit and shown >= args.limit:
            remaining = len(bad) - shown
            print(f"  ... {remaining} more bad files not shown")
            break
        size_text = _format_size(size) if size >= 0 else "missing"
        print(f"BAD {size_text:>10s}  {path}  | {reason}")
        shown += 1

    if args.delete_bad and bad:
        print("\nDeleting bad cache files...")
        deleted = 0
        for path, _reason, _size in bad:
            try:
                path.unlink()
                deleted += 1
                print(f"  deleted {path}")
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"  failed {path}: {exc}", file=sys.stderr)
        print(f"Deleted {deleted} bad files.")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
