#!/usr/bin/env python3
"""Run post-extraction stages from a given stage onward.

This is a thin convenience wrapper around scripts/run_full_colab.py. It expands
the canonical POST_EXTRACT_WAVES list, keeps only stages from --from-stage, and
invokes the full runner with --only ... --parallel-post.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_full_colab import POST_EXTRACT_WAVES  # noqa: E402


def _flatten_waves() -> list[str]:
    stages = []
    for wave in POST_EXTRACT_WAVES:
        stages.extend(wave)
    return stages


def main() -> int:
    parser = argparse.ArgumentParser(description="Continue post-extract run from a selected stage.")
    parser.add_argument("--config", default="configs/full_colab_l4.json")
    parser.add_argument("--from-stage", required=True, help="First stage to run, e.g. run_ablations.")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--stop-before",
        default=None,
        help="Optional stage name where the resumed run should stop before running it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected stages and command without executing.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not pass --resume to run_full_colab.py. By default completed later stages are skipped.",
    )
    args = parser.parse_args()

    stages = _flatten_waves()
    if args.from_stage not in stages:
        raise SystemExit(f"Unknown --from-stage {args.from_stage!r}. Known stages: {' '.join(stages)}")
    start = stages.index(args.from_stage)
    end = len(stages)
    if args.stop_before is not None:
        if args.stop_before not in stages:
            raise SystemExit(f"Unknown --stop-before {args.stop_before!r}. Known stages: {' '.join(stages)}")
        end = stages.index(args.stop_before)
        if end <= start:
            raise SystemExit("--stop-before must appear after --from-stage in POST_EXTRACT_WAVES.")

    selected = stages[start:end]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_full_colab.py"),
        "--config",
        args.config,
        "--only",
        *selected,
        "--parallel-post",
        "--jobs",
        str(args.jobs),
    ]
    if not args.no_resume:
        cmd.append("--resume")

    print("Selected stages:")
    print("  " + " ".join(selected))
    print("\nCommand:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
