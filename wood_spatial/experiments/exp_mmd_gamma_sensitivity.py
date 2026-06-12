#!/usr/bin/env python3
"""RBF-MMD bandwidth sensitivity for the reference-bank monitor.

The monitor experiments use an RBF-MMD score with a median-heuristic bandwidth
computed for each reference/target comparison. That is a reasonable detector,
but it also means raw MMD values are not automatically comparable across shift
conditions. This experiment recomputes each original clean or real-shift
source--target comparison under shared bandwidth policies, then aggregates the
record-level scores by condition. Keeping the original comparison unit avoids
confounding MMD with mixtures of unrelated datasets or transfer directions.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import queue
import traceback
from multiprocessing import get_context
from pathlib import Path

# Avoid nested parallelism: process-level jobs already parallelize backbones.
_BLAS_THREADS = os.environ.get("WOOD_BLAS_THREADS", "1")
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = _BLAS_THREADS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wood_spatial.config import BB_LABEL, BB_ORDER
from wood_spatial.result_io import csv_dir, figure_dir, require_csv, write_provenance
from wood_spatial.experiments.exp_monitor_on_real_shift import _cap, _load_norm_cache
from wood_spatial.experiments.exp_tierc_cross_source_shift import (
    _features_by_species,
    _resolve_species_csv,
    _shared_species,
    _species_table,
)


CONDITION_ORDER = [
    "clean_TierA",
    "TierD_xmag_x10x20",
    "TierD_xmag_x10x50",
    "TierD_xmag_x20x50",
    "TierC_DTSR14_WOODAUTH",
    "TierC_BFS46_FSDM41",
]

XMAG_CONDITIONS = {
    "x10<->x20": "TierD_xmag_x10x20",
    "x10<->x50": "TierD_xmag_x10x50",
    "x20<->x50": "TierD_xmag_x20x50",
}

EXPECTED_MONITOR_COUNTS = {
    "TierA_clean": 21,
    "TierD_xmag": 42,
    "TierC_DTSR14_WOODAUTH": 14,
    "TierC_BFS46_FSDM41": 14,
}

_THREADPOOL_GUARD = None


def _stable_seed(*parts: object) -> int:
    text = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(text, digest_size=4).digest(), "little")


def _limit_worker_threads() -> None:
    """Limit NumPy/BLAS pools inside each process worker."""
    global _THREADPOOL_GUARD
    try:
        from threadpoolctl import threadpool_limits
        _THREADPOOL_GUARD = threadpool_limits(limits=int(_BLAS_THREADS))
    except Exception:
        _THREADPOOL_GUARD = None


def _io_dirs() -> tuple[Path, Path]:
    return csv_dir(), figure_dir()


def _find_csv(name: str) -> Path:
    return require_csv(name)


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(
        0.0,
        np.sum(a * a, axis=1, keepdims=True) + np.sum(b * b, axis=1)[None, :] - 2.0 * (a @ b.T),
    )


def _median_gamma_global(refs: list[np.ndarray], cap: int, seed: int) -> float:
    parts = []
    per = max(32, cap // max(1, len(refs)))
    for i, x in enumerate(refs):
        if len(x):
            parts.append(_cap(x, per, seed + i))
    if not parts:
        return 1.0
    z = _cap(np.vstack(parts), cap, seed + 999)
    d2 = _sqdist(z, z)
    vals = d2[np.triu_indices_from(d2, k=1)]
    vals = vals[vals > 1e-12]
    med = float(np.median(vals)) if len(vals) else 1.0
    return 1.0 / max(2.0 * med, 1e-12)


def _record_mmds(
    ref: np.ndarray,
    target: np.ndarray,
    shared_gammas: dict[str, float],
    cap: int,
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Compute all bandwidth policies from one sampled pair and distance pass."""
    x = _cap(ref, cap, seed)
    y = _cap(target, cap, seed + 1)
    dxx = _sqdist(x, x)
    dyy = _sqdist(y, y)
    dxy = _sqdist(x, y)

    # Match the deployed monitor: estimate the pair-specific median from at
    # most ``cap`` pooled samples, while evaluating MMD on up to ``cap`` samples
    # from each side.
    gamma_sample = _cap(np.vstack([x, y]), cap, 42)
    gamma_distances = _sqdist(gamma_sample, gamma_sample)
    pair_distances = gamma_distances[np.triu_indices_from(gamma_distances, k=1)]
    pair_distances = pair_distances[pair_distances > 1e-12]
    median = float(np.median(pair_distances)) if len(pair_distances) else 1.0
    pair_gamma = 1.0 / max(2.0 * median, 1e-12)

    gammas = {"per_pair_median": pair_gamma, **shared_gammas}
    scores = {}
    for policy, gamma in gammas.items():
        score = (
            np.exp(-gamma * dxx).mean()
            + np.exp(-gamma * dyy).mean()
            - 2.0 * np.exp(-gamma * dxy).mean()
        )
        scores[policy] = max(float(score), 0.0)
    return pair_gamma, scores


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).correlation)
    except Exception:
        xr = pd.Series(x).rank(method="average").to_numpy()
        yr = pd.Series(y).rank(method="average").to_numpy()
        return float(np.corrcoef(xr, yr)[0, 1])


def _load_record_features(row: pd.Series, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    bb = str(row["backbone"])
    condition = str(row["condition"])
    src = str(row["reference_dataset"])
    dst = str(row["target_dataset"])
    if condition == "TierA_clean":
        clean = _load_norm_cache(bb, src, "original")
        mid = max(1, len(clean) // 2)
        return clean[:mid], clean[mid:]
    if condition == "TierD_xmag":
        return (
            _load_norm_cache(bb, src, "original"),
            _load_norm_cache(bb, dst, "original"),
        )
    if condition.startswith("TierC_"):
        species = _shared_species(table, src, dst)
        src_feats = _features_by_species(src, bb, table)
        dst_feats = _features_by_species(dst, bb, table)
        present = [s for s in species if s in src_feats and s in dst_feats]
        if not present:
            raise RuntimeError(f"No shared mapped species for {bb}/{src}->{dst}")
        return (
            np.vstack([src_feats[s] for s in present]),
            np.vstack([dst_feats[s] for s in present]),
        )
    raise ValueError(f"Unsupported condition for gamma sensitivity: {condition}")


def _analysis_condition(row: pd.Series, tierd_key: pd.DataFrame) -> tuple[str, float]:
    condition = str(row["condition"])
    bb = str(row["backbone"])
    src = str(row["reference_dataset"])
    dst = str(row["target_dataset"])
    if condition == "TierA_clean":
        return "clean_TierA", 0.0
    if condition == "TierD_xmag":
        item = tierd_key.loc[(bb, src, dst)]
        pair = str(item["mag_pair"])
        return XMAG_CONDITIONS[pair], float(item["accuracy_drop"])
    if condition.startswith("TierC_"):
        return condition, np.nan
    raise ValueError(condition)


def _run_backbone(task: tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]) -> tuple[str, list[dict]]:
    bb, table, monitor_real, tierd, tierc, cap = task
    _limit_worker_threads()
    pid = os.getpid()
    print(f"[start] pid={pid} backbone={bb}", flush=True)
    rows = []
    records = monitor_real[monitor_real["backbone"] == bb].copy()
    if records.empty:
        return bb, rows

    tierd_key = tierd.set_index(["backbone", "source_mag", "target_mag"])
    tierc_key = tierc.set_index(["direction", "backbone"])
    loaded = []
    for _, monitor_row in records.iterrows():
        ref, target = _load_record_features(monitor_row, table)
        condition, failure = _analysis_condition(monitor_row, tierd_key)
        if str(monitor_row["condition"]).startswith("TierC_"):
            direction = f"{monitor_row['reference_dataset']}->{monitor_row['target_dataset']}"
            item = tierc_key.loc[(direction, bb)]
            failure = 1.0 - float(item["cross_source_accuracy"])
        loaded.append((monitor_row, condition, failure, ref, target))

    # One bandwidth per backbone, estimated only from trusted clean Tier-A banks.
    global_refs = [ref for row, _cond, _failure, ref, _target in loaded
                   if str(row["condition"]) == "TierA_clean"]
    gamma0 = _median_gamma_global(global_refs, cap, _stable_seed(bb, "global"))
    shared_gammas = {
        "global_median": gamma0,
        "global_median_x0.5": gamma0 * 0.5,
        "global_median_x2.0": gamma0 * 2.0,
    }
    for monitor_row, cond, failure, ref, target in loaded:
        record_id = (
            bb,
            str(monitor_row["reference_dataset"]),
            str(monitor_row["target_dataset"]),
            str(monitor_row["target_tag"]),
        )
        seed = _stable_seed(*record_id)
        pair_gamma, scores = _record_mmds(ref, target, shared_gammas, cap, seed)
        policy_gammas = {"per_pair_median": pair_gamma, **shared_gammas}
        for policy, score in scores.items():
            rows.append({
                "backbone": bb,
                "backbone_label": BB_LABEL.get(bb, bb),
                "condition": cond,
                "reference_dataset": str(monitor_row["reference_dataset"]),
                "target_dataset": str(monitor_row["target_dataset"]),
                "target_tag": str(monitor_row["target_tag"]),
                "policy": policy,
                "gamma": float(policy_gammas[policy]),
                "mmd": score,
                "failure": float(failure),
                "n_reference": int(len(ref)),
                "n_target": int(len(target)),
            })
    print(f"[done] pid={pid} backbone={bb}: records={len(loaded)} gamma0={gamma0:.6g}", flush=True)
    return bb, rows


def _worker_entry(task, result_queue) -> None:
    """Run exactly one backbone, report its result, then terminate."""
    bb = task[0]
    try:
        _bb, rows = _run_backbone(task)
        result_queue.put(("ok", bb, rows, ""))
    except Exception:
        result_queue.put(("error", bb, [], traceback.format_exc()))


def _run_dynamic(tasks, jobs: int, checkpoint_dir: Path, cap: int) -> list[dict]:
    """Bounded scheduler whose worker count shrinks once no tasks remain."""
    try:
        ctx = get_context("fork")
    except ValueError:
        ctx = get_context()
    result_queue = ctx.Queue()
    pending = list(tasks)
    active = {}
    rows: list[dict] = []

    def launch() -> None:
        while pending and len(active) < jobs:
            task = pending.pop(0)
            bb = task[0]
            proc = ctx.Process(
                target=_worker_entry,
                args=(task, result_queue),
                name=f"mmd-gamma-{bb}",
            )
            proc.start()
            active[bb] = proc
            print(
                f"[spawn] pid={proc.pid} backbone={bb}; "
                f"active={len(active)} pending={len(pending)}",
                flush=True,
            )

    launch()
    while active:
        try:
            status, bb, bb_rows, error = result_queue.get(timeout=5)
        except queue.Empty:
            crashed = [
                (bb, proc) for bb, proc in active.items()
                if not proc.is_alive() and proc.exitcode not in (0, None)
            ]
            if crashed:
                names = ", ".join(f"{bb}(exit={proc.exitcode})" for bb, proc in crashed)
                raise RuntimeError(f"MMD gamma worker exited without a result: {names}")
            continue

        proc = active.pop(bb)
        proc.join()
        if status != "ok":
            raise RuntimeError(f"MMD gamma worker failed for {bb}:\n{error}")
        if not bb_rows:
            print(f"[skip] {bb}: no condition features", flush=True)
        else:
            pd.DataFrame(bb_rows).to_csv(
                checkpoint_dir / f"{bb}_cap{cap}.csv", index=False
            )
            rows.extend(bb_rows)
            print(
                f"[checkpoint] saved {bb}; active={len(active)} "
                f"pending={len(pending)}",
                flush=True,
            )
        launch()

    result_queue.close()
    result_queue.join_thread()
    return rows


def run_real(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    csv_dir, _fig_dir = _io_dirs()
    checkpoint_dir = csv_dir / "exp_mmd_gamma_sensitivity_checkpoints_v2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    table = _species_table(_resolve_species_csv(args.csv))
    monitor = pd.read_csv(_find_csv("exp_monitor_on_real_shift_scores.csv"))
    monitor_real = monitor[
        monitor["condition"].isin(EXPECTED_MONITOR_COUNTS)
    ].copy()
    observed_counts = monitor_real["condition"].value_counts().to_dict()
    if observed_counts != EXPECTED_MONITOR_COUNTS:
        raise RuntimeError(
            "Incomplete real-shift monitor records. "
            f"Expected {EXPECTED_MONITOR_COUNTS}, observed {observed_counts}."
        )
    tierd = pd.read_csv(_find_csv("exp5_crossmag_drift_drop.csv"))
    tierc = pd.read_csv(_find_csv("exp_tierc_cross_source_transfer.csv"))

    rows: list[dict] = []
    pending = []
    for bb in BB_ORDER:
        checkpoint = checkpoint_dir / f"{bb}_cap{args.cap}.csv"
        if checkpoint.exists() and not args.force:
            cached = pd.read_csv(checkpoint)
            if len(cached):
                rows.extend(cached.to_dict("records"))
                print(f"[resume] {bb}: loaded {len(cached)} rows from {checkpoint}", flush=True)
                continue
        pending.append((bb, table, monitor_real, tierd, tierc, args.cap))

    tasks = pending
    jobs = max(1, int(args.jobs))
    if not tasks:
        print("[resume] all backbone checkpoints are complete", flush=True)
    elif jobs == 1:
        for task in tasks:
            bb, bb_rows = _run_backbone(task)
            if not bb_rows:
                print(f"[skip] {bb}: no condition features", flush=True)
            else:
                pd.DataFrame(bb_rows).to_csv(
                    checkpoint_dir / f"{bb}_cap{args.cap}.csv", index=False
                )
            rows.extend(bb_rows)
    else:
        workers = min(jobs, len(tasks))
        print(
            f"[parallel] running {len(tasks)} pending backbones with "
            f"workers={workers}, BLAS threads/worker={_BLAS_THREADS}",
            flush=True,
        )
        rows.extend(_run_dynamic(tasks, workers, checkpoint_dir, args.cap))

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("No MMD gamma-sensitivity rows produced. Check feature caches and prerequisite CSVs.")
    expected_per_policy = sum(EXPECTED_MONITOR_COUNTS.values())
    policy_counts = detail["policy"].value_counts().to_dict()
    if any(count != expected_per_policy for count in policy_counts.values()) or len(policy_counts) != 4:
        raise RuntimeError(
            "Incomplete gamma-policy output. "
            f"Expected {expected_per_policy} records for each of four policies, "
            f"observed {policy_counts}."
        )
    by_condition = detail.groupby(["policy", "condition"], as_index=False).agg(
        mmd=("mmd", "mean"),
        failure=("failure", "mean"),
        gamma=("gamma", "mean"),
        n_backbones=("backbone", "nunique"),
        n_records=("mmd", "size"),
    )
    summary_rows = []
    for policy, sub in by_condition.groupby("policy"):
        sub = sub[sub["condition"].isin(CONDITION_ORDER)].copy()
        sub["order"] = sub["condition"].map({c: i for i, c in enumerate(CONDITION_ORDER)})
        sub = sub.sort_values("order")
        worst = sub[sub["condition"] == "TierC_BFS46_FSDM41"]
        milder = sub[sub["condition"] == "TierC_DTSR14_WOODAUTH"]
        summary_rows.append({
            "policy": policy,
            "condition_spearman": _spearman(sub["mmd"].to_numpy(), sub["failure"].to_numpy()),
            "record_spearman": _spearman(
                detail.loc[detail["policy"] == policy, "mmd"].to_numpy(),
                detail.loc[detail["policy"] == policy, "failure"].to_numpy(),
            ),
            "worst_pair_mmd": float(worst.iloc[0]["mmd"]) if len(worst) else np.nan,
            "milder_pair_mmd": float(milder.iloc[0]["mmd"]) if len(milder) else np.nan,
            "worst_below_milder": bool(float(worst.iloc[0]["mmd"]) < float(milder.iloc[0]["mmd"])) if len(worst) and len(milder) else np.nan,
            "n_conditions": int(len(sub)),
            "n_records": int(len(detail[detail["policy"] == policy])),
        })
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(csv_dir / "exp_mmd_gamma_sensitivity_records.csv", index=False)
    by_condition.to_csv(csv_dir / "exp_mmd_gamma_sensitivity_by_condition.csv", index=False)
    summary.to_csv(csv_dir / "exp_mmd_gamma_sensitivity_summary.csv", index=False)
    return detail, by_condition, summary


def plot_summary(by_condition: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    policies = ["per_pair_median", "global_median", "global_median_x0.5", "global_median_x2.0"]
    labels = {
        "clean_TierA": "clean",
        "TierD_xmag_x10x20": "D x10/20",
        "TierD_xmag_x10x50": "D x10/50",
        "TierD_xmag_x20x50": "D x20/50",
        "TierC_DTSR14_WOODAUTH": "C DTSR/WA",
        "TierC_BFS46_FSDM41": "C BFS/FSDM",
    }
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(CONDITION_ORDER))
    width = 0.18
    for i, policy in enumerate(policies):
        sub = by_condition[by_condition["policy"] == policy].set_index("condition")
        vals = [sub.loc[c, "mmd"] if c in sub.index else np.nan for c in CONDITION_ORDER]
        rho = summary.loc[summary["policy"] == policy, "condition_spearman"]
        rho_txt = f", rho={float(rho.iloc[0]):.2f}" if len(rho) else ""
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=f"{policy}{rho_txt}")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in CONDITION_ORDER], rotation=20, ha="right")
    ax.set_ylabel("mean RBF-MMD")
    ax.set_title("MMD severity ranking under bandwidth policies")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_demo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    rows = []
    failures = {
        "clean_TierA": 0.0,
        "TierD_xmag_x10x20": 0.42,
        "TierD_xmag_x10x50": 0.75,
        "TierD_xmag_x20x50": 0.83,
        "TierC_DTSR14_WOODAUTH": 0.68,
        "TierC_BFS46_FSDM41": 0.99,
    }
    for policy in ("per_pair_median", "global_median", "global_median_x0.5", "global_median_x2.0"):
        for cond, failure in failures.items():
            rows.append({
                "policy": policy,
                "condition": cond,
                "mmd": failure * 0.15 + rng.normal(0, 0.01),
                "failure": failure,
                "gamma": 1.0,
                "n_backbones": 7,
                "n_records": 14,
            })
    by_condition = pd.DataFrame(rows)
    summary = by_condition.groupby("policy").apply(lambda s: pd.Series({
        "condition_spearman": _spearman(s["mmd"].to_numpy(), s["failure"].to_numpy()),
        "record_spearman": np.nan,
        "worst_pair_mmd": float(s.loc[s["condition"] == "TierC_BFS46_FSDM41", "mmd"].iloc[0]),
        "milder_pair_mmd": float(s.loc[s["condition"] == "TierC_DTSR14_WOODAUTH", "mmd"].iloc[0]),
        "worst_below_milder": bool(float(s.loc[s["condition"] == "TierC_BFS46_FSDM41", "mmd"].iloc[0]) < float(s.loc[s["condition"] == "TierC_DTSR14_WOODAUTH", "mmd"].iloc[0])),
        "n_conditions": int(len(s)),
        "n_records": int(s["n_records"].sum()),
    })).reset_index()
    return by_condition, by_condition, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--csv", default="all_public_datasets_standardized.csv")
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=1, help="parallel backbone workers; use 2-4 on Colab depending on RAM")
    ap.add_argument("--force", action="store_true", help="ignore per-backbone checkpoints and recompute")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    if args.real:
        _detail, by_condition, summary = run_real(args)
    else:
        if not args.demo:
            print("[no mode given] defaulting to --demo", flush=True)
        _detail, by_condition, summary = make_demo()
    print("\n=== MMD gamma sensitivity: by condition ===")
    print(by_condition.to_string(index=False))
    print("\n=== MMD gamma sensitivity: summary ===")
    print(summary.to_string(index=False))
    if args.real and not args.no_fig:
        _csv_dir, fig_dir = _io_dirs()
        plot_summary(by_condition, summary, fig_dir / "mmd_gamma_sensitivity.png")
        print(f"\nSaved figure to {fig_dir / 'mmd_gamma_sensitivity.png'}")
    if args.real:
        csv_dir, fig_dir = _io_dirs()
        outputs = [
            csv_dir / "exp_mmd_gamma_sensitivity_records.csv",
            csv_dir / "exp_mmd_gamma_sensitivity_by_condition.csv",
            csv_dir / "exp_mmd_gamma_sensitivity_summary.csv",
        ]
        if not args.no_fig:
            outputs.extend([
                fig_dir / "mmd_gamma_sensitivity.png",
                fig_dir / "mmd_gamma_sensitivity.pdf",
            ])
        write_provenance(
            "exp_mmd_gamma_sensitivity",
            outputs,
            protocol="shared_gamma_feature_sampling_v1",
            parameters={
                "cap": args.cap,
                "jobs": args.jobs,
                "force": args.force,
                "seed_policy": "blake2s_stable",
            },
            inputs=[
                require_csv("exp5_crossmag_drift_drop.csv"),
                require_csv("exp_tierc_cross_source_transfer.csv"),
            ],
        )


if __name__ == "__main__":
    main()
