#!/usr/bin/env python3
"""
Tier-C real cross-source acquisition-shift analysis.

This experiment evaluates clean-feature transfer between public datasets that
share accepted species names. It is unpaired: there is no same image under two
conditions. The drift signal is therefore distributional class-centroid drift,
not paired clean/shifted feature drift.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wood_spatial.config import BB_LABEL, BB_ORDER, BASE, V2_CACHE_DIR, V4_FEAT_CACHE
from wood_spatial.result_io import csv_dir, figure_dir, require_csv


PAIRS = [("BFS46", "FSDM41"), ("DTSR14", "WOODAUTH")]
K = 5


def _norm(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def _norm_key(s: object) -> str:
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


def _is_real_binomial(name: object) -> bool:
    toks = str(name).split()
    return len(toks) >= 2 and toks[0] != "Plantae" and toks[1][:1].islower()


def _find_csv(name: str, preferred_dir: Path) -> Path:
    del preferred_dir
    return require_csv(name)


def _resolve_species_csv(csv_arg: str) -> Path:
    path = Path(csv_arg)
    candidates = [
        path,
        Path.cwd() / path,
        BASE / path,
        Path(__file__).resolve().parents[2] / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the standardized species CSV. Put it at the repository "
        f"root or pass --csv explicitly. Tried: {', '.join(str(c) for c in candidates)}"
    )


def _output_dirs() -> tuple[Path, Path]:
    return csv_dir(), figure_dir()


def _safe_tag(tag: str, maxlen: int = 60) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)[:maxlen]


def _configured_results_dirs() -> list[Path]:
    dirs = []
    env_dir = os.environ.get("WOOD_RESULTS_DIR")
    if env_dir:
        dirs.append(Path(env_dir))
    cfg_path = BASE / "configs" / "full_colab_l4.json"
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            results_dir = cfg.get("paths", {}).get("results_dir")
            if results_dir:
                dirs.append(Path(results_dir))
        except Exception:
            pass
    dirs.append(BASE / "results")
    out = []
    seen = set()
    for d in dirs:
        key = str(d)
        if key not in seen:
            out.append(d)
            seen.add(key)
    return out


def _cache_candidates(backbone: str, dataset: str, tag: str = "original") -> list[Path]:
    safe = _safe_tag(tag)
    candidates = []
    for results_dir in _configured_results_dirs():
        candidates.append(results_dir / "feature_cache" / f"{backbone}_{dataset}_{safe}.npz")
    candidates.append(V4_FEAT_CACHE / f"{backbone}_{dataset}_{safe}.npz")
    candidates.append(V2_CACHE_DIR / f"{backbone}_{dataset}_all_{safe}.npz")
    return candidates


def _load_feature_cache_np(backbone: str, dataset: str, tag: str = "original"):
    checked = _cache_candidates(backbone, dataset, tag)
    for path in checked:
        if path.exists():
            data = np.load(path, allow_pickle=True)
            return data["features"], data["labels"], data["paths"], path
    raise FileNotFoundError(
        f"Cache not found for {backbone}/{dataset}/{tag}. Checked: "
        + ", ".join(str(p) for p in checked)
    )


def _species_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"dataset", "original_name", "gbif_accepted_name"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain dataset/original_name/gbif_accepted_name columns")
    df = df[df["gbif_accepted_name"].map(_is_real_binomial)].copy()
    df["folder_key"] = df["original_name"].map(_norm_key)
    return df


def _shared_species(df: pd.DataFrame, ds_a: str, ds_b: str) -> list[str]:
    a = set(df.loc[df["dataset"] == ds_a, "gbif_accepted_name"])
    b = set(df.loc[df["dataset"] == ds_b, "gbif_accepted_name"])
    return sorted(a & b)


def _label_map(df: pd.DataFrame, dataset: str) -> dict[str, str]:
    sub = df[df["dataset"] == dataset]
    out = {}
    for _, row in sub.iterrows():
        accepted = str(row["gbif_accepted_name"]).strip()
        for col in ("original_name", "standardized_name", "canonical_binomial", "gbif_accepted_name"):
            if col in row and pd.notna(row[col]):
                out[_norm_key(row[col])] = accepted
    return out


def _features_by_species(dataset: str, backbone: str, table: pd.DataFrame) -> dict[str, np.ndarray]:
    features, labels, paths, _cache_path = _load_feature_cache_np(backbone, dataset, "original")
    features = _norm(np.asarray(features, dtype=np.float32))
    labels = np.asarray(labels)
    paths = np.asarray(paths)
    name_map = _label_map(table, dataset)

    by_species: dict[str, list[np.ndarray]] = {}
    for feat, lab, path in zip(features, labels, paths):
        candidates = []
        if path is not None and str(path):
            candidates.append(Path(str(path)).parent.name)
        candidates.append(str(lab))
        accepted = None
        for cand in candidates:
            accepted = name_map.get(_norm_key(cand))
            if accepted:
                break
        if accepted is None:
            continue
        by_species.setdefault(accepted, []).append(feat)

    return {s: np.vstack(v) for s, v in by_species.items() if v}


def _cache_species_counts(dataset: str, backbone: str, table: pd.DataFrame) -> tuple[dict[str, int], dict]:
    features, labels, paths, cache_path = _load_feature_cache_np(backbone, dataset, "original")
    labels = np.asarray(labels)
    paths = np.asarray(paths)
    name_map = _label_map(table, dataset)
    counts: dict[str, int] = {}
    unmatched = 0
    unmatched_examples = []
    path_hits = 0
    label_hits = 0
    for lab, path in zip(labels, paths):
        candidates = []
        if path is not None and str(path):
            candidates.append(("path", Path(str(path)).parent.name))
        candidates.append(("label", str(lab)))
        accepted = None
        hit_kind = None
        for kind, cand in candidates:
            accepted = name_map.get(_norm_key(cand))
            if accepted:
                hit_kind = kind
                break
        if accepted is None:
            unmatched += 1
            if len(unmatched_examples) < 5:
                unmatched_examples.append({"label": str(lab), "path_parent": Path(str(path)).parent.name if path is not None else ""})
            continue
        counts[accepted] = counts.get(accepted, 0) + 1
        if hit_kind == "path":
            path_hits += 1
        elif hit_kind == "label":
            label_hits += 1
    info = {
        "n_features": int(len(features)),
        "n_labels": int(len(labels)),
        "n_paths": int(len(paths)),
        "n_mapped": int(sum(counts.values())),
        "n_unmatched": int(unmatched),
        "path_hits": int(path_hits),
        "label_hits": int(label_hits),
        "unmatched_examples": unmatched_examples,
        "cache_path": str(cache_path),
    }
    return counts, info


def _check_only(table: pd.DataFrame) -> None:
    print("\n=== Tier-C shared species check ===", flush=True)
    all_ok = True
    for a, b in PAIRS:
        species = _shared_species(table, a, b)
        print(f"\n{a}<->{b}: {len(species)} shared accepted species", flush=True)
        for i, s in enumerate(species, start=1):
            print(f"  {i:02d}. {s}", flush=True)

    print("\n=== Cache and species-mapping check ===", flush=True)
    for a, b in PAIRS:
        required = _shared_species(table, a, b)
        print(f"\nPair {a}<->{b}", flush=True)
        for bb in BB_ORDER:
            pair_ok = True
            per_ds = {}
            for ds in (a, b):
                try:
                    counts, info = _cache_species_counts(ds, bb, table)
                except FileNotFoundError as exc:
                    all_ok = False
                    pair_ok = False
                    print(f"  FAIL {bb}/{ds}: missing cache ({exc})", flush=True)
                    continue
                missing = [s for s in required if counts.get(s, 0) == 0]
                if missing:
                    all_ok = False
                    pair_ok = False
                per_ds[ds] = (counts, info, missing)
                print(
                    f"  {bb}/{ds}: cache_rows={info['n_features']} mapped={info['n_mapped']} "
                    f"unmatched={info['n_unmatched']} path_hits={info['path_hits']} label_hits={info['label_hits']} "
                    f"shared_present={len(required)-len(missing)}/{len(required)}",
                    flush=True,
                )
                print(f"    cache: {info['cache_path']}", flush=True)
                if missing:
                    print("    missing shared species:", "; ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""), flush=True)
                if info["unmatched_examples"]:
                    ex = info["unmatched_examples"][0]
                    print(f"    example unmatched: label={ex['label']} path_parent={ex['path_parent']}", flush=True)
            if pair_ok:
                mins = []
                for s in required:
                    vals = [per_ds[ds][0].get(s, 0) for ds in (a, b) if ds in per_ds]
                    if vals:
                        mins.append(min(vals))
                print(f"  OK {bb}/{a}<->{b}: all shared species mapped; min paired images/species-side={min(mins) if mins else 0}", flush=True)
    if not all_ok:
        raise SystemExit("\nCheck failed: fix missing caches/species mapping before running --real.")
    print("\nCheck passed: all Tier-C shared species are present in the dataset-level clean caches.", flush=True)


def _knn_predict(gallery: np.ndarray, y_gallery: np.ndarray, query: np.ndarray, k: int = K) -> np.ndarray:
    sims = query @ gallery.T
    kk = min(k, gallery.shape[0])
    idx = np.argpartition(-sims, kth=kk - 1, axis=1)[:, :kk]
    votes = y_gallery[idx]
    pred = []
    for row in votes:
        vals, counts = np.unique(row, return_counts=True)
        pred.append(vals[np.argmax(counts)])
    return np.asarray(pred)


def _within_recall(features_by_species: dict[str, np.ndarray], species: list[str]) -> dict[str, float]:
    Xs, ys = [], []
    for j, s in enumerate(species):
        X = features_by_species[s]
        Xs.append(X)
        ys.extend([j] * len(X))
    X = np.vstack(Xs)
    y = np.asarray(ys)
    sims = X @ X.T
    np.fill_diagonal(sims, -np.inf)
    kk = min(K, max(1, X.shape[0] - 1))
    idx = np.argpartition(-sims, kth=kk - 1, axis=1)[:, :kk]
    votes = y[idx]
    pred = []
    for row in votes:
        vals, counts = np.unique(row, return_counts=True)
        pred.append(vals[np.argmax(counts)])
    pred = np.asarray(pred)
    return {
        s: float(np.mean(pred[y == j] == j))
        for j, s in enumerate(species)
    }


def _direction_records(
    feats_src: dict[str, np.ndarray],
    feats_dst: dict[str, np.ndarray],
    species: list[str],
    backbone: str,
    pair: str,
    direction: str,
) -> tuple[list[dict], float]:
    sp_to_int = {s: i for i, s in enumerate(species)}
    gX, gy, qX, qy = [], [], [], []
    for s in species:
        gX.append(feats_src[s])
        gy.extend([sp_to_int[s]] * len(feats_src[s]))
        qX.append(feats_dst[s])
        qy.extend([sp_to_int[s]] * len(feats_dst[s]))
    gX = np.vstack(gX)
    qX = np.vstack(qX)
    gy = np.asarray(gy)
    qy = np.asarray(qy)
    pred = _knn_predict(gX, gy, qX)
    overall_acc = float(np.mean(pred == qy))

    within = _within_recall(feats_src, species)
    rows = []
    for s in species:
        cls = sp_to_int[s]
        mask = qy == cls
        cross_recall = float(np.mean(pred[mask] == cls)) if np.any(mask) else np.nan
        cs = feats_src[s].mean(axis=0, keepdims=True)
        cd = feats_dst[s].mean(axis=0, keepdims=True)
        cs = _norm(cs)[0]
        cd = _norm(cd)[0]
        drift = float(1.0 - cs @ cd)
        clean_recall = within[s]
        rows.append({
            "backbone": backbone,
            "pair": pair,
            "direction": direction,
            "source_dataset": direction.split("->")[0],
            "target_dataset": direction.split("->")[1],
            "species": s,
            "centroid_drift": drift,
            "within_source_recall": clean_recall,
            "cross_source_recall": cross_recall,
            "accuracy_drop": clean_recall - cross_recall,
            "n_source": int(len(feats_src[s])),
            "n_target": int(len(feats_dst[s])),
        })
    return rows, overall_acc


def _tier_a_line(csv_dir: Path) -> tuple[float, float, bool, str]:
    path = _find_csv("exp1b_feature_geometry.csv", csv_dir)
    if not path.exists():
        return np.nan, np.nan, False, str(path)
    df = pd.read_csv(path)
    if "accuracy_drop" not in df.columns and "drop" in df.columns:
        df = df.rename(columns={"drop": "accuracy_drop"})
    if not {"feature_drift", "accuracy_drop"}.issubset(df.columns):
        return np.nan, np.nan, False, str(path)
    x = df["feature_drift"].to_numpy(dtype=float)
    y = df["accuracy_drop"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan, False, str(path)
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return float(slope), float(intercept), True, str(path)


def _demo(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    rows = []
    transfer = []
    for a, b in PAIRS:
        species = _shared_species(table, a, b)
        for bb in BB_ORDER:
            for src, dst, offset in [(a, b, 0.0), (b, a, 0.08)]:
                accs = []
                for s in species:
                    drift = float(np.clip(rng.normal(0.35 + offset, 0.12), 0.03, 0.9))
                    drop = float(np.clip(0.05 + 0.8 * drift + rng.normal(0, 0.12), -0.2, 1.0))
                    cross = float(np.clip(0.95 - drop, 0, 1))
                    rows.append({
                        "backbone": bb,
                        "pair": f"{a}<->{b}",
                        "direction": f"{src}->{dst}",
                        "source_dataset": src,
                        "target_dataset": dst,
                        "species": s,
                        "centroid_drift": drift,
                        "within_source_recall": 0.95,
                        "cross_source_recall": cross,
                        "accuracy_drop": drop,
                        "n_source": 50,
                        "n_target": 50,
                    })
                    accs.append(cross)
                transfer.append({
                    "pair": f"{a}<->{b}",
                    "direction": f"{src}->{dst}",
                    "backbone": bb,
                    "cross_source_accuracy": float(np.mean(accs)),
                    "n_species": len(species),
                })
    return pd.DataFrame(rows), pd.DataFrame(transfer)


def _run_real(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    transfer = []
    for a, b in PAIRS:
        species_all = _shared_species(table, a, b)
        for bb in BB_ORDER:
            fa = _features_by_species(a, bb, table)
            fb = _features_by_species(b, bb, table)
            species = [s for s in species_all if s in fa and s in fb]
            if not species:
                continue
            for src, dst, fs, fd in [(a, b, fa, fb), (b, a, fb, fa)]:
                recs, acc = _direction_records(
                    fs, fd, species, bb, f"{a}<->{b}", f"{src}->{dst}"
                )
                rows.extend(recs)
                transfer.append({
                    "pair": f"{a}<->{b}",
                    "direction": f"{src}->{dst}",
                    "backbone": bb,
                    "cross_source_accuracy": acc,
                    "n_species": len(species),
                })
            print(f"[done] {a}<->{b} / {bb}: shared_species={len(species)}", flush=True)
    if not rows:
        raise RuntimeError("No usable shared-species cache rows found. Check feature caches and standardized CSV.")
    return pd.DataFrame(rows), pd.DataFrame(transfer)


def _summaries(by_cell: pd.DataFrame, transfer: pd.DataFrame, csv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = by_cell["centroid_drift"].to_numpy(dtype=float)
    y = by_cell["accuracy_drop"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 3 else np.nan
    tier_slope, tier_intercept, tier_ok, tier_csv = _tier_a_line(csv_dir)
    if tier_ok:
        pred = tier_slope * x[ok] + tier_intercept
        residual = float(np.mean(np.abs(y[ok] - pred)))
        ratio = float(slope / tier_slope) if tier_slope else np.nan
    else:
        residual = np.nan
        ratio = np.nan
    summary = pd.DataFrame([{
        "tier_a_csv": tier_csv,
        "tier_a_available": bool(tier_ok),
        "tier_a_slope": tier_slope,
        "tier_a_intercept": tier_intercept,
        "crosssource_slope": float(slope),
        "crosssource_intercept": float(intercept),
        "crosssource_drift_drop_r": r,
        "crosssource_to_tier_a_slope_ratio": ratio,
        "crosssource_mean_abs_residual_on_tier_a_line": residual,
        "n_crosssource_records": int(ok.sum()),
    }])

    means = transfer.groupby(["pair", "direction"], as_index=False).agg(
        mean_cross_source_accuracy=("cross_source_accuracy", "mean"),
        std_cross_source_accuracy=("cross_source_accuracy", "std"),
        n_backbones=("backbone", "nunique"),
        n_species=("n_species", "max"),
    )
    asym_rows = []
    for pair, sub in means.groupby("pair"):
        dirs = {r["direction"]: r for _, r in sub.iterrows()}
        parts = pair.split("<->")
        if len(parts) != 2:
            continue
        a, b = parts
        ab = dirs.get(f"{a}->{b}")
        ba = dirs.get(f"{b}->{a}")
        if ab is None or ba is None:
            continue
        asym_rows.append({
            "pair": pair,
            "acc_a_to_b": float(ab["mean_cross_source_accuracy"]),
            "acc_b_to_a": float(ba["mean_cross_source_accuracy"]),
            "signed_asymmetry": float(ab["mean_cross_source_accuracy"] - ba["mean_cross_source_accuracy"]),
            "n_backbones": int(ab["n_backbones"]),
            "n_species": int(ab["n_species"]),
        })
    by_pair = pd.DataFrame(asym_rows)
    return summary, by_pair


def _plot(by_cell: pd.DataFrame, transfer: pd.DataFrame, summary: pd.DataFrame, fig_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    x = by_cell["centroid_drift"].to_numpy(dtype=float)
    y = by_cell["accuracy_drop"].to_numpy(dtype=float)
    ax.scatter(x, y, s=14, alpha=0.35, label="species x direction x backbone")
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    xs = np.linspace(max(0, xmin - 0.02), xmax + 0.02, 100)
    row = summary.iloc[0]
    if bool(row["tier_a_available"]):
        ax.plot(xs, row["tier_a_slope"] * xs + row["tier_a_intercept"], "k--", lw=1.5, label="Tier-A fit")
    ax.plot(xs, row["crosssource_slope"] * xs + row["crosssource_intercept"], color="#E45756", lw=1.5, label="cross-source fit")
    ax.set_xlabel("distributional feature drift (class-centroid cosine)")
    ax.set_ylabel("accuracy drop")
    ax.set_title(f"(a) Cross-source drift tracks drop (r={row['crosssource_drift_drop_r']:.3f})")
    ax.legend(fontsize=8)

    ax = axes[1]
    means = transfer.groupby(["pair", "direction"], as_index=False)["cross_source_accuracy"].mean()
    labels = [f"{r.pair}: {r.direction}" for _, r in means.iterrows()]
    vals = means["cross_source_accuracy"].to_numpy()
    ax.barh(np.arange(len(vals)), vals, color="#4C78A8")
    ax.set_yticks(np.arange(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean cross-source transfer accuracy")
    ax.set_title("(b) Directed real cross-source transfer")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Run Tier-C real cross-source acquisition-shift analysis.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="use clean feature caches")
    mode.add_argument("--demo", action="store_true", help="use synthetic demo data")
    mode.add_argument("--check-only", action="store_true", help="check shared species and clean cache mapping without running kNN transfer")
    ap.add_argument("--csv", default="all_public_datasets_standardized.csv", help="standardized public-dataset species CSV")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    csv_dir, fig_dir = _output_dirs()
    csv_path = _resolve_species_csv(args.csv)
    table = _species_table(csv_path)
    print(f"Species CSV: {csv_path}", flush=True)
    print(f"Mode: {'check-only' if args.check_only else 'demo' if args.demo else 'real'}", flush=True)
    for a, b in PAIRS:
        print(f"Shared species {a}<->{b}: {len(_shared_species(table, a, b))}", flush=True)

    if args.check_only:
        _check_only(table)
        return
    elif args.demo:
        by_cell, transfer = _demo(table)
    else:
        by_cell, transfer = _run_real(table)

    summary, by_pair = _summaries(by_cell, transfer, csv_dir)
    print("\n=== Tier-C cross-source summary ===")
    print(summary.to_string(index=False))
    print("\n=== Directional transfer means ===")
    print(transfer.groupby(["pair", "direction"], as_index=False)["cross_source_accuracy"].mean().to_string(index=False))
    print("\n=== Pair asymmetry ===")
    print(by_pair.to_string(index=False))

    save = (not args.no_save) and (not args.demo)
    if save:
        csv_dir.mkdir(parents=True, exist_ok=True)
        by_cell.to_csv(csv_dir / "exp_tierc_cross_source_by_cell.csv", index=False)
        transfer.to_csv(csv_dir / "exp_tierc_cross_source_transfer.csv", index=False)
        by_pair.to_csv(csv_dir / "exp_tierc_cross_source_by_pair.csv", index=False)
        summary.to_csv(csv_dir / "exp_tierc_cross_source_summary.csv", index=False)
        if not args.no_fig:
            _plot(by_cell, transfer, summary, fig_dir / "tierc_cross_source_shift.png")
        print(f"\nSaved CSV outputs to {csv_dir}")
        if not args.no_fig:
            print(f"Saved figure to {fig_dir / 'tierc_cross_source_shift.png'}")
    elif args.demo:
        print("\nDemo mode does not save outputs. Run with --real on the full cache to write CSV/figure files.")
    elif not args.no_fig:
        _plot(by_cell, transfer, summary, Path("tierc_cross_source_shift.png"))


if __name__ == "__main__":
    main()
