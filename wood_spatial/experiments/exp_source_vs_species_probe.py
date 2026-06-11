#!/usr/bin/env python3
"""
Tier-C source-vs-species geometry probe.

This experiment directly tests whether frozen features on shared-species
cross-source pairs are organized more by acquisition source than by species.
It reuses the same standardized species table and cache mapping as
exp_tierc_cross_source_shift.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wood_spatial.config import BB_LABEL, BB_ORDER
from wood_spatial.experiments.exp_tierc_cross_source_shift import (
    PAIRS,
    _features_by_species,
    _norm,
    _output_dirs,
    _resolve_species_csv,
    _shared_species,
    _species_table,
)


def _centroid_predict(gallery_centroids: np.ndarray, labels: np.ndarray, query: np.ndarray) -> np.ndarray:
    sims = query @ gallery_centroids.T
    return labels[np.argmax(sims, axis=1)]


def _cross_species_acc(gallery: dict[str, np.ndarray], query: dict[str, np.ndarray], species: list[str]) -> float:
    centroids = _norm(np.vstack([gallery[s].mean(axis=0, keepdims=True) for s in species]))
    labels = np.arange(len(species))
    accs = []
    for j, s in enumerate(species):
        pred = _centroid_predict(centroids, labels, query[s])
        accs.append(float(np.mean(pred == j)))
    return float(np.mean(accs)) if accs else np.nan


def _mean_pairwise_centroid_dist(features_by_species: dict[str, np.ndarray], species: list[str]) -> float:
    if len(species) < 2:
        return np.nan
    centroids = _norm(np.vstack([features_by_species[s].mean(axis=0, keepdims=True) for s in species]))
    vals = []
    for i in range(len(species)):
        for j in range(i + 1, len(species)):
            vals.append(np.linalg.norm(centroids[i] - centroids[j]))
    return float(np.mean(vals)) if vals else np.nan


def probe_pair(feats_a: dict[str, np.ndarray], feats_b: dict[str, np.ndarray], ds_a: str, ds_b: str, species: list[str]) -> dict:
    xa = np.vstack([feats_a[s] for s in species])
    xb = np.vstack([feats_b[s] for s in species])

    ca = _norm(xa.mean(axis=0, keepdims=True))
    cb = _norm(xb.mean(axis=0, keepdims=True))
    source_centroids = np.vstack([ca, cb])
    source_labels = np.array([0, 1])
    x_all = np.vstack([xa, xb])
    y_source = np.array([0] * len(xa) + [1] * len(xb))
    source_pred = _centroid_predict(source_centroids, source_labels, x_all)

    ab = _cross_species_acc(feats_a, feats_b, species)
    ba = _cross_species_acc(feats_b, feats_a, species)
    d_source = float(np.linalg.norm(ca[0] - cb[0]))
    d_species = 0.5 * (
        _mean_pairwise_centroid_dist(feats_a, species)
        + _mean_pairwise_centroid_dist(feats_b, species)
    )

    return {
        "pair": f"{ds_a}<->{ds_b}",
        "n_species": int(len(species)),
        "source_acc": float(np.mean(source_pred == y_source)),
        "cross_source_species_acc": float(0.5 * (ab + ba)),
        "cross_source_species_acc_a_to_b": float(ab),
        "cross_source_species_acc_b_to_a": float(ba),
        "between_source_dist": d_source,
        "between_species_dist": float(d_species),
        "source_to_species_ratio": float(d_source / (d_species + 1e-12)),
        "n_features_a": int(len(xa)),
        "n_features_b": int(len(xb)),
    }


def _demo(table: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for ds_a, ds_b in PAIRS:
        n_species = len(_shared_species(table, ds_a, ds_b)) or 12
        for bb in BB_ORDER:
            d = 64
            per = 40
            centers = _norm(rng.normal(size=(n_species, d)))
            off_a = rng.normal(size=d)
            off_b = rng.normal(size=d)
            off_a = 1.2 * off_a / np.linalg.norm(off_a)
            off_b = 1.2 * off_b / np.linalg.norm(off_b)
            perm = rng.permutation(n_species)
            fa, fb = {}, {}
            species = [f"sp{i:02d}" for i in range(n_species)]
            for i, s in enumerate(species):
                fa[s] = _norm(centers[i] + off_a + rng.normal(0, 0.25, size=(per, d)))
                fb[s] = _norm(centers[perm[i]] + off_b + rng.normal(0, 0.25, size=(per, d)))
            row = probe_pair(fa, fb, ds_a, ds_b, species)
            row["backbone"] = bb
            rows.append(row)
    return pd.DataFrame(rows)


def _run_real(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds_a, ds_b in PAIRS:
        species_all = _shared_species(table, ds_a, ds_b)
        for bb in BB_ORDER:
            fa = _features_by_species(ds_a, bb, table)
            fb = _features_by_species(ds_b, bb, table)
            species = [s for s in species_all if s in fa and s in fb]
            if not species:
                print(f"[skip] {ds_a}<->{ds_b} / {bb}: no shared species in caches", flush=True)
                continue
            row = probe_pair(fa, fb, ds_a, ds_b, species)
            row["backbone"] = bb
            rows.append(row)
            print(f"[done] {ds_a}<->{ds_b} / {bb}: shared_species={len(species)}", flush=True)
    if not rows:
        raise RuntimeError("No source-vs-species probe rows produced. Check Tier-C caches and species mapping.")
    return pd.DataFrame(rows)


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby("pair", as_index=False).agg(
        n_species=("n_species", "max"),
        source_acc_mean=("source_acc", "mean"),
        source_acc_std=("source_acc", "std"),
        cross_source_species_acc_mean=("cross_source_species_acc", "mean"),
        cross_source_species_acc_std=("cross_source_species_acc", "std"),
        source_to_species_ratio_mean=("source_to_species_ratio", "mean"),
        source_to_species_ratio_std=("source_to_species_ratio", "std"),
        between_source_dist_mean=("between_source_dist", "mean"),
        between_species_dist_mean=("between_species_dist", "mean"),
        n_backbones=("backbone", "nunique"),
    )


def _plot(rows: pd.DataFrame, summary: pd.DataFrame, fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    pairs = summary["pair"].tolist()
    x = np.arange(len(pairs))

    ax = axes[0]
    width = 0.36
    ax.bar(x - width / 2, summary["source_acc_mean"], width, label="source accuracy", color="#4C78A8")
    ax.bar(x + width / 2, summary["cross_source_species_acc_mean"], width, label="cross-source species accuracy", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("nearest-centroid accuracy")
    ax.set_title("(a) Source separates while species transfer collapses")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x, summary["source_to_species_ratio_mean"], color="#72B7B2")
    ax.axhline(1.0, color="k", ls="--", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=15, ha="right")
    ax.set_ylabel("between-source / between-species distance")
    ax.set_title("(b) Source gap relative to species geometry")

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Tier-C source-vs-species geometry probe.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true")
    mode.add_argument("--demo", action="store_true")
    ap.add_argument("--csv", default="all_public_datasets_standardized.csv")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    csv_dir, fig_dir = _output_dirs()
    table = _species_table(_resolve_species_csv(args.csv))
    rows = _run_real(table) if args.real else _demo(table)
    summary = _summarize(rows)

    print("\n=== Source-vs-species probe by pair ===")
    print(summary.to_string(index=False))
    print("\n=== Per-backbone rows ===")
    print(rows.to_string(index=False))

    if args.real and not args.no_save:
        csv_dir.mkdir(parents=True, exist_ok=True)
        rows.to_csv(csv_dir / "exp_source_vs_species_probe_by_backbone.csv", index=False)
        summary.to_csv(csv_dir / "exp_source_vs_species_probe_summary.csv", index=False)
        if not args.no_fig:
            _plot(rows, summary, fig_dir / "source_vs_species_probe.png")
        print(f"\nSaved CSV outputs to {csv_dir}")
        if not args.no_fig:
            print(f"Saved figure to {fig_dir / 'source_vs_species_probe.png'}")
    elif not args.no_fig:
        _plot(rows, summary, Path("source_vs_species_probe.png"))


if __name__ == "__main__":
    main()
