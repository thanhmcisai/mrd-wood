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


def _balanced_source_arrays(
    feats_a: dict[str, np.ndarray],
    feats_b: dict[str, np.ndarray],
    species: list[str],
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Balance each species across sources so source prediction cannot use counts."""
    out_a, out_b = {}, {}
    for s in species:
        n = min(len(feats_a[s]), len(feats_b[s]))
        ia = rng.choice(len(feats_a[s]), n, replace=False)
        ib = rng.choice(len(feats_b[s]), n, replace=False)
        out_a[s] = feats_a[s][ia]
        out_b[s] = feats_b[s][ib]
    return out_a, out_b


def _source_accuracy_resubstitution(
    feats_a: dict[str, np.ndarray],
    feats_b: dict[str, np.ndarray],
    species: list[str],
) -> float:
    xa = np.vstack([feats_a[s] for s in species])
    xb = np.vstack([feats_b[s] for s in species])
    centroids = _norm(np.vstack([xa.mean(axis=0), xb.mean(axis=0)]))
    x = np.vstack([xa, xb])
    y = np.r_[np.zeros(len(xa), dtype=int), np.ones(len(xb), dtype=int)]
    return float(np.mean(_centroid_predict(centroids, np.array([0, 1]), x) == y))


def _source_accuracy_heldout_images(
    feats_a: dict[str, np.ndarray],
    feats_b: dict[str, np.ndarray],
    species: list[str],
    seeds: tuple[int, ...] = (11, 23, 37, 53, 71),
) -> tuple[float, float]:
    scores = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ba, bb = _balanced_source_arrays(feats_a, feats_b, species, rng)
        train_a, train_b, test_a, test_b = [], [], [], []
        for s in species:
            n = len(ba[s])
            order = rng.permutation(n)
            n_test = max(1, int(round(0.2 * n)))
            n_test = min(n_test, n - 1)
            test_idx, train_idx = order[:n_test], order[n_test:]
            train_a.append(ba[s][train_idx])
            train_b.append(bb[s][train_idx])
            test_a.append(ba[s][test_idx])
            test_b.append(bb[s][test_idx])
        train_a, train_b = np.vstack(train_a), np.vstack(train_b)
        test_a, test_b = np.vstack(test_a), np.vstack(test_b)
        centroids = _norm(np.vstack([train_a.mean(axis=0), train_b.mean(axis=0)]))
        x = np.vstack([test_a, test_b])
        y = np.r_[np.zeros(len(test_a), dtype=int), np.ones(len(test_b), dtype=int)]
        scores.append(float(np.mean(_centroid_predict(centroids, np.array([0, 1]), x) == y)))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def _source_accuracy_leave_one_species_out(
    feats_a: dict[str, np.ndarray],
    feats_b: dict[str, np.ndarray],
    species: list[str],
    seeds: tuple[int, ...] = tuple(range(20)),
) -> tuple[float, float]:
    seed_scores = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ba, bb = _balanced_source_arrays(feats_a, feats_b, species, rng)
        fold_scores = []
        fold_sizes = []
        for held_out in species:
            train_species = [s for s in species if s != held_out]
            train_a = np.vstack([ba[s] for s in train_species])
            train_b = np.vstack([bb[s] for s in train_species])
            centroids = _norm(np.vstack([train_a.mean(axis=0), train_b.mean(axis=0)]))
            test_a, test_b = ba[held_out], bb[held_out]
            x = np.vstack([test_a, test_b])
            y = np.r_[np.zeros(len(test_a), dtype=int), np.ones(len(test_b), dtype=int)]
            pred = _centroid_predict(centroids, np.array([0, 1]), x)
            fold_scores.append(float(np.mean(pred == y)))
            fold_sizes.append(len(y))
        seed_scores.append(float(np.average(fold_scores, weights=fold_sizes)))
    return float(np.mean(seed_scores)), float(np.std(seed_scores, ddof=1))


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
    source_resub = _source_accuracy_resubstitution(feats_a, feats_b, species)
    source_heldout, source_heldout_std = _source_accuracy_heldout_images(
        feats_a, feats_b, species
    )
    source_loso, source_loso_std = _source_accuracy_leave_one_species_out(
        feats_a, feats_b, species
    )

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
        "source_acc_resubstitution": source_resub,
        "source_acc_heldout_image": source_heldout,
        "source_acc_heldout_image_split_std": source_heldout_std,
        "source_acc_leave_one_species_out": source_loso,
        "source_acc_leave_one_species_out_seed_std": source_loso_std,
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
        source_acc_resubstitution_mean=("source_acc_resubstitution", "mean"),
        source_acc_heldout_image_mean=("source_acc_heldout_image", "mean"),
        source_acc_heldout_image_std=("source_acc_heldout_image", "std"),
        source_acc_leave_one_species_out_mean=("source_acc_leave_one_species_out", "mean"),
        source_acc_leave_one_species_out_std=("source_acc_leave_one_species_out", "std"),
        cross_source_species_acc_mean=("cross_source_species_acc", "mean"),
        cross_source_species_acc_std=("cross_source_species_acc", "std"),
        source_to_species_ratio_mean=("source_to_species_ratio", "mean"),
        source_to_species_ratio_std=("source_to_species_ratio", "std"),
        between_source_dist_mean=("between_source_dist", "mean"),
        between_species_dist_mean=("between_species_dist", "mean"),
        n_backbones=("backbone", "nunique"),
    )


def _plot(rows: pd.DataFrame, summary: pd.DataFrame, fig_path: Path) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.8),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    plot_rows = rows.copy()
    plot_rows["backbone_label"] = plot_rows["backbone"].map(BB_LABEL).fillna(plot_rows["backbone"])
    plot_rows["_order"] = plot_rows["backbone"].map({bb: i for i, bb in enumerate(BB_ORDER)})
    plot_rows = plot_rows.sort_values("_order")

    ax = axes[0]
    y = np.arange(len(plot_rows))
    for yi, row in zip(y, plot_rows.itertuples(index=False)):
        ax.plot(
            [row.source_acc_leave_one_species_out, row.cross_source_species_acc],
            [yi, yi],
            color="#9AA3AD",
            lw=1.4,
            zorder=1,
        )
    ax.scatter(
        plot_rows["source_acc_leave_one_species_out"],
        y,
        s=58,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.7,
        label="source identity, held-out species",
        zorder=3,
    )
    ax.scatter(
        plot_rows["cross_source_species_acc"],
        y,
        s=58,
        color="#E45756",
        edgecolor="white",
        linewidth=0.7,
        label="species transfer, cross-source",
        zorder=3,
    )
    n_species = float(summary["n_species"].iloc[0])
    ax.axvline(0.5, color="#4C78A8", ls=":", lw=1.1, alpha=0.7)
    ax.axvline(1.0 / n_species, color="#E45756", ls=":", lw=1.1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_rows["backbone_label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("nearest-centroid accuracy")
    ax.set_title("(a) Source remains separable while species transfer drops")
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    ax.legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.55, -0.16),
        ncol=2,
        frameon=False,
    )
    ax.text(0.505, len(plot_rows) - 0.15, "source chance", fontsize=7, color="#4C78A8", rotation=90, va="bottom")
    ax.text(1.0 / n_species + 0.006, len(plot_rows) - 0.15, "species chance", fontsize=7, color="#E45756", rotation=90, va="bottom")

    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0.20, 0.82, len(plot_rows)))
    ax.barh(y, plot_rows["source_to_species_ratio"], color=colors, edgecolor="white", linewidth=0.6)
    ax.axvline(1.0, color="k", ls="--", lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("between-source / between-species distance")
    ax.set_title("(b) Source gap relative to species geometry")
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    for yi, value in zip(y, plot_rows["source_to_species_ratio"]):
        ax.text(value + 0.012, yi, f"{value:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, max(1.05, float(plot_rows["source_to_species_ratio"].max()) + 0.12))

    fig.tight_layout(rect=[0, 0.03, 1, 1.0])
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
