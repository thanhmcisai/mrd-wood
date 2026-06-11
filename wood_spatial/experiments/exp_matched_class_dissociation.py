#!/usr/bin/env python3
"""Compare matched-class marginal MMD with class-conditional transfer failure."""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wood_spatial.config import BB_ORDER, BASE, V4_CSV, V4_FIGURES
from wood_spatial.experiments.exp_tierc_cross_source_shift import (
    K,
    PAIRS,
    _features_by_species,
    _knn_predict,
    _resolve_species_csv,
    _shared_species,
    _species_table,
)


def _csv_dir() -> Path:
    path = V4_CSV if os.environ.get("WOOD_RESULTS_DIR") else BASE / "results" / "csv"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _figure_dir() -> Path:
    path = V4_FIGURES if os.environ.get("WOOD_RESULTS_DIR") else BASE / "results" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _find_csv(name: str) -> Path:
    candidates = (_csv_dir() / name, V4_CSV / name, BASE / "results" / "csv" / name)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Required result not found: {name}")


def _selected_species_by_seed(matched: pd.DataFrame) -> dict[int, list[str]]:
    required = {"seed", "selected_species"}
    if not required.issubset(matched.columns):
        raise ValueError(f"Matched-class CSV must contain {sorted(required)}")
    out: dict[int, list[str]] = {}
    for seed, group in matched.groupby("seed"):
        values = group["selected_species"].dropna().unique()
        if len(values) != 1:
            raise ValueError(f"Seed {seed} has {len(values)} selected-species definitions.")
        species = str(values[0]).split("|")
        if len(species) != 4:
            raise ValueError(f"Seed {seed} selects {len(species)} species, expected 4.")
        out[int(seed)] = species
    return out


def _transfer_accuracy(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    species: list[str],
) -> float:
    missing = [name for name in species if name not in source or name not in target]
    if missing:
        raise RuntimeError(f"Selected species missing from cache: {missing}")
    labels = {name: idx for idx, name in enumerate(species)}
    gallery = np.vstack([source[name] for name in species])
    query = np.vstack([target[name] for name in species])
    y_gallery = np.concatenate([
        np.full(len(source[name]), labels[name], dtype=np.int32) for name in species
    ])
    y_query = np.concatenate([
        np.full(len(target[name]), labels[name], dtype=np.int32) for name in species
    ])
    prediction = _knn_predict(gallery, y_gallery, query, k=K)
    return float(np.mean(prediction == y_query))


def run_real(species_csv: str, jobs: int) -> dict[str, pd.DataFrame]:
    table = _species_table(_resolve_species_csv(species_csv))
    matched_mmd = pd.read_csv(_find_csv("exp_mmd_class_count_matched.csv"))
    full_terms = pd.read_csv(_find_csv("exp_mmd_confound_terms.csv"))
    selected_by_seed = _selected_species_by_seed(matched_mmd)

    features: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for dataset_a, dataset_b in PAIRS:
        for backbone in BB_ORDER:
            for dataset in (dataset_a, dataset_b):
                features[(backbone, dataset)] = _features_by_species(
                    dataset, backbone, table
                )

    matched_lookup = matched_mmd.set_index(["seed", "direction", "backbone"])
    full_lookup = full_terms.set_index(["pair", "direction", "backbone"])
    tasks = []
    first_seed = min(selected_by_seed)
    for seed, large_species in sorted(selected_by_seed.items()):
        for pair_index, (dataset_a, dataset_b) in enumerate(PAIRS):
            if pair_index == 1 and seed != first_seed:
                continue
            pair = f"{dataset_a}<->{dataset_b}"
            species = (
                large_species
                if pair == "BFS46<->FSDM41"
                else _shared_species(table, dataset_a, dataset_b)
            )
            if len(species) != 4:
                raise RuntimeError(f"{pair} has {len(species)} matched species, expected 4.")
            for backbone in BB_ORDER:
                for source, target in ((dataset_a, dataset_b), (dataset_b, dataset_a)):
                    tasks.append((seed, pair, source, target, backbone, species))

    def compute(task: tuple) -> dict:
        seed, pair, source, target, backbone, species = task
        direction = f"{source}->{target}"
        accuracy = _transfer_accuracy(
            features[(backbone, source)],
            features[(backbone, target)],
            species,
        )
        if pair == "BFS46<->FSDM41":
            mmd2 = float(matched_lookup.loc[(seed, direction, backbone), "mmd2"])
        else:
            mmd2 = float(full_lookup.loc[(pair, direction, backbone), "mmd2"])
        return {
            "seed": seed,
            "pair": pair,
            "direction": direction,
            "backbone": backbone,
            "n_species": len(species),
            "selected_species": "|".join(species),
            "accuracy": accuracy,
            "failure": 1.0 - accuracy,
            "chance_accuracy": 1.0 / len(species),
            "mmd2": mmd2,
        }

    print(f"[parallel] matched-class transfer tasks={len(tasks)} jobs={jobs}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        rows = list(pool.map(compute, tasks))
    small_rows = [
        row for row in rows
        if row["pair"] == "DTSR14<->WOODAUTH" and row["seed"] == first_seed
    ]
    for seed in sorted(selected_by_seed):
        if seed == first_seed:
            continue
        for row in small_rows:
            rows.append({**row, "seed": seed})
    by_cell = pd.DataFrame(rows)
    for seed in sorted(selected_by_seed):
        print(f"[done] matched-class seed {seed + 1}/{len(selected_by_seed)}", flush=True)

    by_seed = (
        by_cell.groupby(["seed", "pair"], as_index=False)
        .agg(
            mean_mmd2=("mmd2", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_failure=("failure", "mean"),
            n_cells=("accuracy", "size"),
        )
    )
    summary_rows = []
    for pair, group in by_seed.groupby("pair"):
        summary_rows.append({
            "pair": pair,
            "n_species": 4,
            "n_seeds": int(group["seed"].nunique()),
            "mean_mmd2": float(group["mean_mmd2"].mean()),
            "sd_mmd2": float(group["mean_mmd2"].std(ddof=1)),
            "mean_accuracy": float(group["mean_accuracy"].mean()),
            "sd_accuracy": float(group["mean_accuracy"].std(ddof=1)),
            "accuracy_ci_low": float(group["mean_accuracy"].quantile(0.025)),
            "accuracy_ci_high": float(group["mean_accuracy"].quantile(0.975)),
            "mean_failure": float(group["mean_failure"].mean()),
            "chance_accuracy": 0.25,
        })
    summary = pd.DataFrame(summary_rows)
    return {"by_cell": by_cell, "by_seed": by_seed, "summary": summary}


def make_demo() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    rows = []
    for seed in range(20):
        for pair, mmd, accuracy in (
            ("BFS46<->FSDM41", 0.209, 0.05),
            ("DTSR14<->WOODAUTH", 0.193, 0.32),
        ):
            rows.append({
                "seed": seed,
                "pair": pair,
                "mean_mmd2": float(rng.normal(mmd, 0.02)),
                "mean_accuracy": float(np.clip(rng.normal(accuracy, 0.02), 0, 1)),
            })
    by_seed = pd.DataFrame(rows)
    by_seed["mean_failure"] = 1.0 - by_seed["mean_accuracy"]
    return {"by_cell": pd.DataFrame(), "by_seed": by_seed, "summary": pd.DataFrame()}


def make_figure(by_seed: pd.DataFrame, path: Path) -> None:
    pairs = ["BFS46<->FSDM41", "DTSR14<->WOODAUTH"]
    labels = ["BFS46/FSDM41", "DTSR14/WOODAUTH"]
    colors = ["#E45756", "#4C78A8"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for ax, column, ylabel, title in (
        (axes[0], "mean_mmd2", r"matched four-species RBF-MMD$^2$",
         "(a) Marginal shift magnitude"),
        (axes[1], "mean_accuracy", "cross-source kNN-5 accuracy",
         "(b) Class-conditional transfer"),
    ):
        sampled = by_seed.loc[by_seed["pair"].eq(pairs[0]), column].to_numpy()
        fixed = float(
            by_seed.loc[by_seed["pair"].eq(pairs[1]), column].iloc[0]
        )
        box = ax.boxplot([sampled], positions=[1], patch_artist=True, widths=0.55)
        box["boxes"][0].set_facecolor(colors[0])
        box["boxes"][0].set_alpha(0.6)
        jitter = np.linspace(-0.08, 0.08, len(sampled))
        ax.scatter(1 + jitter, sampled, s=18, alpha=0.55, color=colors[0], zorder=3)
        ax.scatter(
            [2], [fixed], marker="D", s=52, color=colors[1], zorder=4,
            label="fixed four-species pair",
        )
        ax.hlines(fixed, 1.72, 2.28, color=colors[1], linewidth=1.5)
        ax.set_xticks([1, 2], labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[1].axhline(0.25, color="#555555", linestyle="--", label="chance (4 classes)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true")
    mode.add_argument("--demo", action="store_true")
    parser.add_argument("--csv", default="all_public_datasets_standardized.csv")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-fig", action="store_true")
    args = parser.parse_args()

    tables = run_real(args.csv, args.jobs) if args.real else make_demo()
    if args.real:
        print("\n=== Matched-class MMD/failure dissociation ===")
        print(tables["summary"].round(4).to_string(index=False))
        if not args.no_save:
            out = _csv_dir()
            tables["by_cell"].to_csv(
                out / "exp_matched_class_dissociation_by_cell.csv", index=False
            )
            tables["by_seed"].to_csv(
                out / "exp_matched_class_dissociation_by_seed.csv", index=False
            )
            tables["summary"].to_csv(
                out / "exp_matched_class_dissociation_summary.csv", index=False
            )
            print(f"\nSaved CSV outputs to {out}")
            if not args.no_fig:
                figure = _figure_dir() / "matched_class_dissociation.png"
                make_figure(tables["by_seed"], figure)
                print(f"Saved figure to {figure}")


if __name__ == "__main__":
    main()
