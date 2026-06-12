#!/usr/bin/env python3
"""Audit the MMD mechanism sign and the Tier-C shared-class-count confound."""
from __future__ import annotations

import argparse
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wood_spatial.config import BB_ORDER
from wood_spatial.result_io import csv_dir, figure_dir, require_csv, write_provenance
from wood_spatial.experiments.exp_monitor_on_real_shift import _cap, _median_gamma
from wood_spatial.experiments.exp_tierc_cross_source_shift import (
    PAIRS,
    _features_by_species,
    _resolve_species_csv,
    _shared_species,
    _species_table,
)


def _output_dir() -> Path:
    return csv_dir()


def _find_csv(name: str) -> Path:
    return require_csv(name)


def _figure_dir() -> Path:
    return figure_dir()


def _stack(by_species: dict[str, np.ndarray], species: list[str]) -> np.ndarray:
    present = [name for name in species if name in by_species]
    if not present:
        raise RuntimeError("No selected species are present in the feature cache.")
    return np.vstack([by_species[name] for name in present])


def _stable_seed(*parts: object) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8"))


def _sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)


def _mmd_terms(
    reference: np.ndarray,
    target: np.ndarray,
    cap: int,
    seed: int,
) -> dict[str, float]:
    reference = _cap(np.asarray(reference, dtype=np.float32), cap, seed + 11)
    target = _cap(np.asarray(target, dtype=np.float32), cap, seed + 29)
    gamma = _median_gamma(reference, target, max_samples=cap)
    d_aa = _sq_dist(reference, reference)
    d_bb = _sq_dist(target, target)
    d_ab = _sq_dist(reference, target)
    k_aa = float(np.exp(-gamma * d_aa).mean())
    k_bb = float(np.exp(-gamma * d_bb).mean())
    k_ab = float(np.exp(-gamma * d_ab).mean())
    mmd2 = max(k_aa + k_bb - 2.0 * k_ab, 0.0)

    if len(target) > 1:
        upper = np.triu_indices(len(target), k=1)
        target_gram = target @ target.T
        within_batch_spread = float(np.mean(1.0 - target_gram[upper]))
    else:
        within_batch_spread = np.nan
    return {
        "gamma": float(gamma),
        "K_AA": k_aa,
        "K_BB": k_bb,
        "K_AB": k_ab,
        "mmd2": float(mmd2),
        "within_batch_spread": within_batch_spread,
        "n_reference_used": int(len(reference)),
        "n_target_used": int(len(target)),
    }


def _regression_summary(df: pd.DataFrame, scope: str) -> dict[str, float | str | int]:
    required = df[["mmd2", "within_batch_spread", "failure"]].dropna()
    if len(required) < 4:
        return {"scope": scope, "n": int(len(required))}
    y = required["failure"].to_numpy(dtype=float)
    x = required[["mmd2", "within_batch_spread"]].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = design @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / max(ss_tot, 1e-12)

    x_std = (x - x.mean(axis=0)) / np.clip(x.std(axis=0), 1e-12, None)
    beta_std = np.linalg.lstsq(
        np.column_stack([np.ones(len(x_std)), x_std]), y, rcond=None
    )[0]
    return {
        "scope": scope,
        "n": int(len(required)),
        "intercept": float(beta[0]),
        "beta_mmd2": float(beta[1]),
        "beta_spread": float(beta[2]),
        "standardized_beta_mmd2": float(beta_std[1]),
        "standardized_beta_spread": float(beta_std[2]),
        "r2_mmd_plus_spread": float(r2),
    }


def _severity_regression_from_saved_csv() -> dict[str, float | str | int]:
    path = _find_csv("exp_monitor_severity_dissociation_by_condition.csv")
    if not path.exists():
        return {"scope": "eight_condition_groups", "n": 0}
    df = pd.read_csv(path)
    if "within_batch_spread" not in df.columns and "compactness" in df.columns:
        df = df.rename(columns={"compactness": "within_batch_spread"})
    return _regression_summary(
        df.rename(columns={"mmd": "mmd2"}), "eight_condition_groups"
    )


def run_real(
    species_csv: str,
    cap: int,
    seeds: int,
    jobs: int,
) -> dict[str, pd.DataFrame]:
    table = _species_table(_resolve_species_csv(species_csv))
    transfer = pd.read_csv(_find_csv("exp_tierc_cross_source_transfer.csv"))
    transfer_key = transfer.set_index(["pair", "direction", "backbone"])

    features: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for dataset_a, dataset_b in PAIRS:
        for backbone in BB_ORDER:
            for dataset in (dataset_a, dataset_b):
                features[(backbone, dataset)] = _features_by_species(dataset, backbone, table)

    term_tasks = []
    for dataset_a, dataset_b in PAIRS:
        pair = f"{dataset_a}<->{dataset_b}"
        species = _shared_species(table, dataset_a, dataset_b)
        for backbone in BB_ORDER:
            for source, target in ((dataset_a, dataset_b), (dataset_b, dataset_a)):
                term_tasks.append((pair, species, backbone, source, target))

    def compute_terms(task):
        pair, species, backbone, source, target = task
        key = (pair, f"{source}->{target}", backbone)
        if key not in transfer_key.index:
            raise RuntimeError(f"Missing Tier-C transfer result for {key}.")
        terms = _mmd_terms(
            _stack(features[(backbone, source)], species),
            _stack(features[(backbone, target)], species),
            cap,
            _stable_seed(*key),
        )
        accuracy = float(transfer_key.loc[key, "cross_source_accuracy"])
        return {
            "pair": pair,
            "direction": f"{source}->{target}",
            "backbone": backbone,
            "n_shared_species": len(species),
            "cross_source_accuracy": accuracy,
            "failure": 1.0 - accuracy,
            **terms,
        }

    print(f"[parallel] MMD decomposition tasks={len(term_tasks)} jobs={jobs}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        term_rows = list(pool.map(compute_terms, term_tasks))
    for row in term_rows:
        print(
            f"[terms] {row['pair']} {row['direction']} {row['backbone']}: "
            f"MMD2={row['mmd2']:.4f} K_BB={row['K_BB']:.4f} "
            f"spread={row['within_batch_spread']:.4f}",
            flush=True,
        )
    terms_df = pd.DataFrame(term_rows)

    big_a, big_b = PAIRS[0]
    small_a, small_b = PAIRS[1]
    big_pair = f"{big_a}<->{big_b}"
    small_pair = f"{small_a}<->{small_b}"
    big_species = _shared_species(table, big_a, big_b)
    small_species = _shared_species(table, small_a, small_b)
    match_n = len(small_species)
    if len(big_species) < match_n:
        raise RuntimeError("The large Tier-C pair has fewer species than the control pair.")

    matched_tasks = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        selected = sorted(rng.choice(big_species, size=match_n, replace=False).tolist())
        for backbone in BB_ORDER:
            for source, target in ((big_a, big_b), (big_b, big_a)):
                matched_tasks.append((seed, selected, backbone, source, target))

    def compute_matched(task):
        seed, selected, backbone, source, target = task
        terms = _mmd_terms(
            _stack(features[(backbone, source)], selected),
            _stack(features[(backbone, target)], selected),
            cap,
            seed * 1000 + BB_ORDER.index(backbone) * 10 + int(source == big_b),
        )
        return {
            "seed": seed,
            "pair": big_pair,
            "direction": f"{source}->{target}",
            "backbone": backbone,
            "n_shared_species": match_n,
            "selected_species": "|".join(selected),
            **terms,
        }

    print(f"[parallel] class-match tasks={len(matched_tasks)} jobs={jobs}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        matched_rows = list(pool.map(compute_matched, matched_tasks))
    for seed in range(seeds):
        print(f"[class match] seed {seed + 1}/{seeds} complete", flush=True)
    matched_df = pd.DataFrame(matched_rows)

    small_mean = float(terms_df.loc[terms_df["pair"].eq(small_pair), "mmd2"].mean())
    raw_big_mean = float(terms_df.loc[terms_df["pair"].eq(big_pair), "mmd2"].mean())
    seed_means = matched_df.groupby("seed")["mmd2"].mean()
    class_summary = pd.DataFrame([{
        "large_pair": big_pair,
        "small_pair": small_pair,
        "large_pair_raw_mmd2": raw_big_mean,
        "small_pair_raw_mmd2": small_mean,
        "large_pair_matched_4class_mmd2_mean": float(seed_means.mean()),
        "large_pair_matched_4class_mmd2_std": float(seed_means.std(ddof=1)),
        "large_pair_matched_4class_mmd2_ci_low": float(seed_means.quantile(0.025)),
        "large_pair_matched_4class_mmd2_ci_high": float(seed_means.quantile(0.975)),
        "fraction_seeds_inversion_survives": float(np.mean(seed_means < small_mean)),
        "n_seeds": int(seeds),
        "matched_species_count": int(match_n),
    }])

    regression = pd.DataFrame([
        _regression_summary(terms_df, "tier_c_direction_backbone_cells"),
        _severity_regression_from_saved_csv(),
    ])
    return {
        "terms": terms_df,
        "regression": regression,
        "matched": matched_df,
        "class_summary": class_summary,
    }


def make_demo() -> dict[str, pd.DataFrame]:
    conditions = pd.DataFrame({
        "mmd2": [0.03, 0.07, 0.11, 0.14, 0.20, 0.10],
        "within_batch_spread": [0.50, 0.53, 0.50, 0.50, 0.47, 0.52],
        "failure": [0.00, 0.42, 0.75, 0.83, 0.68, 0.99],
    })
    return {
        "terms": conditions,
        "regression": pd.DataFrame([_regression_summary(conditions, "demo")]),
        "matched": pd.DataFrame(),
        "class_summary": pd.DataFrame([{
            "large_pair_raw_mmd2": 0.10,
            "small_pair_raw_mmd2": 0.20,
            "large_pair_matched_4class_mmd2_mean": 0.12,
            "fraction_seeds_inversion_survives": 1.0,
        }]),
    }


def make_figure(tables: dict[str, pd.DataFrame], path: Path) -> None:
    terms = tables["terms"]
    matched = tables["matched"]
    class_summary = tables["class_summary"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    if {"pair", "K_AA", "K_BB", "K_AB"}.issubset(terms.columns):
        means = terms.groupby("pair")[["K_AA", "K_BB", "K_AB"]].mean()
        x = np.arange(len(means))
        width = 0.24
        for offset, column, label, color in (
            (-width, "K_AA", r"$K_{AA}$", "#4C78A8"),
            (0.0, "K_BB", r"$K_{BB}$", "#E45756"),
            (width, "K_AB", r"$K_{AB}$", "#72B7B2"),
        ):
            ax.bar(x + offset, means[column], width, label=label, color=color)
        ax.set_xticks(x, [p.replace("<->", "\n<->\n") for p in means.index])
        ax.set_ylabel("mean RBF-kernel term")
        ax.legend(frameon=False, ncol=3)
    else:
        ax.text(0.5, 0.5, "Run --real for kernel-term decomposition",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_title("(a) MMD decomposition by cross-source pair")

    ax = axes[1]
    if not matched.empty:
        seed_means = matched.groupby("seed")["mmd2"].mean().to_numpy()
        ax.boxplot(
            seed_means,
            positions=[0],
            widths=0.45,
            patch_artist=True,
            boxprops={"facecolor": "#E45756", "alpha": 0.65},
            medianprops={"color": "black"},
        )
        ax.scatter(
            np.zeros(len(seed_means)),
            seed_means,
            s=18,
            alpha=0.55,
            color="#B22222",
            zorder=3,
        )
        ax.axhline(
            float(class_summary["small_pair_raw_mmd2"]),
            color="#4C78A8",
            linestyle="--",
            label="4-species DTSR14/WOODAUTH",
        )
        ax.axhline(
            float(class_summary["large_pair_raw_mmd2"]),
            color="#555555",
            linestyle=":",
            label="raw 24-species baseline",
        )
        ax.axhline(
            float(class_summary["large_pair_matched_4class_mmd2_mean"]),
            color="#B22222",
            linestyle="-.",
            label="matched 4-species mean",
        )
        ax.set_xticks([0], ["BFS46/FSDM41\nmatched to 4 species"])
        ax.set_ylabel(r"RBF-MMD$^2$")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    else:
        ax.text(0.5, 0.5, "Run --real for matched-class control",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_title("(b) Shared-class-count sensitivity")

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
    parser.add_argument("--cap", type=int, default=512)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Parallel MMD computations. Use 2 on Colab L4 to limit RAM pressure.",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-fig", action="store_true")
    args = parser.parse_args()

    tables = (
        run_real(args.csv, args.cap, args.seeds, args.jobs)
        if args.real
        else make_demo()
    )
    print("\n=== MMD decomposition/regression ===")
    print(tables["regression"].round(4).to_string(index=False))
    print("\n=== Shared-class-count control ===")
    print(tables["class_summary"].round(4).to_string(index=False))

    if args.real and not args.no_save:
        out = _output_dir()
        tables["terms"].to_csv(out / "exp_mmd_confound_terms.csv", index=False)
        tables["regression"].to_csv(out / "exp_mmd_confound_regression.csv", index=False)
        tables["matched"].to_csv(out / "exp_mmd_class_count_matched.csv", index=False)
        tables["class_summary"].to_csv(out / "exp_mmd_confound_summary.csv", index=False)
        print(f"\nSaved CSV outputs to {out}")
        if not args.no_fig:
            fig_path = _figure_dir() / "mmd_confound_and_class_count.png"
            make_figure(tables, fig_path)
            print(f"Saved figure to {fig_path}")
        outputs = [
            out / "exp_mmd_confound_terms.csv",
            out / "exp_mmd_confound_regression.csv",
            out / "exp_mmd_class_count_matched.csv",
            out / "exp_mmd_confound_summary.csv",
        ]
        if not args.no_fig:
            outputs.extend([fig_path, fig_path.with_suffix(".pdf")])
        write_provenance(
            "exp_mmd_confound_and_sign",
            outputs,
            protocol="tier_c_mmd_decomposition_v1",
            parameters={
                "cap": args.cap,
                "seeds": args.seeds,
                "jobs": args.jobs,
                "seed_policy": "crc32_stable",
            },
            inputs=[require_csv("exp_tierc_cross_source_transfer.csv")],
        )


if __name__ == "__main__":
    main()
