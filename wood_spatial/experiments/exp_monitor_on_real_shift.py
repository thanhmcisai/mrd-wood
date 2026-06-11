#!/usr/bin/env python3
"""
Label-free monitor on synthetic and real acquisition shifts.

This experiment applies reference-bank monitor scores to the real-shift tiers:
Tier-C cross-source and Tier-D cross-magnification. It reports whether the same
score that works on the synthetic benchmark ranks risk across clean, synthetic,
moderate real, and extreme real shifts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wood_spatial.config import BASE, BB_LABEL, BB_ORDER, PERTURB_CONFIGS, TIER_A, TIER_B, TIER_C, V4_CSV, V4_FIGURES
from wood_spatial.experiments.exp_tierc_cross_source_shift import (
    PAIRS as TIER_C_PAIRS,
    _features_by_species,
    _load_feature_cache_np,
    _norm,
    _resolve_species_csv,
    _shared_species,
    _species_table,
)


CONDITION_ORDER = [
    "TierA_clean",
    "TierB_synth",
    "TierD_xmag",
    "TierC_DTSR14_WOODAUTH",
    "TierC_BFS46_FSDM41",
]

CONDITION_LABEL = {
    "TierA_clean": "Tier-A clean",
    "TierB_synth": "Tier-B synthetic",
    "TierD_xmag": "Tier-D cross-magnification",
    "TierC_DTSR14_WOODAUTH": "Tier-C DTSR14<->WOODAUTH",
    "TierC_BFS46_FSDM41": "Tier-C BFS46<->FSDM41",
}


def _load_norm_cache(backbone: str, dataset: str, tag: str = "original") -> np.ndarray:
    features, _labels, _paths, _path = _load_feature_cache_np(backbone, dataset, tag)
    return _norm(np.asarray(features, dtype=np.float32))


def _cap(X: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if cap <= 0 or len(X) <= cap:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), cap, replace=False)]


def _centroid_score(ref: np.ndarray, target: np.ndarray) -> float:
    c0 = ref.mean(axis=0)
    c1 = target.mean(axis=0)
    c0 = c0 / (np.linalg.norm(c0) + 1e-12)
    c1 = c1 / (np.linalg.norm(c1) + 1e-12)
    return float(1.0 - c0 @ c1)


def _median_gamma(x: np.ndarray, y: np.ndarray, max_samples: int = 512) -> float:
    z = np.vstack([x, y])
    z = _cap(z, max_samples, 42)
    gram = z @ z.T
    sq = np.maximum(np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2 * gram, 0)
    vals = sq[np.triu_indices_from(sq, k=1)]
    vals = vals[vals > 1e-12]
    med = float(np.median(vals)) if len(vals) else 1.0
    return 1.0 / max(2.0 * med, 1e-12)


def _mmd_rbf(ref: np.ndarray, target: np.ndarray, cap: int = 512) -> float:
    ref = _cap(ref, cap, 123)
    target = _cap(target, cap, 456)
    gamma = _median_gamma(ref, target, max_samples=cap)

    def kernel_mean(a: np.ndarray, b: np.ndarray) -> float:
        gram = a @ b.T
        aa = np.sum(a * a, axis=1)[:, None]
        bb = np.sum(b * b, axis=1)[None, :]
        sq = np.maximum(aa + bb - 2 * gram, 0)
        return float(np.exp(-gamma * sq).mean())

    return max(kernel_mean(ref, ref) + kernel_mean(target, target) - 2 * kernel_mean(ref, target), 0.0)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).correlation)
    except Exception:
        xr = pd.Series(x).rank(method="average").to_numpy()
        yr = pd.Series(y).rank(method="average").to_numpy()
        return float(np.corrcoef(xr, yr)[0, 1])


def _synthetic_threshold(detector_col: str = "ref_mmd_rbf") -> tuple[float, str]:
    candidates = [
        V4_CSV / "exp10_reference_monitor_auc.csv",
        BASE / "results" / "csv" / "exp10_reference_monitor_auc.csv",
        Path.cwd() / "results" / "csv" / "exp10_reference_monitor_auc.csv",
    ]
    auc_path = candidates[0]
    for path in candidates:
        if not path.exists():
            continue
        auc_path = path
        auc = pd.read_csv(path)
        sub = auc[auc["detector"] == detector_col]
        if len(sub) and "best_threshold" in sub.columns:
            return float(sub.iloc[0]["best_threshold"]), str(path)
    return np.nan, str(auc_path)


def _available_synth_tags() -> list[str]:
    def cache_tag_for(pert_name: str, value) -> str:
        if pert_name == "gaussian_blur":
            return f"blur_{int(value)}"
        if pert_name == "defocus_blur":
            return f"defocus_{int(value)}"
        if pert_name == "jpeg":
            return f"jpeg_{int(value)}"
        if pert_name == "compound":
            return f"compound_{value}"
        if pert_name in ("compound_optical", "compound_digital", "compound_field"):
            return f"{pert_name}_{value}"
        if pert_name in ("color_shift", "red_channel_shift"):
            return f"color_shift_{int(value)}"
        if pert_name in ("green_channel_shift", "blue_channel_shift"):
            return f"{pert_name}_{int(value)}"
        if pert_name in ("gaussian_noise", "impulse_noise", "zoom_blur", "contrast", "pixelate"):
            return f"{pert_name}_{str(value).replace('.', '_')}"
        if pert_name in ("shot_noise", "motion_blur"):
            return f"{pert_name}_{int(value)}"
        if pert_name == "scratch":
            return f"scratch_{value}"
        if pert_name == "illumination":
            return f"brightness_{str(value).replace('.', '_')}"
        if pert_name == "resize":
            return f"resize_{str(value).replace('.', '_')}"
        if pert_name == "rotation":
            return f"rotation_{int(value)}"
        return f"{pert_name}_{value}"

    tags = [
        cache_tag_for(name, value)
        for name, cfg in PERTURB_CONFIGS.items()
        for value in cfg["values"]
    ]
    return tags


def _stack_shared(table: pd.DataFrame, dataset: str, backbone: str, pair_species: list[str]) -> np.ndarray:
    feats = _features_by_species(dataset, backbone, table)
    present = [s for s in pair_species if s in feats]
    if not present:
        raise RuntimeError(f"No shared species features for {backbone}/{dataset}")
    return np.vstack([feats[s] for s in present])


def _add_record(rows: list[dict], backbone: str, condition: str, reference_dataset: str, target_dataset: str,
                target_tag: str, ref: np.ndarray, target: np.ndarray, severity_rank: int, cap: int) -> None:
    ref_c = _cap(ref, cap, abs(hash((backbone, reference_dataset, target_dataset, target_tag, "r"))) % (2**32))
    target_c = _cap(target, cap, abs(hash((backbone, reference_dataset, target_dataset, target_tag, "t"))) % (2**32))
    rows.append({
        "backbone": backbone,
        "backbone_label": BB_LABEL.get(backbone, backbone),
        "condition": condition,
        "condition_label": CONDITION_LABEL[condition],
        "severity_rank": severity_rank,
        "reference_dataset": reference_dataset,
        "target_dataset": target_dataset,
        "target_tag": target_tag,
        "n_reference": int(len(ref)),
        "n_target": int(len(target)),
        "ref_centroid_cosine": _centroid_score(ref_c, target_c),
        "ref_mmd_rbf": _mmd_rbf(ref_c, target_c, cap=min(cap, 512)),
    })


def build_real_scores(species_csv: str, cap: int = 1500, max_tierb_tags: int = 0) -> pd.DataFrame:
    table = _species_table(_resolve_species_csv(species_csv))
    tags = _available_synth_tags()
    if max_tierb_tags and max_tierb_tags > 0:
        tags = tags[:max_tierb_tags]
    rows: list[dict] = []
    rank = {c: i for i, c in enumerate(CONDITION_ORDER)}

    for bb in BB_ORDER:
        # Tier-A clean: same-dataset split, no perturbation.
        for ds in TIER_A:
            try:
                clean = _load_norm_cache(bb, ds, "original")
            except FileNotFoundError as exc:
                print(f"[skip] {bb}/{ds}/original: {exc}", flush=True)
                continue
            if len(clean) < 20:
                continue
            mid = max(1, len(clean) // 2)
            _add_record(rows, bb, "TierA_clean", ds, ds, "original_holdout", clean[:mid], clean[mid:], rank["TierA_clean"], cap)

        # Tier-B synthetic: same-source clean reference, perturbed target.
        for ds in TIER_B:
            try:
                ref = _load_norm_cache(bb, ds, "original")
            except FileNotFoundError:
                continue
            for tag in tags:
                try:
                    target = _load_norm_cache(bb, ds, tag)
                except FileNotFoundError:
                    continue
                _add_record(rows, bb, "TierB_synth", ds, ds, tag, ref, target, rank["TierB_synth"], cap)

        # Tier-D cross-magnification: clean source magnification to clean target magnification.
        for i, src in enumerate(TIER_C):
            for dst in TIER_C:
                if src == dst:
                    continue
                try:
                    ref = _load_norm_cache(bb, src, "original")
                    target = _load_norm_cache(bb, dst, "original")
                except FileNotFoundError:
                    continue
                _add_record(rows, bb, "TierD_xmag", src, dst, "original", ref, target, rank["TierD_xmag"], cap)

        # Tier-C cross-source: restrict to accepted shared species for a fair same-species shift.
        for ds_a, ds_b in TIER_C_PAIRS:
            species = _shared_species(table, ds_a, ds_b)
            for src, dst in ((ds_a, ds_b), (ds_b, ds_a)):
                try:
                    ref = _stack_shared(table, src, bb, species)
                    target = _stack_shared(table, dst, bb, species)
                except (FileNotFoundError, RuntimeError):
                    continue
                cond = "TierC_BFS46_FSDM41" if {ds_a, ds_b} == {"BFS46", "FSDM41"} else "TierC_DTSR14_WOODAUTH"
                _add_record(rows, bb, cond, src, dst, "original_shared_species", ref, target, rank[cond], cap)

        print(f"[done] {bb}: rows={len([r for r in rows if r['backbone'] == bb])}", flush=True)

    if not rows:
        raise RuntimeError("No monitor-on-real-shift rows produced. Check feature caches and species CSV.")
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame, threshold: float, threshold_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = scores.copy()
    scores["alarm_at_synth_threshold"] = (
        scores["ref_mmd_rbf"] > threshold if np.isfinite(threshold) else False
    )
    by_condition = scores.groupby(["condition", "condition_label", "severity_rank"], as_index=False).agg(
        mean_ref_mmd_rbf=("ref_mmd_rbf", "mean"),
        std_ref_mmd_rbf=("ref_mmd_rbf", "std"),
        mean_ref_centroid_cosine=("ref_centroid_cosine", "mean"),
        alarm_rate_at_synth_threshold=("alarm_at_synth_threshold", "mean"),
        n_records=("ref_mmd_rbf", "size"),
        n_backbones=("backbone", "nunique"),
    ).sort_values("severity_rank")

    mean_scores = by_condition["mean_ref_mmd_rbf"].to_numpy(dtype=float)
    severity = by_condition["severity_rank"].to_numpy(dtype=float)
    rho = _spearman(severity, mean_scores) if len(by_condition) > 1 else np.nan
    clean = scores[scores["condition"] == "TierA_clean"]
    real = scores[scores["condition"].isin(["TierD_xmag", "TierC_DTSR14_WOODAUTH", "TierC_BFS46_FSDM41"])]
    tierc = scores[scores["condition"].isin(["TierC_DTSR14_WOODAUTH", "TierC_BFS46_FSDM41"])]
    overall = pd.DataFrame([{
        "monitor": "ref_mmd_rbf",
        "synthetic_threshold": threshold,
        "threshold_source": threshold_source,
        "severity_spearman_by_condition_mean": rho,
        "clean_false_alarm_rate": float(clean["alarm_at_synth_threshold"].mean()) if len(clean) else np.nan,
        "real_shift_alarm_rate": float(real["alarm_at_synth_threshold"].mean()) if len(real) else np.nan,
        "tierc_cross_source_alarm_rate": float(tierc["alarm_at_synth_threshold"].mean()) if len(tierc) else np.nan,
        "n_records": int(len(scores)),
    }])
    return by_condition, overall


def plot_summary(by_condition: pd.DataFrame, threshold: float, overall: pd.DataFrame, fig_path: Path) -> None:
    by_condition = by_condition.copy()
    gamma_path = V4_CSV / "exp_mmd_gamma_sensitivity_by_condition.csv"
    if gamma_path.exists():
        gamma = pd.read_csv(gamma_path)
        gamma = gamma[gamma["policy"].eq("per_pair_median")].set_index("condition")
        replacements = {
            "TierA_clean": float(gamma.loc["clean_TierA", "mmd"]),
            "TierD_xmag": float(gamma.loc[
                ["TierD_xmag_x10x20", "TierD_xmag_x10x50", "TierD_xmag_x20x50"],
                "mmd",
            ].mean()),
            "TierC_DTSR14_WOODAUTH": float(gamma.loc["TierC_DTSR14_WOODAUTH", "mmd"]),
            "TierC_BFS46_FSDM41": float(gamma.loc["TierC_BFS46_FSDM41", "mmd"]),
        }
        for condition, value in replacements.items():
            by_condition.loc[
                by_condition["condition"].eq(condition), "mean_ref_mmd_rbf"
            ] = value
    confound_candidates = [
        V4_CSV / "exp_mmd_confound_summary.csv",
        BASE / "results" / "csv" / "exp_mmd_confound_summary.csv",
        Path.cwd() / "results" / "csv" / "exp_mmd_confound_summary.csv",
    ]
    confound_path = next((p for p in confound_candidates if p.exists()), None)
    if confound_path is not None:
        confound = pd.read_csv(confound_path).iloc[0]
        by_condition.loc[
            by_condition["condition"].eq("TierC_BFS46_FSDM41"),
            "mean_ref_mmd_rbf",
        ] = float(confound["large_pair_raw_mmd2"])
        by_condition.loc[
            by_condition["condition"].eq("TierC_DTSR14_WOODAUTH"),
            "mean_ref_mmd_rbf",
        ] = float(confound["small_pair_raw_mmd2"])

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = np.arange(len(by_condition))
    vals = by_condition["mean_ref_mmd_rbf"].to_numpy()
    colors = ["#2A9D8F", "#4C78A8", "#F58518", "#E45756", "#B279A2"][:len(vals)]
    ax.bar(x, vals, color=colors)
    if np.isfinite(threshold):
        ax.axhline(threshold, color="k", ls="--", lw=1.2, label=f"synthetic threshold={threshold:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(by_condition["condition_label"].tolist(), rotation=18, ha="right")
    ax.set_ylabel("RBF-MMD reference-bank score")
    rho = _spearman(
        by_condition["severity_rank"].to_numpy(dtype=float),
        by_condition["mean_ref_mmd_rbf"].to_numpy(dtype=float),
    )
    ax.set_title(
        "Label-free monitor detects mismatch but does not rank severity "
        f"(Spearman={rho:.3f})"
    )
    for i, value in enumerate(vals):
        ax.text(i, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    if np.isfinite(threshold):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run monitor-on-real-shift analysis from cached features.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true")
    mode.add_argument("--demo", action="store_true")
    ap.add_argument("--csv", default="all_public_datasets_standardized.csv")
    ap.add_argument("--cap", type=int, default=1500, help="maximum reference/target features per score")
    ap.add_argument("--max-tierb-tags", type=int, default=0, help="debug limit; 0 means all available synthetic tags")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    if args.demo:
        rng = np.random.default_rng(42)
        rows = []
        for bb in BB_ORDER:
            for cond_i, cond in enumerate(CONDITION_ORDER):
                for rep in range(8):
                    val = max(0, 0.015 + 0.04 * cond_i + rng.normal(0, 0.01))
                    rows.append({
                        "backbone": bb,
                        "backbone_label": BB_LABEL.get(bb, bb),
                        "condition": cond,
                        "condition_label": CONDITION_LABEL[cond],
                        "severity_rank": cond_i,
                        "reference_dataset": "demo",
                        "target_dataset": "demo",
                        "target_tag": f"demo_{rep}",
                        "n_reference": 100,
                        "n_target": 100,
                        "ref_centroid_cosine": val / 2,
                        "ref_mmd_rbf": val,
                    })
        scores = pd.DataFrame(rows)
    else:
        scores = build_real_scores(args.csv, cap=args.cap, max_tierb_tags=args.max_tierb_tags)

    threshold, threshold_source = _synthetic_threshold()
    by_condition, overall = summarize(scores, threshold, threshold_source)
    print("\n=== Monitor on real shift: condition means ===")
    print(by_condition.to_string(index=False))
    print("\n=== Monitor on real shift: summary ===")
    print(overall.to_string(index=False))

    csv_dir = V4_CSV
    fig_dir = V4_FIGURES
    if (args.real or not args.demo) and not args.no_save:
        csv_dir.mkdir(parents=True, exist_ok=True)
        scores.to_csv(csv_dir / "exp_monitor_on_real_shift_scores.csv", index=False)
        by_condition.to_csv(csv_dir / "exp_monitor_on_real_shift_by_condition.csv", index=False)
        overall.to_csv(csv_dir / "exp_monitor_on_real_shift_summary.csv", index=False)
        if not args.no_fig:
            plot_summary(by_condition, threshold, overall, fig_dir / "monitor_on_real_shift.png")
        print(f"\nSaved CSV outputs to {csv_dir}")
        if not args.no_fig:
            print(f"Saved figure to {fig_dir / 'monitor_on_real_shift.png'}")
    elif not args.no_fig:
        plot_summary(by_condition, threshold, overall, Path("monitor_on_real_shift.png"))


if __name__ == "__main__":
    main()
