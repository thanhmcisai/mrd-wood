#!/usr/bin/env python3
"""
Cross-magnification asymmetry analysis for VN26.

This lightweight analysis uses the full 3x3 VN26 cross-magnification accuracy
matrix and, in --real mode, the cached clean VN26 features to compute class-centroid
feature drift between magnification levels. It is CPU-only once feature caches exist.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wood_spatial.config import BB_LABEL, BB_ORDER, TIER_C
from wood_spatial.result_io import csv_dir, figure_dir, require_csv, write_provenance

LOGGER = logging.getLogger(__name__)

MAG_VALUE = {"VN26_x10": 10, "VN26_x20": 20, "VN26_x50": 50}
MAG_LABEL = {"VN26_x10": "x10", "VN26_x20": "x20", "VN26_x50": "x50"}


def _norm(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def _load_accuracy(csv_dir: Path) -> pd.DataFrame:
    path = _find_csv("exp5_full_crossmag_accuracy.csv", csv_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run exp5_full_crossmag before this analysis."
        )
    return pd.read_csv(path)


def _find_csv(name: str, preferred_dir: Path) -> Path:
    del preferred_dir
    return require_csv(name)


def _output_dirs() -> tuple[Path, Path]:
    return csv_dir(), figure_dir()


def _load_saved_tables(csv_path: Path) -> dict[str, pd.DataFrame]:
    names = {
        "by_backbone": "exp5_crossmag_asymmetry_by_backbone.csv",
        "by_pair": "exp5_crossmag_asymmetry_by_pair.csv",
        "by_ratio": "exp5_crossmag_ratio_summary.csv",
        "drift_drop": "exp5_crossmag_drift_drop.csv",
        "summary": "exp5_crossmag_asymmetry_summary.csv",
    }
    return {
        key: pd.read_csv(csv_path / name)
        for key, name in names.items()
    }


def _centroids(features: np.ndarray, labels: np.ndarray) -> dict:
    z = _norm(np.asarray(features, dtype=np.float32))
    labels = np.asarray(labels)
    out = {}
    for lab in sorted(set(labels.tolist())):
        idx = labels == lab
        if np.any(idx):
            c = z[idx].mean(axis=0, keepdims=True)
            out[lab] = _norm(c)[0]
    return out


def _centroid_drift(ca: dict, cb: dict) -> float:
    common = sorted(set(ca).intersection(cb))
    if not common:
        return np.nan
    a = np.stack([ca[k] for k in common])
    b = np.stack([cb[k] for k in common])
    return float(np.mean(1.0 - np.sum(a * b, axis=1)))


def _load_centroid_drifts(backbones: list[str], mags: list[str]) -> pd.DataFrame:
    from wood_spatial.core.cache import load_cache

    rows = []
    for bb in backbones:
        cache = {}
        for mag in mags:
            try:
                feats, labels, _paths = load_cache(bb, mag, "original")
            except FileNotFoundError as exc:
                LOGGER.warning("Missing feature cache for %s/%s: %s", bb, mag, exc)
                continue
            cache[mag] = _centroids(feats, labels)
        for a in mags:
            for b in mags:
                if a == b or a not in cache or b not in cache:
                    continue
                rows.append({
                    "backbone": bb,
                    "source_mag": a,
                    "target_mag": b,
                    "centroid_drift": _centroid_drift(cache[a], cache[b]),
                })
    return pd.DataFrame(rows)


def _demo_drifts(acc: pd.DataFrame) -> pd.DataFrame:
    """Deterministic shape-compatible demo drifts when feature caches are absent."""
    rows = []
    for _, row in acc[acc["train_mag"] != acc["test_mag"]].iterrows():
        a = MAG_VALUE[row["train_mag"]]
        b = MAG_VALUE[row["test_mag"]]
        ratio = max(a, b) / min(a, b)
        x50_penalty = 0.10 if (a == 50 or b == 50) else 0.0
        bb_offset = (BB_ORDER.index(row["backbone"]) + 1) * 0.003 if row["backbone"] in BB_ORDER else 0.0
        rows.append({
            "backbone": row["backbone"],
            "source_mag": row["train_mag"],
            "target_mag": row["test_mag"],
            "centroid_drift": 0.08 * np.log(ratio) + x50_penalty + bb_offset,
        })
    return pd.DataFrame(rows)


def _direction(a: str, b: str) -> str:
    va, vb = MAG_VALUE[a], MAG_VALUE[b]
    if va < vb:
        return "coarse_to_fine"
    if va > vb:
        return "fine_to_coarse"
    return "within"


def _pair_name(a: str, b: str) -> str:
    vals = sorted([a, b], key=lambda x: MAG_VALUE[x])
    return f"{MAG_LABEL[vals[0]]}<->{MAG_LABEL[vals[1]]}"


def _build_tables(acc: pd.DataFrame, drifts: pd.DataFrame, tier_csv: Path) -> dict[str, pd.DataFrame]:
    acc = acc.copy()
    acc["is_cross"] = acc["train_mag"] != acc["test_mag"]
    within = acc[~acc["is_cross"]][["backbone", "train_mag", "accuracy"]].rename(
        columns={"train_mag": "source_mag", "accuracy": "within_accuracy"}
    )
    cross = acc[acc["is_cross"]].rename(
        columns={"train_mag": "source_mag", "test_mag": "target_mag", "accuracy": "cross_accuracy"}
    )
    cross = cross.merge(within, on=["backbone", "source_mag"], how="left")
    cross["accuracy_drop"] = cross["within_accuracy"] - cross["cross_accuracy"]
    cross["direction"] = [_direction(a, b) for a, b in zip(cross["source_mag"], cross["target_mag"])]
    cross["mag_pair"] = [_pair_name(a, b) for a, b in zip(cross["source_mag"], cross["target_mag"])]
    cross = cross.merge(drifts, on=["backbone", "source_mag", "target_mag"], how="left")

    # Pairwise signed asymmetry: acc(coarse->fine) - acc(fine->coarse).
    pair_rows = []
    bb_rows = []
    for coarse, fine in [("VN26_x10", "VN26_x20"), ("VN26_x10", "VN26_x50"), ("VN26_x20", "VN26_x50")]:
        pair = _pair_name(coarse, fine)
        for bb in sorted(cross["backbone"].unique()):
            cf = cross[(cross["backbone"] == bb) & (cross["source_mag"] == coarse) & (cross["target_mag"] == fine)]
            fc = cross[(cross["backbone"] == bb) & (cross["source_mag"] == fine) & (cross["target_mag"] == coarse)]
            if cf.empty or fc.empty:
                continue
            asym = float(cf["cross_accuracy"].iloc[0] - fc["cross_accuracy"].iloc[0])
            if abs(asym) < 0.02:
                direction = "near_symmetric"
            elif asym > 0:
                direction = "coarse_to_fine_easier"
            else:
                direction = "fine_to_coarse_easier"
            bb_rows.append({
                "backbone": bb,
                "mag_pair": pair,
                "coarse_mag": coarse,
                "fine_mag": fine,
                "acc_coarse_to_fine": float(cf["cross_accuracy"].iloc[0]),
                "acc_fine_to_coarse": float(fc["cross_accuracy"].iloc[0]),
                "signed_asymmetry": asym,
                "easier_direction": direction,
                "ratio": MAG_VALUE[fine] / MAG_VALUE[coarse],
            })
        sub = pd.DataFrame([r for r in bb_rows if r["mag_pair"] == pair])
        if sub.empty:
            continue
        sign = np.sign(sub["signed_asymmetry"].to_numpy())
        nonzero = sign[sign != 0]
        modal = 0 if len(nonzero) == 0 else int(pd.Series(nonzero).mode().iloc[0])
        agree = float(np.mean(sign == modal)) if modal != 0 else float(np.mean(np.abs(sub["signed_asymmetry"]) < 0.02))
        mean_asym = float(sub["signed_asymmetry"].mean())
        rng = np.random.default_rng(42)
        asym_values = sub["signed_asymmetry"].to_numpy(dtype=float)
        bootstrap_means = asym_values[
            rng.integers(0, len(asym_values), size=(10000, len(asym_values)))
        ].mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
        if abs(mean_asym) < 0.02:
            mean_dir = "near_symmetric"
        elif mean_asym > 0:
            mean_dir = "coarse_to_fine_easier"
        else:
            mean_dir = "fine_to_coarse_easier"
        pair_rows.append({
            "mag_pair": pair,
            "coarse_mag": coarse,
            "fine_mag": fine,
            "mean_acc_coarse_to_fine": float(sub["acc_coarse_to_fine"].mean()),
            "mean_acc_fine_to_coarse": float(sub["acc_fine_to_coarse"].mean()),
            "signed_asymmetry": mean_asym,
            "asymmetry_ci_low": float(ci_low),
            "asymmetry_ci_high": float(ci_high),
            "bootstrap_unit": "backbone",
            "bootstrap_reps": 10000,
            "easier_direction": mean_dir,
            "sign_agreement_fraction": agree,
            "sign_agreement_percent": 100.0 * agree,
            "n_backbones": int(sub["backbone"].nunique()),
            "ratio": MAG_VALUE[fine] / MAG_VALUE[coarse],
        })

    by_backbone = pd.DataFrame(bb_rows)
    by_pair = pd.DataFrame(pair_rows)

    # Non-monotonicity by ratio: average bidirectional transfer per unordered pair.
    ratio_rows = []
    for pair, sub in by_backbone.groupby("mag_pair"):
        ratio_rows.append({
            "mag_pair": pair,
            "ratio": float(sub["ratio"].iloc[0]),
            "mean_bidirectional_accuracy": float(
                pd.concat([sub["acc_coarse_to_fine"], sub["acc_fine_to_coarse"]]).mean()
            ),
            "mean_abs_asymmetry": float(sub["signed_asymmetry"].abs().mean()),
        })
    by_ratio = pd.DataFrame(ratio_rows).sort_values("ratio")

    # Tier-A drift/drop line and cross-mag comparison.
    tier = pd.read_csv(tier_csv) if tier_csv.exists() else pd.DataFrame()
    if not tier.empty and "accuracy_drop" not in tier.columns and "drop" in tier.columns:
        tier = tier.rename(columns={"drop": "accuracy_drop"})
    summary_rows = []
    cross_valid = cross.dropna(subset=["centroid_drift", "accuracy_drop"])
    if len(cross_valid) >= 3:
        cross_x = cross_valid["centroid_drift"].to_numpy(dtype=float)
        cross_y = cross_valid["accuracy_drop"].to_numpy(dtype=float)
        cross_slope, cross_intercept = np.polyfit(cross_x, cross_y, 1)
        cross_r = float(np.corrcoef(cross_x, cross_y)[0, 1])
        row = {
            "tier_a_csv": str(tier_csv),
            "tier_a_available": False,
            "tier_a_slope": np.nan,
            "tier_a_intercept": np.nan,
            "crossmag_slope": float(cross_slope),
            "crossmag_intercept": float(cross_intercept),
            "crossmag_drift_drop_r": cross_r,
            "crossmag_to_tier_a_slope_ratio": np.nan,
            "crossmag_mean_abs_residual_on_tier_a_line": np.nan,
            "n_crossmag_records": int(len(cross_valid)),
            "n_crossmag_missing_drift": int(cross["centroid_drift"].isna().sum()),
        }
        if not tier.empty and {"feature_drift", "accuracy_drop"}.issubset(tier.columns):
            tier_x = tier["feature_drift"].to_numpy(dtype=float)
            tier_y = tier["accuracy_drop"].to_numpy(dtype=float)
            tier_slope, tier_intercept = np.polyfit(tier_x, tier_y, 1)
            pred = tier_slope * cross_x + tier_intercept
            row.update({
                "tier_a_available": True,
                "tier_a_slope": float(tier_slope),
                "tier_a_intercept": float(tier_intercept),
                "crossmag_to_tier_a_slope_ratio": float(cross_slope / tier_slope) if tier_slope else np.nan,
                "crossmag_mean_abs_residual_on_tier_a_line": float(np.mean(np.abs(cross_y - pred))),
            })
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    return {
        "by_backbone": by_backbone,
        "by_pair": by_pair,
        "by_ratio": by_ratio,
        "drift_drop": cross,
        "summary": summary,
    }


def _plot(tables: dict[str, pd.DataFrame], fig_path: Path):
    by_pair = tables["by_pair"]
    by_ratio = tables["by_ratio"]
    dd = tables["drift_drop"].dropna(subset=["centroid_drift", "accuracy_drop"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    colors = ["#4C78A8" if x >= 0 else "#E45756" for x in by_pair["signed_asymmetry"]]
    axes[0].bar(by_pair["mag_pair"], by_pair["signed_asymmetry"], color=colors)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_ylabel("acc(coarse->fine) - acc(fine->coarse)")
    axes[0].set_title("(a) Signed asymmetry")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].plot(by_ratio["ratio"], by_ratio["mean_bidirectional_accuracy"], marker="o", color="#4C78A8")
    for _, row in by_ratio.iterrows():
        axes[1].annotate(row["mag_pair"], (row["ratio"], row["mean_bidirectional_accuracy"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[1].set_xlabel("magnification ratio")
    axes[1].set_ylabel("mean bidirectional accuracy")
    axes[1].set_title("(b) Non-monotone scale transfer")

    if not dd.empty:
        for bb, sub in dd.groupby("backbone"):
            axes[2].scatter(sub["centroid_drift"], sub["accuracy_drop"], s=28, alpha=0.85,
                            label=BB_LABEL.get(bb, bb))
        x = dd["centroid_drift"].to_numpy()
        y = dd["accuracy_drop"].to_numpy()
        if len(dd) >= 3:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
            axes[2].plot(xs, m * xs + b, color="black", lw=1.5, label="cross-mag fit")
    axes[2].set_xlabel("class-centroid feature drift")
    axes[2].set_ylabel("accuracy drop")
    axes[2].set_title("(c) Drift tracks cross-mag drop")
    axes[2].legend(fontsize=6, ncol=1, frameon=False)

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(real: bool = True, save: bool = True) -> dict[str, pd.DataFrame]:
    csv_dir, fig_dir = _output_dirs()
    acc = _load_accuracy(csv_dir)
    mags = [m for m in TIER_C if m in set(acc["train_mag"]).union(acc["test_mag"])]
    backbones = [b for b in BB_ORDER if b in set(acc["backbone"])]
    drifts = _load_centroid_drifts(backbones, mags) if real else _demo_drifts(acc)
    tier_csv = _find_csv("exp1b_feature_geometry.csv", csv_dir)
    tables = _build_tables(acc, drifts, tier_csv)

    if save:
        csv_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        tables["by_backbone"].to_csv(csv_dir / "exp5_crossmag_asymmetry_by_backbone.csv", index=False)
        tables["by_pair"].to_csv(csv_dir / "exp5_crossmag_asymmetry_by_pair.csv", index=False)
        tables["by_ratio"].to_csv(csv_dir / "exp5_crossmag_ratio_summary.csv", index=False)
        tables["drift_drop"].to_csv(csv_dir / "exp5_crossmag_drift_drop.csv", index=False)
        tables["summary"].to_csv(csv_dir / "exp5_crossmag_asymmetry_summary.csv", index=False)
        _plot(tables, fig_dir / "cross_magnification_asymmetry.png")
    return tables


def main():
    parser = argparse.ArgumentParser(description="VN26 cross-magnification asymmetry analysis.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="Use cached VN26 features for centroid drift.")
    mode.add_argument("--demo", action="store_true", help="Use deterministic demo drifts; no feature cache needed.")
    mode.add_argument(
        "--from-csv",
        action="store_true",
        help="Regenerate the figure from canonical saved CSVs without reading feature caches.",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    save = (not args.no_save) and (not args.demo)
    mode_name = "from-csv" if args.from_csv else ("demo" if args.demo else "real")
    print(f"Mode: {mode_name}", flush=True)
    if args.from_csv:
        csv_path, fig_path = _output_dirs()
        tables = _load_saved_tables(csv_path)
        _plot(tables, fig_path / "cross_magnification_asymmetry.png")
    else:
        tables = run(real=not args.demo, save=save)
    print("\n=== Cross-magnification asymmetry by pair ===")
    print(tables["by_pair"].round(4).to_string(index=False))
    print("\n=== Ratio summary ===")
    print(tables["by_ratio"].round(4).to_string(index=False))
    drift_rows = int(tables["drift_drop"]["centroid_drift"].notna().sum()) if "centroid_drift" in tables["drift_drop"] else 0
    print(f"\nDrift rows with centroid_drift: {drift_rows}/{len(tables['drift_drop'])}")
    if len(tables["summary"]):
        print("\n=== Drift/drop summary ===")
        print(tables["summary"].round(4).to_string(index=False))
    else:
        print("\nNo drift/drop summary produced. In --real mode, check VN26 feature caches and exp1b_feature_geometry.csv.")
    if save:
        csv_dir, fig_dir = _output_dirs()
        outputs = [
            csv_dir / "exp5_crossmag_asymmetry_by_backbone.csv",
            csv_dir / "exp5_crossmag_asymmetry_by_pair.csv",
            csv_dir / "exp5_crossmag_ratio_summary.csv",
            csv_dir / "exp5_crossmag_drift_drop.csv",
            csv_dir / "exp5_crossmag_asymmetry_summary.csv",
            fig_dir / "cross_magnification_asymmetry.png",
            fig_dir / "cross_magnification_asymmetry.pdf",
        ]
        write_provenance(
            "exp5_crossmag_asymmetry",
            outputs,
            protocol="vn26_directed_cross_magnification_v1",
            parameters={
                "mode": mode_name,
                "backbones": BB_ORDER,
                "magnifications": TIER_C,
            },
            inputs=[
                require_csv("exp5_full_crossmag_accuracy.csv"),
                require_csv("exp1b_feature_geometry.csv"),
            ],
        )
        print(f"\nSaved CSV outputs to {csv_dir}")
        print(f"Saved figure to {fig_dir / 'cross_magnification_asymmetry.png'}")
    elif args.demo:
        print("\nDemo mode does not save outputs. Run with --real on the full cache to write CSV/figure files.")


if __name__ == "__main__":
    main()
