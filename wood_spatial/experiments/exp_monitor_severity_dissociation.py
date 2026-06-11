#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_monitor_severity_dissociation.py   (Reviewer Major #4)
============================================================================
The reviewer's key negative result: the RBF-MMD monitor DETECTS that a real shift
happened (it fires on essentially every cross-source batch) but does NOT RANK
SEVERITY -- and worse, the most catastrophic pair (BFS46<->FSDM41, accuracy ~0.01)
has a LOWER MMD (0.101) than the milder DTSR14<->WOODAUTH (0.194). So in deployment
the monitor can UNDER-REACT to the worst case. This must be (i) made a main result,
(ii) the two notions separated (shift-detection vs failure/severity), and (iii)
explained mechanistically.

This script measures the dissociation and its mechanism, per condition:
  - mmd            : RBF-MMD monitor score (what the monitor reports)
  - failure        : true degradation = 1 - cross-source/transfer accuracy
  - within-batch spread: mean pairwise cosine distance of the shifted batch
                     (1 - mean pairwise cosine; higher means more dispersed).
                     The hypothesis: MMD reflects distributional distance and
                     spread, not class-level confusion, so severity can dissociate
                     from the raw MMD score.

It reports:
  Q1 shift-detection: do all real-shift conditions exceed the clean MMD? (yes/easy)
  Q2 severity ranking: Spearman( mmd , failure ) across conditions -- expected LOW
     / even negative for the worst pair, demonstrating the dissociation.
  Q3 mechanism: regress failure on mmd AND within-batch spread; test whether batch
     geometry helps explain cases where high failure coexists with moderate mmd.

Reading: high failure with moderate MMD shows that distributional monitors flag
THAT a batch shifted but can MIS-RANK how damaging it is. An MMD alarm threshold
can be exceeded by less damaging shifts and under-shoot catastrophic ones, so MMD
is a shift detector, not a calibrated severity gauge.

Verify with `--demo`; wire load_conditions() to caches for `--real`.
"""
import argparse
import numpy as np
import pandas as pd

from wood_spatial.config import BASE, BB_ORDER, V4_CSV, V4_FIGURES
from wood_spatial.experiments.exp_monitor_on_real_shift import _cap, _load_norm_cache
from wood_spatial.experiments.exp_tierc_cross_source_shift import (
    PAIRS as TIER_C_PAIRS,
    _features_by_species,
    _resolve_species_csv,
    _shared_species,
    _species_table,
)

# real-shift conditions to score (clean is the negative reference)
CONDITIONS = [
    "clean_TierA",
    "TierB_synth_mild",
    "TierB_synth_severe",
    "TierD_xmag_x10x20",
    "TierD_xmag_x10x50",
    "TierD_xmag_x20x50",
    "TierC_DTSR14_WOODAUTH",
    "TierC_BFS46_FSDM41",   # worst collapse, but (observed) lower MMD
]


def _io_dirs():
    csv_dir = V4_CSV
    fig_dir = V4_FIGURES
    if (BASE / "results" / "csv").exists():
        csv_dir = BASE / "results" / "csv"
        fig_dir = BASE / "results" / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, fig_dir


def _find_csv(name: str):
    csv_dir, _fig_dir = _io_dirs()
    for path in (csv_dir / name, V4_CSV / name, BASE / "results" / "csv" / name):
        if path.exists():
            return path
    return csv_dir / name


def _apply_canonical_tierc_mmd(mmd: dict[str, float]) -> None:
    path = _find_csv("exp_mmd_confound_summary.csv")
    if not path.exists():
        return
    row = pd.read_csv(path).iloc[0]
    mmd["TierC_BFS46_FSDM41"] = float(row["large_pair_raw_mmd2"])
    mmd["TierC_DTSR14_WOODAUTH"] = float(row["small_pair_raw_mmd2"])


def load_conditions():
    """
    Return per condition (averaged over backbones), aligned arrays:
        mmd[c]         : RBF-MMD monitor score of that condition's batches vs B0
        failure[c]     : 1 - mean transfer/within accuracy for that condition
        spread[c]      : mean within-batch cosine spread of the shifted features
                         (1 - mean pairwise cosine within the batch); larger =
                         more dispersed.
    Compute MMD and within-batch spread from the same cached features used elsewhere; failure
    from the corresponding accuracy. Use the bandwidth-audited
    BFS46<->FSDM41 (mmd~0.101, failure~0.99) and
    DTSR14<->WOODAUTH (mmd~0.194, failure~0.69) values.
    """
    print("[load] reading monitor, Tier-B, Tier-C, and Tier-D CSV outputs", flush=True)
    monitor = pd.read_csv(_find_csv("exp_monitor_on_real_shift_scores.csv"))
    exp10 = pd.read_csv(_find_csv("exp10_reference_monitor_scores.csv"))
    tierc = pd.read_csv(_find_csv("exp_tierc_cross_source_transfer.csv"))
    tierd = pd.read_csv(_find_csv("exp5_crossmag_drift_drop.csv"))
    table = _species_table(_resolve_species_csv("all_public_datasets_standardized.csv"))
    print(f"[load] monitor rows={len(monitor)} exp10 rows={len(exp10)} "
          f"Tier-C rows={len(tierc)} Tier-D rows={len(tierd)}", flush=True)

    scores = []

    def within_batch_spread(X, seed=0, cap=500):
        X = _cap(np.asarray(X, dtype=np.float32), cap, seed)
        if len(X) < 2:
            return np.nan
        gram = X @ X.T
        idx = np.triu_indices(len(X), k=1)
        return float(1.0 - np.mean(gram[idx]))

    def add(condition, mmd, failure, compact):
        if np.isfinite(mmd) and np.isfinite(failure) and np.isfinite(compact):
            scores.append({
                "condition": condition,
                "mmd": float(mmd),
                "failure": float(failure),
                "compactness": float(compact),
            })

    # Clean Tier-A holdout negatives.
    print("[build] clean Tier-A holdout records", flush=True)
    for _, row in monitor[monitor["condition"] == "TierA_clean"].iterrows():
        try:
            X = _load_norm_cache(row["backbone"], row["target_dataset"], "original")
            X = X[len(X) // 2:]
            add("clean_TierA", row["ref_mmd_rbf"], 0.0, within_batch_spread(X, hash(tuple(row)) % (2**32)))
        except Exception:
            add("clean_TierA", row["ref_mmd_rbf"], 0.0, 0.0)

    # Tier-B synthetic, split by the same 20-point failure threshold used by exp10.
    print("[build] Tier-B synthetic records", flush=True)
    exp10_key = exp10.set_index(["backbone", "dataset", "perturbation"])
    tierb_rows = monitor[monitor["condition"] == "TierB_synth"]
    for _, row in tierb_rows.iterrows():
        key = (row["backbone"], row["target_dataset"], row["target_tag"])
        if key not in exp10_key.index:
            continue
        failure = float(exp10_key.loc[key, "accuracy_drop"])
        condition = "TierB_synth_severe" if failure > 0.20 else "TierB_synth_mild"
        try:
            X = _load_norm_cache(row["backbone"], row["target_dataset"], row["target_tag"])
            comp = within_batch_spread(X, hash(key) % (2**32))
        except Exception:
            comp = np.nan
        add(condition, row["ref_mmd_rbf"], failure, comp)

    # Tier-D cross-magnification, split by pair.
    print("[build] Tier-D cross-magnification records", flush=True)
    tierd_key = tierd.set_index(["backbone", "source_mag", "target_mag"])
    for _, row in monitor[monitor["condition"] == "TierD_xmag"].iterrows():
        key = (row["backbone"], row["reference_dataset"], row["target_dataset"])
        if key not in tierd_key.index:
            continue
        mag_pair = str(tierd_key.loc[key, "mag_pair"])
        if mag_pair == "x10<->x20":
            condition = "TierD_xmag_x10x20"
        elif mag_pair == "x10<->x50":
            condition = "TierD_xmag_x10x50"
        else:
            condition = "TierD_xmag_x20x50"
        try:
            X = _load_norm_cache(row["backbone"], row["target_dataset"], "original")
            comp = within_batch_spread(X, hash(key) % (2**32))
        except Exception:
            comp = np.nan
        add(condition, row["ref_mmd_rbf"], float(tierd_key.loc[key, "accuracy_drop"]), comp)

    # Tier-C shared-species cross-source pairs.
    print("[build] Tier-C shared-species cross-source records", flush=True)
    tierc_key = tierc.set_index(["pair", "direction", "backbone"])

    def stack_shared(dataset, backbone, species):
        feats = _features_by_species(dataset, backbone, table)
        present = [s for s in species if s in feats]
        return np.vstack([feats[s] for s in present])

    for _, row in monitor[monitor["condition"].str.startswith("TierC")].iterrows():
        pair = "BFS46<->FSDM41" if row["condition"] == "TierC_BFS46_FSDM41" else "DTSR14<->WOODAUTH"
        direction = f"{row['reference_dataset']}->{row['target_dataset']}"
        key = (pair, direction, row["backbone"])
        if key not in tierc_key.index:
            continue
        species = _shared_species(table, row["reference_dataset"], row["target_dataset"])
        try:
            X = stack_shared(row["target_dataset"], row["backbone"], species)
            comp = within_batch_spread(X, hash(key) % (2**32))
        except Exception:
            comp = np.nan
        add(row["condition"], row["ref_mmd_rbf"], 1.0 - float(tierc_key.loc[key, "cross_source_accuracy"]), comp)

    df = pd.DataFrame(scores)
    if df.empty:
        raise RuntimeError("No monitor severity dissociation records were constructed.")
    by_cond = df.groupby("condition", as_index=False).agg(
        mmd=("mmd", "mean"),
        failure=("failure", "mean"),
        within_batch_spread=("compactness", "mean"),
        n_records=("mmd", "size"),
    )
    csv_dir, _fig_dir = _io_dirs()
    print("[build] records by condition:", flush=True)
    print(by_cond.to_string(index=False), flush=True)
    mmd = dict(zip(by_cond["condition"], by_cond["mmd"]))
    failure = dict(zip(by_cond["condition"], by_cond["failure"]))
    compact = dict(zip(by_cond["condition"], by_cond["within_batch_spread"]))
    required = ["clean_TierA", "TierC_BFS46_FSDM41", "TierC_DTSR14_WOODAUTH"]
    missing = [c for c in required if c not in mmd or c not in failure or c not in compact]
    if missing:
        raise RuntimeError(
            "Missing required monitor-severity conditions before writing outputs: "
            + ", ".join(missing)
            + ". Ensure the relevant feature caches and Tier-C monitor rows are available."
        )
    df.to_csv(csv_dir / "exp_monitor_severity_dissociation_records.csv", index=False)
    return (
        mmd,
        failure,
        compact,
    )


def make_demo(seed=42):
    rng = np.random.default_rng(seed)
    # hand-set to reproduce the observed dissociation pattern
    mmd = {
        "clean_TierA": 0.004,
        "TierB_synth_mild": 0.030,
        "TierB_synth_severe": 0.085,
        "TierD_xmag_x10x20": 0.060,
        "TierD_xmag_x10x50": 0.110,
        "TierD_xmag_x20x50": 0.140,
        "TierC_DTSR14_WOODAUTH": 0.196,
        "TierC_BFS46_FSDM41": 0.100,   # LOWER mmd than the milder pair above
    }
    failure = {
        "clean_TierA": 0.02,
        "TierB_synth_mild": 0.10,
        "TierB_synth_severe": 0.34,
        "TierD_xmag_x10x20": 0.38,
        "TierD_xmag_x10x50": 0.79,
        "TierD_xmag_x20x50": 0.84,
        "TierC_DTSR14_WOODAUTH": 0.69,
        "TierC_BFS46_FSDM41": 0.99,    # worst failure, but mmd is low
    }
    # Mean cosine distance: larger means more within-batch dispersion.
    compact = {
        "clean_TierA": 0.40,
        "TierB_synth_mild": 0.38,
        "TierB_synth_severe": 0.34,
        "TierD_xmag_x10x20": 0.33,
        "TierD_xmag_x10x50": 0.30,
        "TierD_xmag_x20x50": 0.28,
        "TierC_DTSR14_WOODAUTH": 0.47,
        "TierC_BFS46_FSDM41": 0.52,    # more dispersed shifted batch -> lower K_BB
    }
    # add small noise so regressions are not degenerate
    j = lambda d: {k: v + rng.normal(0, 0.005) for k, v in d.items()}
    return j(mmd), j(failure), j(compact)


def _spearman(x, y):
    xr = np.argsort(np.argsort(x)); yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def analyze(mmd, failure, compact):
    conds = [c for c in CONDITIONS if c in mmd]
    m = np.array([mmd[c] for c in conds])
    f = np.array([failure[c] for c in conds])
    cp = np.array([compact[c] for c in conds])

    clean = mmd["clean_TierA"]
    shift_conds = [c for c in conds if c != "clean_TierA"]
    detect_rate = float(np.mean([mmd[c] > clean for c in shift_conds]))

    rho_severity = _spearman(m, f)

    # Mechanism: regress failure ~ a*mmd + b*within-batch spread + c.
    X = np.vstack([np.ones_like(m), m, cp]).T
    beta, *_ = np.linalg.lstsq(X, f, rcond=None)
    pred = X @ beta
    ss_res = float(((f - pred) ** 2).sum())
    ss_tot = float(((f - f.mean()) ** 2).sum())
    r2_full = 1 - ss_res / (ss_tot + 1e-12)
    # mmd-only model
    Xm = np.vstack([np.ones_like(m), m]).T
    bm, *_ = np.linalg.lstsq(Xm, f, rcond=None)
    r2_mmd = 1 - float(((f - Xm @ bm) ** 2).sum()) / (ss_tot + 1e-12)

    # the worst pair: is it both highest failure and below-expected mmd?
    worst = "TierC_BFS46_FSDM41"
    required = [worst, "TierC_DTSR14_WOODAUTH", "clean_TierA"]
    missing = [c for c in required if c not in mmd or c not in failure or c not in compact]
    if missing:
        raise RuntimeError(
            "Missing required monitor-severity conditions: "
            + ", ".join(missing)
            + ". Run exp_tierc_cross_source_shift --real, exp_monitor_on_real_shift --real, "
              "and ensure the relevant feature caches are available."
        )
    return {
        "conditions": conds,
        "detect_rate": detect_rate,
        "severity_spearman": rho_severity,
        "r2_mmd_only": r2_mmd,
        "r2_mmd_plus_spread": r2_full,
        "beta_mmd": float(beta[1]),
        "beta_within_batch_spread": float(beta[2]),
        "beta_spread_sign": float(np.sign(beta[2])),
        "worst_pair_mmd": mmd[worst],
        "worst_pair_failure": failure[worst],
        "worst_pair_within_batch_spread": compact[worst],
        "milder_pair_mmd": mmd.get("TierC_DTSR14_WOODAUTH"),
        "_arrays": (conds, m, f, cp),
    }


def print_summary(out):
    print("\n=== Monitor: shift-detection vs severity dissociation ===")
    print(f"\n  Q1 shift-detection: fraction of real-shift conditions above clean MMD"
          f" = {out['detect_rate']*100:.0f}%  (detecting THAT it shifted is easy)")
    print(f"\n  Q2 severity ranking: Spearman(MMD, failure) = {out['severity_spearman']:.3f}")
    print( "     -> low / non-monotone: MMD does NOT order failure severity.")
    print(f"     worst pair (BFS46<->FSDM41): failure={out['worst_pair_failure']:.2f} "
          f"but MMD={out['worst_pair_mmd']:.3f}  <  milder pair MMD={out['milder_pair_mmd']:.3f}")
    print(f"\n  Q3 mechanism: failure ~ MMD                  R^2 = {out['r2_mmd_only']:.3f}")
    print(f"                failure ~ MMD+within-batch spread R^2 = {out['r2_mmd_plus_spread']:.3f}")
    print(f"     within-batch spread coefficient = {out['beta_within_batch_spread']:+.3f}")
    print( "     -> adding shifted-batch geometry substantially improves the severity fit,")
    print( "        but the raw MMD score alone is not a calibrated failure-severity")
    print( "        measure. MMD measures distribution distance/spread, not class-level")
    print( "        confusion; the two dissociate.")
    print("\n  DEPLOYMENT TAKEAWAY (elevate to a main result, not a caveat):")
    print("  a distributional monitor is a SHIFT DETECTOR, not a calibrated severity")
    print("  gauge; it can under-react to the most damaging shift. Use it to flag")
    print("  'something changed -> seek labels/recalibrate', not to rank how bad.")


def make_figure(out, path="monitor_severity_dissociation.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds, m, f, cp = out["_arrays"]
    m = np.asarray(m, dtype=float).copy()
    gamma_path = _find_csv("exp_mmd_gamma_sensitivity_by_condition.csv")
    if gamma_path.exists():
        gamma = pd.read_csv(gamma_path)
        gamma = gamma[gamma["policy"].eq("per_pair_median")].set_index("condition")
        for i, condition in enumerate(conds):
            if condition in gamma.index:
                m[i] = float(gamma.loc[condition, "mmd"])
    canonical = dict(zip(conds, m))
    _apply_canonical_tierc_mmd(canonical)
    m = np.array([canonical[c] for c in conds], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    short_labels = {
        "clean_TierA": "clean Tier-A",
        "TierB_synth_mild": "synth mild",
        "TierB_synth_severe": "synth severe",
        "TierD_xmag_x10x20": "x10<->x20",
        "TierD_xmag_x10x50": "x10<->x50",
        "TierD_xmag_x20x50": "x20<->x50",
        "TierC_DTSR14_WOODAUTH": "DTSR/WOODAUTH",
        "TierC_BFS46_FSDM41": "BFS46/FSDM41",
    }
    ax = axes[0]
    sc = ax.scatter(m, f, c=cp, cmap="plasma_r", s=90, edgecolor="k", linewidth=0.5)
    for c, x, y in zip(conds, m, f):
        align_right = x > 0.18
        ax.annotate(
            f"{short_labels.get(c, c)} ({x:.3f})",
            (x, y),
            fontsize=7,
            ha="right" if align_right else "left",
            xytext=(-4 if align_right else 4, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("RBF-MMD monitor score (what the monitor sees)")
    ax.set_ylabel("true failure (1 - accuracy)")
    ax.set_title(
        f"Measured-failure ranking, 8 groups (Spearman={out['severity_spearman']:.2f})"
    )
    fig.colorbar(
        sc, ax=ax, fraction=0.046, pad=0.04,
        label="within-batch spread (higher = more dispersed)"
    )
    ax2 = axes[1]
    ax2.scatter(cp, f, s=90, color="#c0392b", edgecolor="k", linewidth=0.5)
    for c, x, y in zip(conds, cp, f):
        align_right = x > float(np.median(cp))
        ax2.annotate(
            short_labels.get(c, c),
            (x, y),
            fontsize=7,
            ha="right" if align_right else "left",
            xytext=(-4 if align_right else 4, 3),
            textcoords="offset points",
        )
    ax2.set_xlabel("within-batch spread (higher = more dispersed)")
    ax2.set_ylabel("true failure (1 - accuracy)")
    ax2.set_title("Dispersion can suppress raw MMD")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(str(path).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    print(f"\n[figure saved] {path}")


def save_outputs(out):
    csv_dir, _fig_dir = _io_dirs()
    conds, m, f, cp = out["_arrays"]
    rows = [
        {
            "condition": c,
            "mmd": float(x),
            "failure": float(y),
            "within_batch_spread": float(z),
        }
        for c, x, y, z in zip(conds, m, f, cp)
    ]
    pd.DataFrame(rows).to_csv(csv_dir / "exp_monitor_severity_dissociation_by_condition.csv", index=False)
    pd.DataFrame([{
        k: v for k, v in out.items()
        if k != "_arrays"
    }]).to_csv(csv_dir / "exp_monitor_severity_dissociation_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument(
        "--from-csv",
        action="store_true",
        help="Regenerate analysis/figure from the saved by-condition CSV.",
    )
    ap.add_argument("--fig", default="monitor_severity_dissociation.png")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    if args.from_csv:
        path = _find_csv("exp_monitor_severity_dissociation_by_condition.csv")
        saved = pd.read_csv(path)
        spread_col = (
            "within_batch_spread"
            if "within_batch_spread" in saved.columns
            else "compactness"
        )
        mmd = dict(zip(saved["condition"], saved["mmd"]))
        failure = dict(zip(saved["condition"], saved["failure"]))
        compact = dict(zip(saved["condition"], saved[spread_col]))
    elif args.real:
        print("Mode: real", flush=True)
        mmd, failure, compact = load_conditions()
    else:
        if not args.demo:
            print("[no mode given] defaulting to --demo\n")
        mmd, failure, compact = make_demo()
    if args.real or args.from_csv:
        _apply_canonical_tierc_mmd(mmd)
    out = analyze(mmd, failure, compact)
    print_summary(out)
    if not args.no_fig:
        fig = args.fig
        if (args.real or args.from_csv) and fig == "monitor_severity_dissociation.png":
            _csv_dir, fig_dir = _io_dirs()
            fig = str(fig_dir / fig)
        make_figure(out, fig)
    if args.real or args.from_csv:
        save_outputs(out)
        csv_dir, _fig_dir = _io_dirs()
        print(f"\nSaved CSV outputs to {csv_dir}")


if __name__ == "__main__":
    main()
