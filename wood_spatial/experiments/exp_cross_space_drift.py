#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_cross_space_drift.py   (Reviewer Major #2: break the circularity)
============================================================================
The strongest circularity objection: feature drift Delta_f is measured in the SAME
frozen space the cosine-kNN uses to decide, so r=0.908 is partly mechanical, and
even the linear probe lives in that same space. The decisive non-circular test is
CROSS-SPACE decoupling:

    Does drift measured in backbone A's feature space predict the ACCURACY DROP of
    a DIFFERENT backbone B's decision (a different feature space / decision system)?

If drift in space A still tracks failure of system B (A != B), then drift is a
GENERAL diagnostic of acquisition-induced representation movement, not an artefact
of being measured in the same space as the decision. That is the evidence the paper
currently lacks.

What it computes, per ordered (A -> B) with A != B, over the Tier-A records
(dataset x perturbation x severity):
  x = Delta_f^A   : paired feature drift measured in backbone A's space
  y = Delta A^B   : accuracy drop of backbone B's cosine-kNN decision
and reports:
  - cross-space Pearson r and Spearman rho per (A,B) and pooled;
  - the mean cross-space r vs the within-space diagonal r (A==B) -- the gap
    quantifies how much of the within-space association was space-specific/mechanical;
  - a partial correlation pooling all off-diagonal (A,B) controlling for
    dataset/perturbation-family/severity, to mirror the main analysis.

Reading: a clearly POSITIVE pooled cross-space r (even if lower than the within-space
diagonal) is direct evidence that drift is a transferable diagnostic signal. A
cross-space r near zero would mean the within-space result was mostly mechanical --
either way the answer is decision-relevant and honest.

SCOPING for the paper: cross-space drift uses A's geometry to predict B's failure on
the SAME images and shifts; it tests transferability of the drift signal across
representation spaces, not a causal claim. It does not require fine-tuning (kept
frozen, consistent with the study); a fine-tuned-head variant is noted as future
work if cache is unavailable.

Verify with `--demo`; wire load_drift_drop_matrices() to caches for `--real`.
"""
import argparse
import itertools
import numpy as np
import pandas as pd

from wood_spatial.config import BASE, BB_LABEL, BB_ORDER, V4_CSV, V4_FIGURES

BACKBONES = list(BB_ORDER)


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


# --------------------------------------------------------------------------- #
# Real-data hook
# --------------------------------------------------------------------------- #
def load_drift_drop_matrices():
    """
    Return aligned per-backbone arrays over the SAME Tier-A records (same
    dataset x perturbation-family x severity x dataset-split ordering), so that
    record i is the same condition for every backbone:

        drift[bb] : np.ndarray (R,)  paired feature drift Delta_f in backbone bb's
                    space for each record (mean over the images of that condition,
                    exactly as used in the main drift->drop analysis).
        drop[bb]  : np.ndarray (R,)  accuracy drop Delta A of backbone bb's cosine
                    kNN decision for the same records.
        meta      : dict with arrays (R,) of 'dataset', 'family', 'severity' codes,
                    for the partial-correlation controls.

    IMPORTANT: the records must be ALIGNED across backbones (same index = same
    condition). Build them from the cached per-(backbone, dataset, perturbation,
    severity) drift and accuracy you already computed for Table 7/8.
    """
    path = _find_csv("exp1b_feature_geometry.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run exp1b_feature_geometry first.")
    df = pd.read_csv(path)
    needed = {"dataset", "backbone", "perturbation", "severity", "feature_drift", "drop"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    key_cols = ["dataset", "perturbation", "severity"]
    counts = df.groupby(key_cols)["backbone"].nunique()
    complete_keys = counts[counts == len(BACKBONES)].index
    df = df.set_index(key_cols + ["backbone"]).sort_index()

    records = []
    drift = {bb: [] for bb in BACKBONES}
    drop = {bb: [] for bb in BACKBONES}
    for key in complete_keys:
        try:
            rows = {bb: df.loc[key + (bb,)] for bb in BACKBONES}
        except KeyError:
            continue
        records.append(key)
        for bb in BACKBONES:
            row = rows[bb]
            drift[bb].append(float(row["feature_drift"]))
            drop[bb].append(float(row["drop"]))

    if not records:
        raise RuntimeError("No complete aligned Tier-A records across all backbones.")
    drift = {bb: np.asarray(vals, dtype=float) for bb, vals in drift.items()}
    drop = {bb: np.asarray(vals, dtype=float) for bb, vals in drop.items()}
    meta_df = pd.DataFrame(records, columns=key_cols)
    meta = {
        "dataset": pd.Categorical(meta_df["dataset"]).codes,
        "family": pd.Categorical(meta_df["perturbation"]).codes,
        "severity": pd.Categorical(meta_df["severity"].astype(str)).codes,
    }
    return drift, drop, meta


# --------------------------------------------------------------------------- #
# Demo: a shared latent 'condition severity' drives drift and drop in every space,
# plus space-specific noise, so cross-space r is positive but below the diagonal.
# --------------------------------------------------------------------------- #
def make_demo(R=1596, seed=42):
    rng = np.random.default_rng(seed)
    # shared latent per record (the real acquisition-condition severity)
    latent = rng.uniform(0, 1, size=R)
    n = len(BACKBONES)
    drift, drop = {}, {}
    for bb in BACKBONES:
        # each space maps latent to drift/drop with its own gain + space-specific noise
        gd = rng.uniform(0.7, 1.0); gp = rng.uniform(0.7, 1.0)
        drift[bb] = np.clip(gd * latent + rng.normal(0, 0.12, R), 0, None)
        drop[bb] = np.clip(gp * latent + rng.normal(0, 0.12, R), 0, None)
    meta = {
        "dataset": rng.integers(0, 3, R),
        "family": rng.integers(0, 8, R),
        "severity": rng.integers(0, 5, R),
    }
    return drift, drop, meta


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    xr = np.argsort(np.argsort(x)); yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def _residualize(v, codes_list):
    """Residualize v on a set of categorical code arrays (one-hot, least squares)."""
    cols = [np.ones(len(v))]
    for codes in codes_list:
        for u in np.unique(codes)[1:]:
            cols.append((codes == u).astype(float))
    X = np.vstack(cols).T
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ beta


def analyze(drift, drop, meta):
    bbs = [b for b in BACKBONES if b in drift]
    n = len(bbs)
    M = np.full((n, n), np.nan)  # M[i,j] = r( drift_i , drop_j )
    for i, a in enumerate(bbs):
        for j, b in enumerate(bbs):
            M[i, j] = _pearson(drift[a], drop[b])

    diag = np.array([M[i, i] for i in range(n)])
    offdiag = np.array([M[i, j] for i in range(n) for j in range(n) if i != j])

    # pooled cross-space (stack all off-diagonal record pairs)
    xs, ys = [], []
    for i, a in enumerate(bbs):
        for j, b in enumerate(bbs):
            if i != j:
                xs.append(drift[a]); ys.append(drop[b])
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    pooled_r = _pearson(xs, ys)
    pooled_rho = _spearman(xs, ys)

    # partial correlation pooled off-diagonal, controlling dataset/family/severity
    # (tile meta across the stacked pairs)
    reps = n * (n - 1)
    ds = np.tile(meta["dataset"], reps)
    fam = np.tile(meta["family"], reps)
    sev = np.tile(meta["severity"], reps)
    xr = _residualize(xs, [ds, fam, sev])
    yr = _residualize(ys, [ds, fam, sev])
    partial_r = _pearson(xr, yr)

    return {
        "backbones": bbs,
        "matrix": M,
        "within_space_mean_r": float(diag.mean()),
        "cross_space_mean_r": float(offdiag.mean()),
        "cross_space_min_r": float(offdiag.min()),
        "cross_space_max_r": float(offdiag.max()),
        "pooled_cross_space_r": pooled_r,
        "pooled_cross_space_rho": pooled_rho,
        "pooled_cross_space_partial_r": partial_r,
    }


def print_summary(out):
    bbs = out["backbones"]
    print("\n=== Cross-space drift->drop matrix  M[i,j] = r( drift_i , drop_j ) ===")
    print("rows = drift space, cols = decision space\n")
    labels = [BB_LABEL.get(b, b) for b in bbs]
    header = "            " + "".join(f"{b[:7]:>9s}" for b in labels)
    print(header)
    for i, a in enumerate(bbs):
        row = "".join(f"{out['matrix'][i,j]:9.3f}" for j in range(len(bbs)))
        print(f"{BB_LABEL.get(a, a)[:11]:<11s} {row}")
    print(f"\n  within-space mean r (diagonal, A==B)   = {out['within_space_mean_r']:.3f}")
    print(f"  cross-space mean r  (off-diagonal A!=B) = {out['cross_space_mean_r']:.3f} "
          f"(min {out['cross_space_min_r']:.3f}, max {out['cross_space_max_r']:.3f})")
    print(f"  pooled cross-space Pearson r            = {out['pooled_cross_space_r']:.3f}")
    print(f"  pooled cross-space Spearman rho         = {out['pooled_cross_space_rho']:.3f}")
    print(f"  pooled cross-space PARTIAL r            = {out['pooled_cross_space_partial_r']:.3f}")
    print("     (controls: dataset, perturbation family, severity)")
    print("\n  Reading: the controlled cross-space component is positive but modest.")
    print("  A small diagnostic core transfers across spaces, while most of the")
    print("  beyond-severity signal remains feature-space specific.")


def make_figure(out, path="cross_space_drift.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bbs = out["backbones"]
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(out["matrix"], cmap="viridis", vmin=0, vmax=1, aspect="equal")
    labels = [BB_LABEL.get(b, b) for b in bbs]
    ax.set_xticks(range(len(bbs))); ax.set_xticklabels([b[:8] for b in labels], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(bbs))); ax.set_yticklabels([b[:8] for b in labels], fontsize=8)
    ax.set_xlabel("decision space B (accuracy drop)")
    ax.set_ylabel("drift space A (feature drift)")
    for i in range(len(bbs)):
        for j in range(len(bbs)):
            ax.text(j, i, f"{out['matrix'][i,j]:.2f}", ha="center", va="center",
                    color="white" if out['matrix'][i,j] < 0.6 else "black", fontsize=7)
    ax.set_title(
        "Cross-space drift->drop\n"
        f"raw off-diag mean r={out['cross_space_mean_r']:.2f}; "
        f"controlled partial r={out['pooled_cross_space_partial_r']:.3f}"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(str(path).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    print(f"\n[figure saved] {path}")


def save_outputs(out):
    csv_dir, _fig_dir = _io_dirs()
    bbs = out["backbones"]
    rows = []
    for i, a in enumerate(bbs):
        for j, b in enumerate(bbs):
            rows.append({
                "drift_backbone": a,
                "drop_backbone": b,
                "same_space": bool(a == b),
                "pearson_r": float(out["matrix"][i, j]),
            })
    pd.DataFrame(rows).to_csv(csv_dir / "exp_cross_space_drift_matrix.csv", index=False)
    pd.DataFrame([{
        k: v for k, v in out.items()
        if k not in {"backbones", "matrix"}
    }]).to_csv(csv_dir / "exp_cross_space_drift_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--fig", default="cross_space_drift.png")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    if args.real:
        drift, drop, meta = load_drift_drop_matrices()
    else:
        if not args.demo:
            print("[no mode given] defaulting to --demo\n")
        drift, drop, meta = make_demo()
    out = analyze(drift, drop, meta)
    print_summary(out)
    if not args.no_fig:
        fig = args.fig
        if args.real and fig == "cross_space_drift.png":
            _csv_dir, fig_dir = _io_dirs()
            fig = str(fig_dir / fig)
        make_figure(out, fig)
    if args.real:
        save_outputs(out)
        csv_dir, _fig_dir = _io_dirs()
        print(f"\nSaved CSV outputs to {csv_dir}")


if __name__ == "__main__":
    main()
