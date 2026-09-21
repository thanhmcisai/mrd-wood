#!/usr/bin/env python3
"""Filter stale WOODAUTH rows and recompute lightweight summary CSVs.

This script does not regenerate feature caches. It removes WOODAUTH rows from
CSV outputs that are simple tabular summaries and recomputes aggregate monitor
metrics from the remaining records.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


CSV_DIR = Path("results/csv")
DROP_DATASET = "WOODAUTH"


def filter_dataset_csv(name: str) -> pd.DataFrame:
    path = CSV_DIR / name
    df = pd.read_csv(path)
    if "dataset" in df.columns:
        df = df[df["dataset"] != DROP_DATASET].copy()
        df.to_csv(path, index=False)
    return df


def auc_row(name: str, scores: pd.Series, y: pd.Series, drop: pd.Series):
    s = pd.to_numeric(scores, errors="coerce")
    valid = s.notna() & y.notna()
    if valid.sum() < 20 or y[valid].nunique() < 2:
        return None
    vals = s[valid].to_numpy()
    yy = y[valid].to_numpy(dtype=int)
    prec, rec, th = precision_recall_curve(yy, vals)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best = int(np.argmax(f1))
    return {
        "detector": name,
        "deployable": not name.startswith("paired_"),
        "auc_roc": float(roc_auc_score(yy, vals)),
        "avg_precision": float(average_precision_score(yy, vals)),
        "best_f1": float(f1[best]),
        "best_threshold": float(th[best]) if best < len(th) else np.nan,
        "r_vs_drop": float(np.corrcoef(vals, drop[valid].to_numpy())[0, 1]),
        "n": int(valid.sum()),
    }


def recompute_exp10_auc() -> None:
    scores = filter_dataset_csv("exp10_reference_monitor_scores.csv")
    y = scores["failure"].astype(int)
    detectors = [
        "ref_knn1_dist",
        "ref_knn5_dist",
        "ref_mahalanobis",
        "ref_mahalanobis_delta",
        "ref_centroid_cosine",
        "ref_mmd_rbf",
        "paired_feature_drift_oracle",
    ]
    rows = []
    for det in detectors:
        row = auc_row(det, scores[det], y, scores["accuracy_drop"])
        if row is not None:
            rows.append(row)
    auc = pd.DataFrame(rows)
    if len(auc):
        auc = auc.sort_values(["deployable", "auc_roc"], ascending=[False, False])
    auc.to_csv(CSV_DIR / "exp10_reference_monitor_auc.csv", index=False)


def recompute_exp9_correlations() -> None:
    geom = filter_dataset_csv("exp9_tierb_geometry.csv")
    if {"dataset", "feature_drift", "drop"}.issubset(geom.columns):
        rows = []
        for ds, group in geom.groupby("dataset", sort=True):
            valid = group[["feature_drift", "drop"]].dropna()
            if len(valid) >= 3:
                rows.append(
                    {
                        "dataset": ds,
                        "r": float(valid["feature_drift"].corr(valid["drop"])),
                        "p": np.nan,
                        "n": int(len(valid)),
                    }
                )
        pd.DataFrame(rows).to_csv(CSV_DIR / "exp9_tierb_correlations.csv", index=False)
        tier = pd.DataFrame(
            [
                {
                    "tier": "Tier-B retained",
                    "n_records": int(len(geom)),
                    "mean_drop": float(geom["drop"].mean()),
                    "r_feature_drift_drop": float(geom["feature_drift"].corr(geom["drop"])),
                }
            ]
        )
        tier.to_csv(CSV_DIR / "exp9_tier_comparison.csv", index=False)


def main() -> None:
    for name in [
        "exp4_tierB_accuracy.csv",
        "exp4_tierB_perturbed.csv",
        "exp10_lodo_monitor.csv",
        "exp10_reference_monitor_scores.csv",
    ]:
        if (CSV_DIR / name).exists():
            filter_dataset_csv(name)
    recompute_exp10_auc()
    recompute_exp9_correlations()
    print("[OK] filtered WOODAUTH rows and recomputed lightweight summaries")


if __name__ == "__main__":
    main()
