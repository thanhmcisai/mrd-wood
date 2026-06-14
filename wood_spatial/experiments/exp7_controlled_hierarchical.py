#!/usr/bin/env python3
"""
Experiment: controlled hierarchical R².

Adds representation groups after explicit dataset/backbone/perturbation/severity
controls, addressing reviewer concerns about severity and grouped dependence.
"""
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from wood_spatial.config import V4_CSV
from wood_spatial.result_io import require_csv, write_provenance

logger = logging.getLogger(__name__)


def _design(df: pd.DataFrame, categorical: list, numeric: list):
    parts = []
    names = []
    if categorical:
        try:
            enc = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        except TypeError:
            enc = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        Xc = enc.fit_transform(df[categorical].astype(str))
        parts.append(Xc)
        names.extend(enc.get_feature_names_out(categorical).tolist())
    if numeric:
        Xn = df[numeric].astype(float).values
        Xn = StandardScaler().fit_transform(Xn)
        parts.append(Xn)
        names.extend(numeric)
    if not parts:
        return np.zeros((len(df), 0)), names
    return np.hstack(parts), names


def _fit_r2(df: pd.DataFrame, y_col: str, categorical: list, numeric: list):
    cols = [y_col] + categorical + numeric
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 30:
        return np.nan, len(sub)
    X, _ = _design(sub, categorical, numeric)
    y = sub[y_col].astype(float).values
    return float(LinearRegression().fit(X, y).score(X, y)), int(len(sub))


def _partial_correlation(
    df: pd.DataFrame,
    metric: str,
    controls: list[str],
) -> tuple[float, float, int]:
    sub = df[[metric, "drop", *controls]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    X, _ = _design(sub, controls, [])
    model = LinearRegression()
    metric_residual = (
        sub[metric].to_numpy(dtype=float)
        - model.fit(X, sub[metric].to_numpy(dtype=float)).predict(X)
    )
    drop_residual = (
        sub["drop"].to_numpy(dtype=float)
        - model.fit(X, sub["drop"].to_numpy(dtype=float)).predict(X)
    )
    r, p = pearsonr(metric_residual, drop_residual)
    return float(r), float(p), int(len(sub))


def _severity_partial_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "feature_drift",
        "delta_fgcs",
        "delta_intra",
        "delta_fsr",
        "inter_collapse",
        "fsr_collapse",
    ]
    rows = []
    for metric in metrics:
        sub = df[[metric, "drop"]].replace([np.inf, -np.inf], np.nan).dropna()
        raw_r, raw_p = pearsonr(sub[metric], sub["drop"])
        no_severity_r, no_severity_p, n = _partial_correlation(
            df, metric, ["dataset", "backbone", "perturbation"]
        )
        severity_r, severity_p, n_severity = _partial_correlation(
            df, metric, ["dataset", "backbone", "perturbation", "severity_control"]
        )
        if n != n_severity:
            raise RuntimeError(
                f"Control protocols use different samples for {metric}: "
                f"{n} versus {n_severity}"
            )
        rows.append({
            "metric": metric,
            "r_raw": float(raw_r),
            "p_raw": float(raw_p),
            "r_partial_no_severity": no_severity_r,
            "p_partial_no_severity": no_severity_p,
            "r_partial_severity": severity_r,
            "p_partial_severity": severity_p,
            "n": n,
        })
    return pd.DataFrame(rows)


def _standardized_mean_difference(selected: pd.Series, omitted: pd.Series) -> float:
    selected = selected.astype(float).dropna()
    omitted = omitted.astype(float).dropna()
    pooled_var = (
        (selected.var(ddof=1) + omitted.var(ddof=1)) / 2.0
    )
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        return np.nan
    return float((selected.mean() - omitted.mean()) / np.sqrt(pooled_var))


def _selection_diagnostics(
    df: pd.DataFrame,
    common_index: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiled = df.copy()
    profiled["spatial_cam_selected"] = profiled.index.isin(common_index)
    rows = []
    selected = profiled[profiled["spatial_cam_selected"]]
    omitted = profiled[~profiled["spatial_cam_selected"]]
    for subset_name, subset in [
        ("all_global_records", profiled),
        ("spatial_cam_selected", selected),
        ("spatial_cam_omitted", omitted),
    ]:
        rows.append({
            "subset": subset_name,
            "n": int(len(subset)),
            "fraction_of_global_records": float(len(subset) / len(profiled)),
            "mean_feature_drift": float(subset["feature_drift"].mean()),
            "mean_accuracy_drop": float(subset["drop"].mean()),
            "smd_feature_drift_selected_vs_omitted": (
                _standardized_mean_difference(
                    selected["feature_drift"], omitted["feature_drift"]
                )
                if subset_name == "all_global_records" else np.nan
            ),
            "smd_accuracy_drop_selected_vs_omitted": (
                _standardized_mean_difference(selected["drop"], omitted["drop"])
                if subset_name == "all_global_records" else np.nan
            ),
        })

    coverage = (
        profiled.groupby(["perturbation", "severity_control"], as_index=False)
        .agg(
            n_global=("spatial_cam_selected", "size"),
            n_spatial_cam=("spatial_cam_selected", "sum"),
        )
    )
    coverage["coverage_fraction"] = (
        coverage["n_spatial_cam"] / coverage["n_global"]
    )
    return pd.DataFrame(rows), coverage


def _subset_primary_sensitivity(
    df: pd.DataFrame,
    *,
    subset_name: str,
    subset_query: str,
) -> pd.DataFrame:
    """Primary drift model on a named subset used for leakage sensitivity."""
    sub = df.query(subset_query).copy()
    sub_base_cat = ["backbone", "perturbation", "severity_control"]
    if sub["dataset"].nunique() > 1:
        sub_base_cat = ["dataset", *sub_base_cat]
    base_r2, base_n = _fit_r2(sub, "drop", sub_base_cat, [])
    drift_r2, drift_n = _fit_r2(sub, "drop", sub_base_cat, ["feature_drift"])
    if base_n != drift_n:
        raise RuntimeError(
            f"{subset_name} sensitivity models use different samples: "
            f"{base_n} versus {drift_n}"
        )
    partial_r, partial_p, partial_n = _partial_correlation(
        sub, "feature_drift", sub_base_cat
    )
    return pd.DataFrame([{
        "subset": subset_name,
        "dataset_filter": subset_query,
        "controls": "+".join(sub_base_cat),
        "base_r2": base_r2,
        "drift_r2": drift_r2,
        "delta_r2": drift_r2 - base_r2,
        "partial_r": partial_r,
        "partial_p": partial_p,
        "n": partial_n,
        "n_datasets": int(sub["dataset"].nunique()),
        "datasets": ",".join(sorted(sub["dataset"].astype(str).unique())),
    }])


def run(save: bool = True) -> pd.DataFrame:
    input_path = require_csv("exp6_multilevel_table.csv")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['drop', 'feature_drift']).copy()
    df['severity_control'] = df['severity'].astype(str)

    base_cat = ['dataset', 'backbone', 'perturbation', 'severity_control']
    specs = [
        ('M0: dataset+backbone+perturbation+severity', []),
        ('M1: + feature drift', ['feature_drift']),
        ('M2: + feature geometry', ['feature_drift', 'delta_intra', 'inter_collapse', 'delta_fgcs']),
        ('M3: + spatial metrics', [
            'feature_drift', 'delta_intra', 'inter_collapse', 'delta_fgcs',
            'sgi_clean', 'spatial_instability_hungarian',
        ]),
        ('M4: + CAM metrics', [
            'feature_drift', 'delta_intra', 'inter_collapse', 'delta_fgcs',
            'sgi_clean', 'spatial_instability_hungarian',
            'cam_shift_jsd', 'cam_entropy_clean',
        ]),
    ]

    full_numeric = sorted({c for _, nums in specs for c in nums})
    common_cols = ['drop'] + base_cat + full_numeric
    common = df[common_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    full_base_r2, full_base_n = _fit_r2(df, "drop", base_cat, [])
    full_drift_r2, full_drift_n = _fit_r2(
        df, "drop", base_cat, ["feature_drift"]
    )
    if full_base_n != full_drift_n:
        raise RuntimeError(
            "Full-record primary models use different samples: "
            f"{full_base_n} versus {full_drift_n}"
        )
    full_partial_r, full_partial_p, full_partial_n = _partial_correlation(
        df, "feature_drift", base_cat
    )
    full_primary = pd.DataFrame([
        {
            "model": "P0: dataset+backbone+perturbation+severity",
            "r2": full_base_r2,
            "delta_r2": full_base_r2,
            "n": full_base_n,
        },
        {
            "model": "P1: + feature drift",
            "r2": full_drift_r2,
            "delta_r2": full_drift_r2 - full_base_r2,
            "n": full_drift_n,
        },
    ])

    rows = []
    prev = 0.0
    for label, nums in specs:
        r2, n = _fit_r2(common, 'drop', base_cat, nums)
        rows.append({
            'model': label,
            'controls': '+'.join(base_cat),
            'numeric_terms': '+'.join(nums) if nums else '',
            'r2': r2,
            'delta_r2': r2 - prev if np.isfinite(r2) else np.nan,
            'n': n,
            'complete_case': True,
        })
        if np.isfinite(r2):
            prev = r2

    out = pd.DataFrame(rows)
    partial = _severity_partial_table(df)
    complete_partial_r, complete_partial_p, complete_partial_n = (
        _partial_correlation(common, "feature_drift", base_cat)
    )
    complete_partial = pd.DataFrame([{
        "metric": "feature_drift",
        "r_partial_complete_case": complete_partial_r,
        "p_partial_complete_case": complete_partial_p,
        "n": complete_partial_n,
        "controls": "+".join(base_cat),
        "subset": "complete_case_for_M0_to_M4",
    }])
    full_partial = pd.DataFrame([{
        "metric": "feature_drift",
        "r_partial_full_record": full_partial_r,
        "p_partial_full_record": full_partial_p,
        "n": full_partial_n,
        "controls": "+".join(base_cat),
        "subset": "all_records_with_global_geometry",
    }])
    selection_summary, selection_coverage = _selection_diagnostics(
        df, common.index
    )
    non_patch_sensitivity = _subset_primary_sensitivity(
        df,
        subset_name="WRD25_only_non_patch_sensitivity",
        subset_query="dataset == 'WRD25'",
    )
    if save:
        full_primary.to_csv(
            V4_CSV / "exp7_full_record_primary_r2.csv", index=False
        )
        full_partial.to_csv(
            V4_CSV / "exp7_full_record_partial_correlation.csv", index=False
        )
        out.to_csv(V4_CSV / 'exp7_controlled_hierarchical_r2.csv', index=False)
        partial.to_csv(
            V4_CSV / "exp7_partial_correlations_severity.csv", index=False
        )
        complete_partial.to_csv(
            V4_CSV / "exp7_complete_case_partial_correlation.csv", index=False
        )
        selection_summary.to_csv(
            V4_CSV / "exp7_spatial_cam_selection_summary.csv", index=False
        )
        selection_coverage.to_csv(
            V4_CSV / "exp7_spatial_cam_selection_coverage.csv", index=False
        )
        non_patch_sensitivity.to_csv(
            V4_CSV / "exp7_non_patch_sensitivity.csv", index=False
        )
        write_provenance(
            "exp7_controlled_hierarchical",
            [
                V4_CSV / "exp7_full_record_primary_r2.csv",
                V4_CSV / "exp7_full_record_partial_correlation.csv",
                V4_CSV / "exp7_controlled_hierarchical_r2.csv",
                V4_CSV / "exp7_partial_correlations_severity.csv",
                V4_CSV / "exp7_complete_case_partial_correlation.csv",
                V4_CSV / "exp7_spatial_cam_selection_summary.csv",
                V4_CSV / "exp7_spatial_cam_selection_coverage.csv",
                V4_CSV / "exp7_non_patch_sensitivity.csv",
            ],
            protocol="metadata_and_categorical_severity_controls_v2",
            parameters={
                "categorical_controls": [
                    "dataset", "backbone", "perturbation", "severity"
                ],
                "residualization": "one_hot_linear_regression",
                "complete_case_hierarchy": True,
                "complete_case_partial_correlation": True,
                "full_record_primary_model": True,
                "spatial_cam_selection_diagnostics": True,
                "non_patch_sensitivity": "WRD25_only",
            },
            inputs=[input_path],
        )
    return out


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    print(run(save=True).to_string(index=False))
