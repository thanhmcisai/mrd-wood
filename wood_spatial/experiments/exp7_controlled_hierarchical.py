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
    if save:
        out.to_csv(V4_CSV / 'exp7_controlled_hierarchical_r2.csv', index=False)
        partial.to_csv(
            V4_CSV / "exp7_partial_correlations_severity.csv", index=False
        )
        write_provenance(
            "exp7_controlled_hierarchical",
            [
                V4_CSV / "exp7_controlled_hierarchical_r2.csv",
                V4_CSV / "exp7_partial_correlations_severity.csv",
            ],
            protocol="metadata_and_categorical_severity_controls_v1",
            parameters={
                "categorical_controls": [
                    "dataset", "backbone", "perturbation", "severity"
                ],
                "residualization": "one_hot_linear_regression",
                "complete_case_hierarchy": True,
            },
            inputs=[input_path],
        )
    return out


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    print(run(save=True).to_string(index=False))
