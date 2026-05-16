#!/usr/bin/env python3
"""
Experiment: controlled hierarchical R².

Adds representation groups after explicit dataset/backbone/perturbation/severity
controls, addressing reviewer concerns about severity and grouped dependence.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from wood_spatial.config import V4_CSV

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


def run(save: bool = True) -> pd.DataFrame:
    df = pd.read_csv(V4_CSV / 'exp6_multilevel_table.csv')
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
    if save:
        out.to_csv(V4_CSV / 'exp7_controlled_hierarchical_r2.csv', index=False)
    return out


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    print(run(save=True).to_string(index=False))
