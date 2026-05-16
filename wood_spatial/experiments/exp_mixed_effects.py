#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mixed-effects and grouped-robustness analysis for feature drift.

Primary models use statsmodels MixedLM:
  1. Random intercept + random feature-drift slope by backbone.
  2. Crossed random intercepts for dataset and backbone, with perturbation and
     severity fixed effects.

If statsmodels is missing, the script installs it automatically. Dependency-free
fixed-effects OLS and clustered bootstrap are still saved as supplementary
robustness checks.
"""
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import linregress

from wood_spatial.config import V4_CSV

OUT = V4_CSV


def ensure_statsmodels():
    try:
        import statsmodels.formula.api as smf  # noqa: F401
        return True, None
    except ImportError as exc:
        print(f"statsmodels missing: {exc}")
        print("Installing statsmodels with pip ...", flush=True)
        cmd = [sys.executable, '-m', 'pip', 'install', '-q', 'statsmodels']
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            return False, f'pip install statsmodels failed with code {proc.returncode}'
        try:
            import statsmodels.formula.api as smf  # noqa: F401
            return True, None
        except Exception as exc2:
            return False, str(exc2)


def fit_mixed_model(df, name, formula, groups, re_formula=None, vc_formula=None):
    import statsmodels.formula.api as smf

    print(f"\n=== MixedLM: {name} ===", flush=True)
    md = smf.mixedlm(
        formula,
        df,
        groups=groups,
        re_formula=re_formula,
        vc_formula=vc_formula,
    )
    last_exc = None
    for method in ['lbfgs', 'bfgs', 'cg', 'powell', 'nm']:
        try:
            result = md.fit(method=method, maxiter=2000, reml=False, disp=False)
            print(result.summary())
            return result, method, None
        except Exception as exc:
            last_exc = exc
            print(f"  method={method} failed: {exc}", flush=True)
    return None, None, str(last_exc)


def mixed_summary_row(df, name, formula, result, method, error=None):
    row = {
        'model': name,
        'formula': formula,
        'optimizer': method or '',
        'converged': bool(getattr(result, 'converged', False)) if result is not None else False,
        'error': error or '',
        'n': int(len(df)),
        'aic': np.nan,
        'bic': np.nan,
        'loglike': np.nan,
        'feature_beta': np.nan,
        'feature_se': np.nan,
        'feature_z': np.nan,
        'feature_p': np.nan,
        'feature_ci_lo': np.nan,
        'feature_ci_hi': np.nan,
    }
    if result is None:
        return row

    row['aic'] = float(getattr(result, 'aic', np.nan))
    row['bic'] = float(getattr(result, 'bic', np.nan))
    row['loglike'] = float(getattr(result, 'llf', np.nan))
    if 'feature_drift' in result.fe_params.index:
        ci = result.conf_int()
        beta = float(result.fe_params['feature_drift'])
        se = float(result.bse_fe['feature_drift'])
        row['feature_beta'] = beta
        row['feature_se'] = se
        row['feature_z'] = beta / max(se, 1e-12)
        row['feature_p'] = float(result.pvalues['feature_drift'])
        if 'feature_drift' in ci.index:
            row['feature_ci_lo'] = float(ci.loc['feature_drift', 0])
            row['feature_ci_hi'] = float(ci.loc['feature_drift', 1])
    return row


def save_fixed_effects(name, result):
    if result is None:
        return
    ci = result.conf_int()
    rows = []
    for param, value in result.fe_params.items():
        rows.append({
            'model': name,
            'term': param,
            'estimate': float(value),
            'se': float(result.bse_fe[param]) if param in result.bse_fe.index else np.nan,
            'p': float(result.pvalues[param]) if param in result.pvalues.index else np.nan,
            'ci_lo': float(ci.loc[param, 0]) if param in ci.index else np.nan,
            'ci_hi': float(ci.loc[param, 1]) if param in ci.index else np.nan,
        })
    pd.DataFrame(rows).to_csv(
        OUT / f'exp_mixed_{name.lower().replace(" ", "_")}_fixed_effects.csv',
        index=False,
    )


def save_variance_components(name, result):
    if result is None:
        return
    rows = []
    try:
        cov_re = result.cov_re
        for rname in cov_re.index:
            for cname in cov_re.columns:
                rows.append({
                    'model': name,
                    'component': f'random_cov:{rname}:{cname}',
                    'value': float(cov_re.loc[rname, cname]),
                })
    except Exception:
        pass
    try:
        for i, value in enumerate(np.asarray(result.vcomp).ravel()):
            rows.append({
                'model': name,
                'component': f'variance_component_{i}',
                'value': float(value),
            })
    except Exception:
        pass
    if rows:
        pd.DataFrame(rows).to_csv(
            OUT / f'exp_mixed_{name.lower().replace(" ", "_")}_variance_components.csv',
            index=False,
        )


def run_statsmodels_models(df):
    has_statsmodels, error = ensure_statsmodels()
    rows = []
    if not has_statsmodels:
        print(f"statsmodels unavailable: {error}", flush=True)
        pd.DataFrame([{
            'model': 'statsmodels_unavailable',
            'formula': '',
            'optimizer': '',
            'converged': False,
            'error': error,
            'n': int(len(df)),
        }]).to_csv(OUT / 'exp_mixed_statsmodels_summary.csv', index=False)
        return rows

    formula_a = 'drop ~ feature_drift'
    result_a, method_a, error_a = fit_mixed_model(
        df,
        name='backbone_random_slope',
        formula=formula_a,
        groups=df['backbone'],
        re_formula='~feature_drift',
    )
    rows.append(mixed_summary_row(df, 'backbone_random_slope', formula_a, result_a, method_a, error_a))
    save_fixed_effects('backbone_random_slope', result_a)
    save_variance_components('backbone_random_slope', result_a)

    df['_all_group'] = 'all'
    formula_b = 'drop ~ feature_drift + C(perturbation) + C(severity)'
    result_b, method_b, error_b = fit_mixed_model(
        df,
        name='crossed_dataset_backbone',
        formula=formula_b,
        groups=df['_all_group'],
        re_formula='0',
        vc_formula={
            'dataset': '0 + C(dataset)',
            'backbone': '0 + C(backbone)',
        },
    )
    rows.append(mixed_summary_row(df, 'crossed_dataset_backbone', formula_b, result_b, method_b, error_b))
    save_fixed_effects('crossed_dataset_backbone', result_b)
    save_variance_components('crossed_dataset_backbone', result_b)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'exp_mixed_statsmodels_summary.csv', index=False)
    print("\n=== statsmodels MixedLM summary rows ===")
    print(out.to_string(index=False))
    print("Saved: exp_mixed_statsmodels_summary.csv")
    return rows


def run_icc_and_per_backbone(df):
    groups = [grp['drop'].values for _, grp in df.groupby('backbone')]
    f_stat, p_anova = scipy_stats.f_oneway(*groups)

    n_per_group = [len(g) for g in groups]
    k = len(groups)
    n_total = sum(n_per_group)
    grand_mean = df['drop'].mean()
    ss_between = sum(n * (g.mean() - grand_mean) ** 2 for n, g in zip(n_per_group, groups))
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_total - k)
    n0 = (n_total - sum(ni ** 2 / n_total for ni in n_per_group)) / (k - 1)
    icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)

    print(f"\n=== ICC (backbone grouping) ===")
    print(f"ICC = {icc:.4f}")
    print(f"F({k-1}, {n_total-k}) = {f_stat:.2f}, p = {p_anova:.3e}")

    pd.DataFrame([{
        'icc': icc,
        'f_oneway': f_stat,
        'p_oneway': p_anova,
        'n_total': n_total,
        'n_backbones': k,
    }]).to_csv(OUT / 'exp_mixed_icc.csv', index=False)

    rows = []
    for bb, grp in df.groupby('backbone'):
        slope, intercept, r_val, p_val, se = linregress(grp['feature_drift'], grp['drop'])
        rows.append({
            'backbone': bb,
            'n': len(grp),
            'slope': slope,
            'intercept': intercept,
            'r': r_val,
            'r_squared': r_val ** 2,
            'p': p_val,
            'se': se,
        })
    df_bb = pd.DataFrame(rows)
    df_bb.to_csv(OUT / 'exp_mixed_perbackbone_ols.csv', index=False)
    print("\n=== Per-backbone OLS ===")
    print(df_bb[['backbone', 'slope', 'r', 'r_squared', 'p']].to_string(index=False))


def run_fixed_effects_ols(df):
    controls = pd.get_dummies(
        df[['dataset', 'backbone', 'perturbation', 'severity']].astype(str),
        drop_first=True,
        dtype=float,
    )
    x_feature = df[['feature_drift']].astype(float).reset_index(drop=True)
    X = pd.concat(
        [pd.Series(1.0, index=df.index, name='intercept'), x_feature, controls.reset_index(drop=True)],
        axis=1,
    ).astype(float)
    y = df['drop'].astype(float).to_numpy()
    X_np = X.to_numpy()

    beta, _, rank, _ = np.linalg.lstsq(X_np, y, rcond=None)
    y_hat = X_np @ beta
    resid = y - y_hat
    n, p = X_np.shape
    dof = max(n - p, 1)
    sigma2 = float((resid @ resid) / dof)
    xtx_inv = np.linalg.pinv(X_np.T @ X_np)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0))
    t_vals = beta / np.maximum(se, 1e-12)
    p_vals = 2 * scipy_stats.t.sf(np.abs(t_vals), dof)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum(resid ** 2))
    r2 = 1.0 - ss_res / ss_tot

    feature_idx = list(X.columns).index('feature_drift')
    row = {
        'model': 'drop ~ feature_drift + dataset + backbone + perturbation + severity',
        'n': int(n),
        'parameters': int(p),
        'rank': int(rank),
        'r_squared': float(r2),
        'feature_beta': float(beta[feature_idx]),
        'feature_se': float(se[feature_idx]),
        'feature_t': float(t_vals[feature_idx]),
        'feature_p': float(p_vals[feature_idx]),
    }
    pd.DataFrame([row]).to_csv(OUT / 'exp_mixed_fixed_effects.csv', index=False)
    print("\n=== Fixed-effects OLS robustness check ===")
    print(pd.DataFrame([row]).to_string(index=False))


def run_clustered_bootstrap(df, boot_b=2000):
    rng = np.random.default_rng(42)
    cluster_cols = ['dataset', 'backbone', 'perturbation']
    clusters = list(df.groupby(cluster_cols).groups.keys())
    cluster_frames = {key: grp.copy() for key, grp in df.groupby(cluster_cols, sort=False)}

    rows = []
    for b in range(boot_b):
        sampled = rng.choice(len(clusters), size=len(clusters), replace=True)
        sample = pd.concat([cluster_frames[clusters[int(i)]] for i in sampled], ignore_index=True)
        if sample['feature_drift'].nunique() < 2 or sample['drop'].nunique() < 2:
            continue
        slope, intercept, r_val, _, _ = linregress(sample['feature_drift'], sample['drop'])
        rows.append({
            'bootstrap_id': b,
            'slope': slope,
            'intercept': intercept,
            'r': r_val,
            'r_squared': r_val ** 2,
            'n': int(len(sample)),
        })

    df_boot = pd.DataFrame(rows)
    df_boot.to_csv(OUT / 'exp_mixed_cluster_bootstrap.csv', index=False)
    if len(df_boot):
        summary = {
            'cluster_definition': '+'.join(cluster_cols),
            'n_clusters': int(len(clusters)),
            'bootstrap_B': int(len(df_boot)),
            'slope_mean': float(df_boot['slope'].mean()),
            'slope_ci_lo': float(df_boot['slope'].quantile(0.025)),
            'slope_ci_hi': float(df_boot['slope'].quantile(0.975)),
            'r_mean': float(df_boot['r'].mean()),
            'r_ci_lo': float(df_boot['r'].quantile(0.025)),
            'r_ci_hi': float(df_boot['r'].quantile(0.975)),
        }
    else:
        summary = {
            'cluster_definition': '+'.join(cluster_cols),
            'n_clusters': int(len(clusters)),
            'bootstrap_B': 0,
            'slope_mean': np.nan,
            'slope_ci_lo': np.nan,
            'slope_ci_hi': np.nan,
            'r_mean': np.nan,
            'r_ci_lo': np.nan,
            'r_ci_hi': np.nan,
        }
    pd.DataFrame([summary]).to_csv(OUT / 'exp_mixed_cluster_bootstrap_summary.csv', index=False)
    print("\n=== Clustered bootstrap ===")
    print(pd.DataFrame([summary]).to_string(index=False))


def main():
    df = pd.read_csv(OUT / 'exp1b_feature_geometry.csv')
    df = df.dropna(subset=['drop', 'feature_drift', 'dataset', 'backbone', 'perturbation', 'severity']).copy()
    df['severity'] = df['severity'].astype(str)
    print(f"Data: {len(df)} rows, {df['backbone'].nunique()} backbones, "
          f"{df['dataset'].nunique()} datasets, {df['perturbation'].nunique()} perturbations")

    run_statsmodels_models(df)
    run_icc_and_per_backbone(df)
    run_fixed_effects_ols(df)
    run_clustered_bootstrap(df)

    print(f"\nSaved mixed-effects outputs to {OUT}")


if __name__ == '__main__':
    main()
