#!/usr/bin/env python3
"""Audit numerical claims and figure provenance for the submitted paper."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "results" / "csv"
FIG_DIR = ROOT / "results" / "figures"
MAIN_TEX = ROOT / "main.tex"


@dataclass
class AuditItem:
    severity: str
    item: str
    message: str


@dataclass
class MetricRecord:
    metric_id: str
    metric_family: str
    evidence_status: str
    value: float
    display_value: str
    source: str
    source_sha256: str
    source_rows: int
    protocol: str
    scientific_role: str
    paper_label: str


def _read(name: str) -> pd.DataFrame:
    path = CSV_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _row_value(name: str, column: str, **selector: object) -> float:
    df = _read(name)
    for key, value in selector.items():
        df = df[df[key].eq(value)]
    if len(df) != 1:
        raise ValueError(f"{name}: selector {selector} returned {len(df)} rows")
    return float(df.iloc[0][column])


def _single_value(name: str, column: str) -> float:
    df = _read(name)
    if len(df) != 1:
        raise ValueError(f"{name}: expected one row, found {len(df)}")
    return float(df.iloc[0][column])


def _rounded(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _metric_specs() -> list[tuple[str, Callable[[], float], int, str, str, str, str]]:
    return [
        (
            "drift_raw_same_space",
            lambda: float(_read("exp1b_feature_geometry.csv")[["feature_drift", "drop"]].corr().iloc[0, 1]),
            3,
            "exp1b_feature_geometry.csv",
            "Pooled Tier-A same-space condition records; Pearson correlation.",
            "Upper-bound association because drift and kNN failure share a feature space.",
            "upper bound",
        ),
        (
            "drift_partial_perturbation_identity",
            lambda: _row_value(
                "exp7_partial_correlations.csv", "r_partial",
                metric="feature_drift",
            ),
            3,
            "exp7_partial_correlations.csv",
            "Residualization on perturbation identity only.",
            "Sensitivity analysis; weaker control than the primary model.",
            "controlling perturbation identity",
        ),
        (
            "drift_partial_metadata_severity",
            lambda: _row_value(
                "exp7_partial_correlations_severity.csv", "r_partial_severity",
                metric="feature_drift",
            ),
            3,
            "exp7_partial_correlations_severity.csv",
            "Residualization on dataset, backbone, perturbation family, and severity.",
            "Primary controlled evidence for the drift-drop relationship.",
            "partial $r=0.623$",
        ),
        (
            "drift_incremental_r2",
            lambda: _row_value(
                "exp7_controlled_hierarchical_r2.csv", "delta_r2",
                model="M1: + feature drift",
            ),
            3,
            "exp7_controlled_hierarchical_r2.csv",
            "Complete-case hierarchical model after metadata and severity controls.",
            "Primary incremental explanatory contribution.",
            "\\Delta R^2=0.032",
        ),
        (
            "cross_space_partial_r",
            lambda: _single_value(
                "exp_cross_space_drift_summary.csv",
                "pooled_cross_space_partial_r",
            ),
            3,
            "exp_cross_space_drift_summary.csv",
            "Off-diagonal drift-space/decision-space pairs with metadata and severity controls.",
            "Small transferable diagnostic component independent of same-space matching.",
            "cross-space partial correlation",
        ),
        (
            "tierc_bfs_fsdm_heldout_species_source_acc",
            lambda: _row_value(
                "exp_source_vs_species_probe_summary.csv",
                "source_acc_leave_one_species_out_mean",
                pair="BFS46<->FSDM41",
            ),
            3,
            "exp_source_vs_species_probe_summary.csv",
            "Balanced leave-one-species-out nearest-centroid source prediction.",
            "Primary held-out evidence that acquisition-source identity transfers across species.",
            "mean accuracy 0.923",
        ),
        (
            "tierc_dtsr_woodauth_heldout_species_source_acc",
            lambda: _row_value(
                "exp_source_vs_species_probe_summary.csv",
                "source_acc_leave_one_species_out_mean",
                pair="DTSR14<->WOODAUTH",
            ),
            3,
            "exp_source_vs_species_probe_summary.csv",
            "Balanced leave-one-species-out nearest-centroid source prediction.",
            "Secondary held-out evidence that acquisition-source identity transfers across species.",
            "and 0.940",
        ),
        (
            "tierc_bfs_fsdm_cross_source_species_acc",
            lambda: _row_value(
                "exp_source_vs_species_probe_summary.csv",
                "cross_source_species_acc_mean",
                pair="BFS46<->FSDM41",
            ),
            3,
            "exp_source_vs_species_probe_summary.csv",
            "Directed cross-source cosine nearest-centroid species transfer.",
            "Class-conditional counterpart to held-out source prediction.",
            "Cross-source species recognition is",
        ),
        (
            "tierc_dtsr_woodauth_cross_source_species_acc",
            lambda: _row_value(
                "exp_source_vs_species_probe_summary.csv",
                "cross_source_species_acc_mean",
                pair="DTSR14<->WOODAUTH",
            ),
            3,
            "exp_source_vs_species_probe_summary.csv",
            "Directed cross-source cosine nearest-centroid species transfer.",
            "Class-conditional counterpart to held-out source prediction.",
            "and 0.278",
        ),
        (
            "condition_level_paired_drift_auc",
            lambda: _row_value(
                "exp8_detector_auc.csv", "auc_roc", detector="feature_drift"
            ),
            3,
            "exp8_detector_auc.csv",
            "Condition-level diagnostic benchmark over 1,596 Tier-A records.",
            "Oracle diagnostic AUC; not a deployable reference-bank score.",
            "condition-level ROC-AUC",
        ),
        (
            "batch_level_paired_drift_auc",
            lambda: _row_value(
                "exp10_reference_monitor_auc.csv",
                "auc_roc",
                detector="paired_feature_drift_oracle",
            ),
            3,
            "exp10_reference_monitor_auc.csv",
            "Batch-level monitor benchmark over 4,256 records.",
            "Paired-clean oracle used only as an upper comparator for the monitor.",
            "batch-level ROC-AUC",
        ),
        (
            "batch_level_mmd_auc",
            lambda: _row_value(
                "exp10_reference_monitor_auc.csv",
                "auc_roc",
                detector="ref_mmd_rbf",
            ),
            3,
            "exp10_reference_monitor_auc.csv",
            "Batch-level capped reference-bank monitor benchmark.",
            "Primary label-free binary acquisition-mismatch detection result.",
            "RBF-MMD reaches batch-level ROC-AUC",
        ),
        (
            "tierc_bfs_fsdm_full_mmd",
            lambda: _single_value(
                "exp_mmd_confound_summary.csv", "large_pair_raw_mmd2"
            ),
            3,
            "exp_mmd_confound_summary.csv",
            "Full-feature, per-pair median-bandwidth Tier-C decomposition.",
            "Marginal shift magnitude for cross-source comparison; not an alarm score.",
            "mean MMD magnitude (0.102)",
        ),
        (
            "tierc_dtsr_woodauth_full_mmd",
            lambda: _single_value(
                "exp_mmd_confound_summary.csv", "small_pair_raw_mmd2"
            ),
            3,
            "exp_mmd_confound_summary.csv",
            "Full-feature, per-pair median-bandwidth Tier-C decomposition.",
            "Marginal shift magnitude for cross-source comparison; not an alarm score.",
            "full-feature decomposition confirms",
        ),
        (
            "tierc_bfs_fsdm_shared_gamma_mmd",
            lambda: _row_value(
                "exp_mmd_gamma_sensitivity_summary.csv",
                "worst_pair_mmd",
                policy="global_median",
            ),
            3,
            "exp_mmd_gamma_sensitivity_summary.csv",
            "Shared global RBF bandwidth across real-shift comparisons.",
            "Bandwidth sensitivity control, not the primary magnitude estimate.",
            "Shared median & 0.097",
        ),
        (
            "tierc_dtsr_woodauth_shared_gamma_mmd",
            lambda: _row_value(
                "exp_mmd_gamma_sensitivity_summary.csv",
                "milder_pair_mmd",
                policy="global_median",
            ),
            3,
            "exp_mmd_gamma_sensitivity_summary.csv",
            "Shared global RBF bandwidth across real-shift comparisons.",
            "Bandwidth sensitivity control, not the primary magnitude estimate.",
            "Shared median & 0.097 & 0.184",
        ),
        (
            "matched_four_species_mmd",
            lambda: _single_value(
                "exp_mmd_confound_summary.csv",
                "large_pair_matched_4class_mmd2_mean",
            ),
            3,
            "exp_mmd_confound_summary.csv",
            "Twenty four-species draws from BFS46/FSDM41.",
            "Shared-class-count sensitivity analysis.",
            "mean MMD from 0.102 to 0.209",
        ),
        (
            "severity_rho_predefined_order",
            lambda: _single_value(
                "exp_monitor_on_real_shift_summary.csv",
                "severity_spearman_by_condition_mean",
            ),
            3,
            "exp_monitor_on_real_shift_summary.csv",
            "Five coarse groups ordered a priori as clean/Tier-B/Tier-D/Tier-C.",
            "Agreement with intended tier ordering.",
            "predefined severity ordering is $\\rho=0.300$",
        ),
        (
            "severity_rho_eight_measured_groups",
            lambda: _single_value(
                "exp_monitor_severity_dissociation_summary.csv",
                "severity_spearman",
            ),
            3,
            "exp_monitor_severity_dissociation_summary.csv",
            "Eight condition groups using the capped reference-bank score and measured failure.",
            "Primary evidence that MMD magnitude is not a calibrated severity gauge.",
            "eight condition groups",
        ),
        (
            "severity_rho_six_group_bandwidth_control",
            lambda: _row_value(
                "exp_mmd_gamma_sensitivity_summary.csv",
                "condition_spearman",
                policy="global_median",
            ),
            3,
            "exp_mmd_gamma_sensitivity_summary.csv",
            "Six clean and real-shift groups under a shared global bandwidth.",
            "Coarser aggregation used only in the bandwidth sensitivity analysis.",
            "$\\rho=0.548$ over eight condition groups and $\\rho=0.486$",
        ),
        (
            "matched_bfs_fsdm_accuracy",
            lambda: _row_value(
                "exp_matched_class_dissociation_summary.csv",
                "mean_accuracy",
                pair="BFS46<->FSDM41",
            ),
            3,
            "exp_matched_class_dissociation_summary.csv",
            "Mean kNN-5 transfer accuracy over twenty matched four-species draws.",
            "Class-conditional transfer after controlling shared-class count.",
            "accuracies are 0.195 and 0.320",
        ),
        (
            "matched_dtsr_woodauth_accuracy",
            lambda: _row_value(
                "exp_matched_class_dissociation_summary.csv",
                "mean_accuracy",
                pair="DTSR14<->WOODAUTH",
            ),
            3,
            "exp_matched_class_dissociation_summary.csv",
            "Fixed four-species cross-source comparison.",
            "Reference class-conditional transfer accuracy.",
            "accuracies are 0.195 and 0.320",
        ),
    ]


METRIC_CLASSIFICATION = {
    "drift_raw_same_space": ("drift_drop_association", "upper_bound"),
    "drift_partial_perturbation_identity": ("drift_drop_association", "sensitivity"),
    "drift_partial_metadata_severity": ("drift_drop_association", "primary"),
    "drift_incremental_r2": ("drift_drop_association", "primary"),
    "cross_space_partial_r": ("drift_drop_association", "transfer_control"),
    "tierc_bfs_fsdm_heldout_species_source_acc": ("source_species_dissociation", "primary"),
    "tierc_dtsr_woodauth_heldout_species_source_acc": ("source_species_dissociation", "secondary_pair"),
    "tierc_bfs_fsdm_cross_source_species_acc": ("source_species_dissociation", "primary"),
    "tierc_dtsr_woodauth_cross_source_species_acc": ("source_species_dissociation", "secondary_pair"),
    "condition_level_paired_drift_auc": ("failure_detection_auc", "diagnostic_oracle"),
    "batch_level_paired_drift_auc": ("failure_detection_auc", "batch_oracle"),
    "batch_level_mmd_auc": ("failure_detection_auc", "primary_monitor"),
    "tierc_bfs_fsdm_full_mmd": ("tier_c_mmd_magnitude", "primary_full_feature"),
    "tierc_dtsr_woodauth_full_mmd": ("tier_c_mmd_magnitude", "primary_full_feature"),
    "tierc_bfs_fsdm_shared_gamma_mmd": ("tier_c_mmd_magnitude", "bandwidth_control"),
    "tierc_dtsr_woodauth_shared_gamma_mmd": ("tier_c_mmd_magnitude", "bandwidth_control"),
    "matched_four_species_mmd": ("tier_c_mmd_magnitude", "class_count_control"),
    "severity_rho_predefined_order": ("mmd_severity_ranking", "coarse_intended_order"),
    "severity_rho_eight_measured_groups": ("mmd_severity_ranking", "primary_measured_failure"),
    "severity_rho_six_group_bandwidth_control": ("mmd_severity_ranking", "bandwidth_control"),
    "matched_bfs_fsdm_accuracy": ("matched_class_transfer", "sampled_pair"),
    "matched_dtsr_woodauth_accuracy": ("matched_class_transfer", "fixed_reference_pair"),
}


FIGURE_SOURCES: dict[str, tuple[str, ...]] = {
    "fig2a_accuracy_heatmap.png": ("exp1_accuracy_matrix.csv",),
    "fig2b_severity_curves.png": ("exp1_accuracy_matrix.csv", "exp1_bootstrap_ci.csv"),
    "fig3_feature_geometry_failure.png": (
        "exp1b_feature_geometry.csv",
        "exp7_partial_correlations_severity.csv",
        "exp7_controlled_hierarchical_r2.csv",
    ),
    "fig5_nemenyi_cd_diagram.png": ("exp1_nemenyi_pvalues.csv",),
    "fig8_roc_failure_detection.png": ("exp8_detector_auc.csv", "exp8_operating_points.csv"),
    "cross_space_drift.png": ("exp_cross_space_drift_matrix.csv", "exp_cross_space_drift_summary.csv"),
    "source_vs_species_probe.png": ("exp_source_vs_species_probe_by_backbone.csv",),
    "tierc_cross_source_shift.png": (
        "exp_tierc_cross_source_by_cell.csv",
        "exp_tierc_cross_source_transfer.csv",
    ),
    "cross_magnification_asymmetry.png": (
        "exp5_crossmag_asymmetry_by_pair.csv",
        "exp5_crossmag_drift_drop.csv",
    ),
    "monitor_on_real_shift.png": (
        "exp_monitor_on_real_shift_by_condition.csv",
    ),
    "monitor_severity_dissociation.png": (
        "exp_monitor_severity_dissociation_by_condition.csv",
    ),
    "mmd_confound_and_class_count.png": (
        "exp_mmd_confound_terms.csv",
        "exp_mmd_confound_summary.csv",
    ),
    "matched_class_dissociation.png": (
        "exp_matched_class_dissociation_by_seed.csv",
        "exp_matched_class_dissociation_summary.csv",
    ),
    "fig_competitor_switching.png": ("exp_competitor_switching_by_condition.csv",),
}


def _check(condition: bool, item: str, ok: str, fail: str, results: list[AuditItem]) -> None:
    results.append(AuditItem("OK" if condition else "FAIL", item, ok if condition else fail))


def _one_hot(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    try:
        encoder = OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        )
    except TypeError:
        encoder = OneHotEncoder(
            drop="first", sparse=False, handle_unknown="ignore"
        )
    return encoder.fit_transform(frame[columns].astype(str))


def audit() -> tuple[list[AuditItem], list[MetricRecord]]:
    results: list[AuditItem] = []
    records: list[MetricRecord] = []
    tex = MAIN_TEX.read_text()

    for metric_id, loader, digits, source, protocol, role, paper_label in _metric_specs():
        try:
            value = loader()
            display = _rounded(value, digits)
            metric_family, evidence_status = METRIC_CLASSIFICATION[metric_id]
            records.append(MetricRecord(
                metric_id=metric_id,
                metric_family=metric_family,
                evidence_status=evidence_status,
                value=value,
                display_value=display,
                source=f"results/csv/{source}",
                source_sha256=_sha256(CSV_DIR / source),
                source_rows=len(_read(source)),
                protocol=protocol,
                scientific_role=role,
                paper_label=paper_label,
            ))
            _check(
                paper_label in tex,
                f"paper:{metric_id}",
                f"labelled locally as '{paper_label}' ({display})",
                f"missing protocol label '{paper_label}' for value {display}",
                results,
            )
            _check(
                display in tex,
                f"paper-value:{metric_id}",
                f"canonical rounded value {display} appears in main.tex",
                f"canonical rounded value {display} is absent from main.tex",
                results,
            )
        except Exception as exc:
            results.append(AuditItem("FAIL", f"metric:{metric_id}", str(exc)))

    # Recompute key identities instead of trusting summary CSVs.
    try:
        geom = _read("exp1b_feature_geometry.csv")
        partial = _row_value(
            "exp7_partial_correlations_severity.csv", "r_raw",
            metric="feature_drift",
        )
        raw = float(geom[["feature_drift", "drop"]].corr().iloc[0, 1])
        _check(
            len(geom) == 1596 and math.isclose(raw, partial, abs_tol=1e-12),
            "formula:raw_drift_correlation",
            f"recomputed from 1,596 rows: r={raw:.12f}",
            f"row count/correlation mismatch: n={len(geom)}, recomputed={raw}, saved={partial}",
            results,
        )
    except Exception as exc:
        results.append(AuditItem("FAIL", "formula:raw_drift_correlation", str(exc)))

    try:
        table = _read("exp6_multilevel_table.csv")
        controls = ["dataset", "backbone", "perturbation", "severity"]
        sub = table[["feature_drift", "drop", *controls]].dropna()
        X = _one_hot(sub, controls)
        drift_residual = (
            sub["feature_drift"].to_numpy()
            - LinearRegression().fit(X, sub["feature_drift"]).predict(X)
        )
        drop_residual = (
            sub["drop"].to_numpy()
            - LinearRegression().fit(X, sub["drop"]).predict(X)
        )
        actual = float(np.corrcoef(drift_residual, drop_residual)[0, 1])
        saved = _row_value(
            "exp7_partial_correlations_severity.csv",
            "r_partial_severity",
            metric="feature_drift",
        )
        _check(
            math.isclose(actual, saved, abs_tol=1e-12),
            "formula:severity_partial_correlation",
            f"recomputed metadata+severity partial r={actual:.12f}",
            f"recomputed partial r={actual} differs from saved {saved}",
            results,
        )
    except Exception as exc:
        results.append(AuditItem(
            "FAIL", "formula:severity_partial_correlation", str(exc)
        ))

    try:
        table = _read("exp6_multilevel_table.csv").copy()
        table["severity_control"] = table["severity"].astype(str)
        full_numeric = [
            "feature_drift", "delta_intra", "inter_collapse", "delta_fgcs",
            "sgi_clean", "spatial_instability_hungarian",
            "cam_shift_jsd", "cam_entropy_clean",
        ]
        controls = [
            "dataset", "backbone", "perturbation", "severity_control"
        ]
        common = table[["drop", *controls, *full_numeric]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        X0 = _one_hot(common, controls)
        drift = StandardScaler().fit_transform(common[["feature_drift"]])
        r2_0 = LinearRegression().fit(X0, common["drop"]).score(
            X0, common["drop"]
        )
        X1 = np.hstack([X0, drift])
        r2_1 = LinearRegression().fit(X1, common["drop"]).score(
            X1, common["drop"]
        )
        actual = float(r2_1 - r2_0)
        saved = _row_value(
            "exp7_controlled_hierarchical_r2.csv",
            "delta_r2",
            model="M1: + feature drift",
        )
        _check(
            math.isclose(actual, saved, abs_tol=1e-12),
            "formula:controlled_delta_r2",
            f"recomputed complete-case Delta R2={actual:.12f}",
            f"recomputed Delta R2={actual} differs from saved {saved}",
            results,
        )
    except Exception as exc:
        results.append(AuditItem("FAIL", "formula:controlled_delta_r2", str(exc)))

    try:
        terms = _read("exp_mmd_confound_terms.csv")
        reconstructed = terms["K_AA"] + terms["K_BB"] - 2.0 * terms["K_AB"]
        error = float(np.max(np.abs(reconstructed - terms["mmd2"])))
        _check(
            error < 2e-6,
            "formula:mmd_decomposition",
            f"K_AA + K_BB - 2*K_AB matches MMD2; max error={error:.2e}",
            f"MMD decomposition mismatch; max error={error:.3g}",
            results,
        )
    except Exception as exc:
        results.append(AuditItem("FAIL", "formula:mmd_decomposition", str(exc)))

    try:
        rows = _read("exp_source_vs_species_probe_by_backbone.csv")
        summary = _read("exp_source_vs_species_probe_summary.csv").set_index("pair")
        for pair, group in rows.groupby("pair"):
            actual = float(group["source_acc_leave_one_species_out"].mean())
            saved = float(summary.loc[pair, "source_acc_leave_one_species_out_mean"])
            minimum = float(group["source_acc_leave_one_species_out"].min())
            _check(
                math.isclose(actual, saved, abs_tol=1e-12) and minimum > 0.5,
                f"formula:heldout_source_probe:{pair}",
                f"backbone mean reproduces held-out-species source accuracy {actual:.6f}; min={minimum:.6f}",
                f"held-out source summary mismatch or a backbone is not above chance: mean={actual}, saved={saved}, min={minimum}",
                results,
            )
    except Exception as exc:
        results.append(AuditItem("FAIL", "formula:heldout_source_probe", str(exc)))

    try:
        scores = _read("exp_monitor_on_real_shift_scores.csv")
        threshold = _row_value(
            "exp10_reference_monitor_auc.csv",
            "best_threshold",
            detector="ref_mmd_rbf",
        )
        real = scores[scores["condition"].isin([
            "TierD_xmag",
            "TierC_DTSR14_WOODAUTH",
            "TierC_BFS46_FSDM41",
        ])]
        alarms = int((real["ref_mmd_rbf"] > threshold).sum())
        _check(
            len(real) == 70 and alarms == 60,
            "formula:real_shift_alarm_count",
            "capped monitor protocol reproduces 60/70 alarms",
            f"expected 60/70 alarms, found {alarms}/{len(real)}",
            results,
        )
    except Exception as exc:
        results.append(AuditItem("FAIL", "formula:real_shift_alarm_count", str(exc)))

    try:
        scores = _read("exp_monitor_on_real_shift_scores.csv")
        saved = _read("exp_monitor_on_real_shift_by_condition.csv").set_index(
            "condition"
        )
        means = scores.groupby("condition")["ref_mmd_rbf"].mean()
        error = float(np.max(np.abs(
            means.loc[saved.index].to_numpy()
            - saved["mean_ref_mmd_rbf"].to_numpy()
        )))
        _check(
            error < 1e-12,
            "formula:monitor_condition_means",
            "monitor figure/table means reproduce capped score records",
            f"monitor condition means differ from score records; max error={error}",
            results,
        )
    except Exception as exc:
        results.append(AuditItem("FAIL", "formula:monitor_condition_means", str(exc)))

    try:
        records_df = _read("exp_monitor_severity_dissociation_records.csv")
        saved = _read(
            "exp_monitor_severity_dissociation_by_condition.csv"
        ).set_index("condition")
        means = records_df.groupby("condition")[["mmd", "failure"]].mean()
        error = float(np.max(np.abs(
            means.loc[saved.index].to_numpy()
            - saved[["mmd", "failure"]].to_numpy()
        )))
        _check(
            error < 1e-12,
            "formula:severity_condition_means",
            "eight-group severity means reproduce capped record-level scores",
            f"severity condition means differ from records; max error={error}",
            results,
        )
    except Exception as exc:
        results.append(AuditItem("FAIL", "formula:severity_condition_means", str(exc)))

    try:
        matched = _read("exp_matched_class_dissociation_by_seed.csv")
        summary = _read("exp_matched_class_dissociation_summary.csv").set_index("pair")
        for pair in summary.index:
            row = matched[matched["pair"].eq(pair)]
            actual = float(row["mean_accuracy"].mean())
            saved = float(summary.loc[pair, "mean_accuracy"])
            _check(
                math.isclose(actual, saved, abs_tol=1e-12),
                f"formula:matched_accuracy:{pair}",
                f"seed-level mean reproduces summary accuracy {actual:.6f}",
                f"seed-level mean {actual} differs from summary {saved}",
                results,
            )
    except Exception as exc:
        results.append(AuditItem("FAIL", "formula:matched_accuracy", str(exc)))

    try:
        severity = _read("exp_monitor_severity_dissociation_summary.csv").iloc[0]
        regression = _read("exp_mmd_confound_regression.csv")
        regression = regression[
            regression["scope"].eq("eight_condition_groups")
        ].iloc[0]
        _check(
            math.isclose(
                float(severity["r2_mmd_plus_spread"]),
                float(regression["r2_mmd_plus_spread"]),
                abs_tol=1e-12,
            ),
            "formula:eight_group_regression_consistency",
            "severity summary and confound regression use the same capped eight-group protocol",
            "severity summary and confound regression are from different protocol versions",
            results,
        )
    except Exception as exc:
        results.append(AuditItem(
            "FAIL", "formula:eight_group_regression_consistency", str(exc)
        ))

    included = {
        Path(match).name
        for match in re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
            tex,
        )
    }
    for figure in sorted(included):
        path = FIG_DIR / figure
        _check(
            path.exists(),
            f"figure:{figure}",
            "figure exists",
            f"missing included figure {path}",
            results,
        )
        for source in FIGURE_SOURCES.get(figure, ()):
            source_path = CSV_DIR / source
            if not source_path.exists() or not path.exists():
                continue
            _check(
                path.stat().st_mtime + 5.0 >= source_path.stat().st_mtime,
                f"freshness:{figure}:{source}",
                "figure is not older than its source CSV",
                f"figure predates results/csv/{source}; regenerate it",
                results,
            )

    legacy_refs = []
    for module in (
        "exp_monitor_on_real_shift.py",
        "exp_tierc_cross_source_shift.py",
        "exp_source_vs_species_probe.py",
        "exp5_crossmag_asymmetry.py",
        "exp_cross_space_drift.py",
        "exp_monitor_severity_dissociation.py",
        "exp_mmd_gamma_sensitivity.py",
        "exp_mmd_confound_and_sign.py",
        "exp_matched_class_dissociation.py",
    ):
        text = (ROOT / "wood_spatial" / "experiments" / module).read_text()
        if "results_v4" in text:
            legacy_refs.append(module)
    _check(
        not legacy_refs,
        "provenance:legacy_result_fallbacks",
        "audited experiments contain no results_v4 fallback",
        f"legacy result fallback remains in: {', '.join(legacy_refs)}",
        results,
    )

    return results, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--strict", action="store_true", help="return non-zero on WARN as well as FAIL")
    args = parser.parse_args()

    results, records = audit()
    counts = {
        severity: sum(item.severity == severity for item in results)
        for severity in ("OK", "WARN", "FAIL")
    }
    print(
        "=== Paper result audit ===\n"
        f"OK={counts['OK']} WARN={counts['WARN']} FAIL={counts['FAIL']}"
    )
    for severity in ("FAIL", "WARN"):
        items = [item for item in results if item.severity == severity]
        if items:
            print(f"\n[{severity}]")
            for item in items:
                print(f"  {item.item}: {item.message}")

    if args.write_report:
        out_dir = ROOT / "results" / "audit"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(record) for record in records]).to_csv(
            out_dir / "paper_metric_registry.csv", index=False
        )
        (out_dir / "paper_result_audit.json").write_text(
            json.dumps(
                {
                    "counts": counts,
                    "results": [asdict(item) for item in results],
                    "metrics": [asdict(record) for record in records],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nWrote audit artifacts to {out_dir}")

    if counts["FAIL"] or (args.strict and counts["WARN"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
