#!/usr/bin/env python3
"""
Validate Wood Spatial full-run outputs.

This script checks the run state, expected CSV artifacts, expected paper figures,
optional cache files, and optional logs for a configured full Colab run.
It does not rerun experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_full_colab import POST_EXTRACT_WAVES  # noqa: E402


@dataclass(frozen=True)
class CsvSpec:
    name: str
    columns: tuple[str, ...] = ()
    min_rows: int = 1
    optional: bool = False


@dataclass
class CheckResult:
    severity: str
    item: str
    message: str


STAGE_CSVS: dict[str, list[CsvSpec]] = {
    "backbone_manifest": [
        CsvSpec(
            "backbone_pretrained_manifest.csv",
            (
                "backbone",
                "timm_model_id",
                "pretrained_tag",
                "architecture",
                "paper_img_size",
                "timm_version",
                "torch_version",
                "torchvision_version",
                "python_version",
            ),
            min_rows=7,
        ),
    ],
    "exp1": [
        CsvSpec("exp1_accuracy_matrix.csv", ("dataset", "backbone", "perturbation", "accuracy", "drop")),
        CsvSpec("exp1_bootstrap_ci.csv"),
        CsvSpec("exp1_statistical_tests.csv"),
        CsvSpec("exp1_wilcoxon_pairwise.csv"),
        CsvSpec("exp1_nemenyi_pvalues.csv", optional=True),
    ],
    "exp1b": [
        CsvSpec("exp1b_feature_geometry.csv", ("dataset", "backbone", "perturbation", "drop", "feature_drift")),
        CsvSpec("exp1b_feature_geometry_correlations.csv"),
    ],
    "exp2": [
        CsvSpec("exp2_sgi.csv"),
        CsvSpec("exp2_csi.csv"),
        CsvSpec("exp2_ari_heatmap.csv"),
        CsvSpec("exp2_silhouette.csv"),
    ],
    "exp3": [
        CsvSpec("exp3_cam_distribution.csv"),
        CsvSpec("exp3_cam_shift.csv"),
        CsvSpec("exp3_cam_entropy.csv"),
    ],
    "exp4": [
        CsvSpec("exp4_tierB_accuracy.csv"),
        CsvSpec("exp4_degradation_correlation.csv"),
        CsvSpec("exp4_kendallW.csv"),
    ],
    "exp5": [
        CsvSpec("exp5_crossmag_accuracy.csv"),
        CsvSpec("exp5_crossmag_csi.csv"),
        CsvSpec("exp5_sgi_by_mag.csv"),
    ],
    "exp5_full_crossmag": [
        CsvSpec("exp5_full_crossmag_accuracy.csv"),
        CsvSpec("exp5_full_crossmag_summary.csv"),
    ],
    "exp5_crossmag_asymmetry": [
        CsvSpec("exp5_crossmag_asymmetry_by_backbone.csv"),
        CsvSpec("exp5_crossmag_asymmetry_by_pair.csv"),
        CsvSpec("exp5_crossmag_ratio_summary.csv"),
        CsvSpec("exp5_crossmag_nonmonotonicity.csv"),
        CsvSpec("exp5_crossmag_drift_drop.csv"),
        CsvSpec("exp5_crossmag_asymmetry_summary.csv"),
    ],
    "exp_tierc_cross_source": [
        CsvSpec("exp_tierc_cross_source_by_cell.csv"),
        CsvSpec(
            "exp_tierc_cross_source_transfer.csv",
            (
                "pair",
                "direction",
                "backbone",
                "cross_source_accuracy",
                "null_mean_accuracy",
                "null_ci_low",
                "null_ci_high",
                "p_below_null",
            ),
        ),
        CsvSpec("exp_tierc_cross_source_by_pair.csv"),
        CsvSpec("exp_tierc_cross_source_summary.csv"),
    ],
    "exp_source_vs_species_probe": [
        CsvSpec(
            "exp_source_vs_species_probe_by_backbone.csv",
            (
                "pair",
                "backbone",
                "source_acc_heldout_image",
                "source_acc_leave_one_species_out",
                "cross_source_species_acc",
            ),
        ),
        CsvSpec(
            "exp_source_vs_species_probe_summary.csv",
            (
                "pair",
                "source_acc_heldout_image_mean",
                "source_acc_leave_one_species_out_mean",
                "cross_source_species_acc_mean",
            ),
        ),
    ],
    "exp5b": [
        CsvSpec("exp5b_sgi_by_mag.csv"),
        CsvSpec("exp5b_crossmag_csi.csv"),
    ],
    "exp6": [
        CsvSpec("exp6_multilevel_table.csv"),
        CsvSpec("exp6_multilevel_correlations.csv"),
        CsvSpec("exp6_backbone_failure_profile.csv"),
        CsvSpec("exp6_perturbation_failure_profile.csv"),
    ],
    "exp7": [
        CsvSpec("exp7_partial_correlations.csv"),
        CsvSpec("exp7_hierarchical_r2.csv"),
        CsvSpec("exp7_consistency_by_dataset.csv"),
        CsvSpec("exp7_consistency_by_backbone.csv"),
        CsvSpec("exp7_consistency_by_perturbation.csv"),
        CsvSpec("exp7_spearman_correlations.csv"),
        CsvSpec("exp7_backbone_path_coefficients.csv"),
    ],
    "exp7_lodo": [
        CsvSpec("exp7_lodo_feature_drift.csv"),
        CsvSpec("exp7_lodo_feature_drift_summary.csv"),
    ],
    "exp7_controlled_hierarchical": [
        CsvSpec("exp7_full_record_primary_r2.csv"),
        CsvSpec("exp7_full_record_partial_correlation.csv"),
        CsvSpec("exp7_controlled_hierarchical_r2.csv"),
        CsvSpec("exp7_partial_correlations_severity.csv"),
        CsvSpec("exp7_complete_case_partial_correlation.csv"),
        CsvSpec("exp7_spatial_cam_selection_summary.csv"),
        CsvSpec("exp7_spatial_cam_selection_coverage.csv"),
        CsvSpec("exp7_non_patch_sensitivity.csv"),
    ],
    "exp8": [
        CsvSpec("exp8_detector_auc.csv"),
        CsvSpec("exp8_threshold_sensitivity.csv"),
        CsvSpec("exp8_backbone_auc.csv"),
        CsvSpec("exp8_perturbation_auc.csv"),
        CsvSpec("exp8_operating_points.csv"),
    ],
    "exp8_baselines": [
        CsvSpec("exp8_additional_baselines.csv"),
        CsvSpec("exp8_new_baselines_auc.csv"),
    ],
    "exp9": [
        CsvSpec("exp9_tierb_geometry.csv"),
        CsvSpec("exp9_tierb_correlations.csv"),
        CsvSpec("exp9_tier_comparison.csv"),
    ],
    "exp10": [
        CsvSpec("exp10_reference_monitor_scores.csv"),
        CsvSpec("exp10_reference_monitor_auc.csv"),
    ],
    "exp10_lopo": [
        CsvSpec("exp10_lopo_monitor.csv"),
        CsvSpec("exp10_lopo_monitor_summary.csv"),
    ],
    "exp10_lodo": [
        CsvSpec("exp10_lodo_monitor.csv"),
        CsvSpec("exp10_lodo_monitor_summary.csv"),
    ],
    "exp10_sensitivity": [
        CsvSpec("exp10_monitor_sensitivity_scores.csv"),
        CsvSpec("exp10_monitor_sensitivity_auc.csv"),
    ],
    "exp_monitor_on_real_shift": [
        CsvSpec("exp_monitor_on_real_shift_scores.csv"),
        CsvSpec("exp_monitor_on_real_shift_by_condition.csv"),
        CsvSpec("exp_monitor_on_real_shift_summary.csv"),
    ],
    "exp_cross_space_drift": [
        CsvSpec("exp_cross_space_drift_matrix.csv"),
        CsvSpec("exp_cross_space_drift_summary.csv"),
    ],
    "exp_monitor_severity_dissociation": [
        CsvSpec("exp_monitor_severity_dissociation_by_condition.csv"),
        CsvSpec("exp_monitor_severity_dissociation_summary.csv"),
    ],
    "exp_mmd_gamma_sensitivity": [
        CsvSpec("exp_mmd_gamma_sensitivity_by_condition.csv"),
        CsvSpec("exp_mmd_gamma_sensitivity_summary.csv"),
    ],
    "exp_mmd_confound_and_sign": [
        CsvSpec("exp_mmd_confound_terms.csv"),
        CsvSpec("exp_mmd_confound_regression.csv"),
        CsvSpec("exp_mmd_class_count_matched.csv"),
        CsvSpec("exp_mmd_confound_summary.csv"),
    ],
    "exp_matched_class_dissociation": [
        CsvSpec("exp_matched_class_dissociation_by_cell.csv"),
        CsvSpec("exp_matched_class_dissociation_by_seed.csv"),
        CsvSpec("exp_matched_class_dissociation_summary.csv"),
    ],
    "exp10_operating_points": [
        CsvSpec("exp10_operating_points_fixed_fpr.csv"),
        CsvSpec("exp10_lopo_operating_points.csv"),
    ],
    "exp_mixed_effects": [
        CsvSpec("exp_mixed_statsmodels_summary.csv"),
        CsvSpec("exp_mixed_icc.csv"),
        CsvSpec("exp_mixed_perbackbone_ols.csv"),
        CsvSpec("exp_mixed_fixed_effects.csv"),
        CsvSpec("exp_mixed_cluster_bootstrap.csv"),
        CsvSpec("exp_mixed_cluster_bootstrap_summary.csv"),
    ],
    "exp_k_ablation": [
        CsvSpec("exp_k_ablation.csv"),
        CsvSpec("exp_k_ablation_summary.csv"),
    ],
    "exp_knn_sensitivity": [
        CsvSpec("exp_knn_sensitivity_accuracy.csv"),
        CsvSpec("exp_knn_sensitivity_summary.csv"),
    ],
    "exp_ci_main_tables": [
        CsvSpec("exp_ci_backbone_robustness.csv"),
        CsvSpec("exp_ci_perturbation_drop.csv"),
        CsvSpec("exp_ci_monitor_auc.csv"),
    ],
    "run_ablations": [
        CsvSpec("exp2_seed_sensitivity.csv"),
        CsvSpec("exp2_seed_sensitivity_summary.csv"),
        CsvSpec("exp_deblur_intervention.csv"),
        CsvSpec("exp8_additional_baselines.csv"),
        CsvSpec("exp8_new_baselines_auc.csv"),
    ],
    "hires_spatial": [
        CsvSpec("exp_hires_spatial_metrics.csv"),
    ],
    "hires_metrics": [
        CsvSpec("exp_hires_spatial_metrics.csv"),
    ],
}


STAGE_FIGURES: dict[str, list[str]] = {
    "exp5_crossmag_asymmetry": [
        "cross_magnification_asymmetry",
    ],
    "exp_tierc_cross_source": [
        "tierc_cross_source_shift",
    ],
    "exp_source_vs_species_probe": [
        "source_vs_species_probe",
    ],
    "exp_monitor_on_real_shift": [
        "monitor_on_real_shift",
    ],
    "exp_cross_space_drift": [
        "cross_space_drift",
    ],
    "exp_monitor_severity_dissociation": [
        "monitor_severity_dissociation",
    ],
    "exp_mmd_gamma_sensitivity": [
        "mmd_gamma_sensitivity",
    ],
    "exp_mmd_confound_and_sign": [
        "mmd_confound_and_class_count",
    ],
    "exp_matched_class_dissociation": [
        "matched_class_dissociation",
    ],
    "fig_vn26": [
        "fig4_spatial_cluster_panels_VN26",
        "fig6_cam_cluster_overlay_VN26",
    ],
    "fig_vn26_perturbation": [
        "fig4b_perturbation_VN26x20",
    ],
}


PAPER_FIGURES = [
    "fig2a_accuracy_heatmap",
    "fig2b_severity_curves",
    "fig3_feature_geometry_failure",
    "fig5_nemenyi_cd_diagram",
    "fig_multilevel_correlations",
    "fig_multilevel_failure_profile",
    "fig_cam_shift_vs_drop",
    "fig_csi_vs_drop",
    "fig_cam_distribution_bars",
    "fig12_crossmag_spatial",
    "fig7_hierarchical_r2",
    "fig8_roc_failure_detection",
    "fig9_partial_correlations",
    "fig10_cam_entropy_change",
    "fig11_tierb_validation",
]


LOG_ERROR_RE = re.compile(
    r"(traceback|file not found|no such file|killed|cuda out of memory|nan|exception|failed|error)",
    re.IGNORECASE,
)


def cache_tag_for(pert_name: str, value) -> str:
    """Mirror wood_spatial.core.perturbations.cache_tag_for without importing torch."""
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


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def result(severity: str, item: str, message: str) -> CheckResult:
    return CheckResult(severity=severity, item=item, message=message)


def read_state(results_dir: Path) -> dict:
    state_path = results_dir / "full_run_state.json"
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_state(cfg: dict, results_dir: Path, selected: set[str] | None) -> list[CheckResult]:
    state = read_state(results_dir)
    if not state:
        return [result("WARN", "state", f"Missing {results_dir / 'full_run_state.json'}")]

    expected = list(cfg.get("experiments", []))
    if selected:
        expected = [stage for stage in expected if stage in selected]

    completed = set(state.get("completed", []))
    out: list[CheckResult] = []
    for stage in expected:
        if stage not in completed:
            out.append(result("WARN", f"state:{stage}", "Stage is not marked completed in full_run_state.json"))
    return out


def csv_summary(path: Path, min_rows: int) -> tuple[bool, str, set[str], int]:
    if not path.exists():
        return False, "missing", set(), 0
    if path.stat().st_size == 0:
        return False, "empty file", set(), 0

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return False, "missing header", set(), 0
            rows = 0
            for rows, _ in enumerate(reader, start=1):
                if rows >= min_rows:
                    break
    except Exception as exc:
        return False, f"unreadable CSV: {exc}", set(), 0

    if rows < min_rows:
        return False, f"only {rows} data rows, expected at least {min_rows}", set(header), rows
    return True, "ok", set(header), rows


def check_csvs(results_dir: Path, stages: Iterable[str]) -> list[CheckResult]:
    out: list[CheckResult] = []
    csv_dir = results_dir / "csv"
    for stage in stages:
        for spec in STAGE_CSVS.get(stage, []):
            path = csv_dir / spec.name
            ok, msg, columns, rows = csv_summary(path, spec.min_rows)
            if not ok:
                severity = "WARN" if spec.optional else "FAIL"
                out.append(result(severity, f"csv:{stage}:{spec.name}", msg))
                continue
            missing_cols = [col for col in spec.columns if col not in columns]
            if missing_cols:
                out.append(result("FAIL", f"csv:{stage}:{spec.name}", f"missing columns: {', '.join(missing_cols)}"))
            else:
                out.append(result("OK", f"csv:{stage}:{spec.name}", f"{rows}+ rows"))
    return out


def check_png(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.stat().st_size < 1024:
        return False, f"too small ({path.stat().st_size} bytes)"
    try:
        with path.open("rb") as f:
            signature = f.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            return False, "invalid PNG signature"
    except OSError as exc:
        return False, str(exc)
    return True, f"{path.stat().st_size} bytes"


def check_pdf(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.stat().st_size < 1024:
        return False, f"too small ({path.stat().st_size} bytes)"
    try:
        with path.open("rb") as f:
            signature = f.read(5)
        if signature != b"%PDF-":
            return False, "invalid PDF signature"
    except OSError as exc:
        return False, str(exc)
    return True, f"{path.stat().st_size} bytes"


def check_figures(results_dir: Path, stages: Iterable[str], include_paper: bool) -> list[CheckResult]:
    figure_names: list[str] = []
    for stage in stages:
        figure_names.extend(STAGE_FIGURES.get(stage, []))
    if include_paper:
        figure_names.extend(PAPER_FIGURES)

    out: list[CheckResult] = []
    fig_dir = results_dir / "figures"
    for name in sorted(set(figure_names)):
        for ext, checker in (("png", check_png), ("pdf", check_pdf)):
            path = fig_dir / f"{name}.{ext}"
            ok, msg = checker(path)
            severity = "OK" if ok else "FAIL"
            out.append(result(severity, f"figure:{name}.{ext}", msg))
    return out


def _npz_ok(path: Path, required: tuple[str, ...]) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "empty file"
    try:
        with zipfile.ZipFile(path) as zf:
            names = {Path(name).stem for name in zf.namelist()}
    except zipfile.BadZipFile:
        return False, "bad npz/zip"
    missing = [key for key in required if key not in names]
    if missing:
        return False, f"missing arrays: {', '.join(missing)}"
    return True, f"{path.stat().st_size} bytes"


def expected_global_cache_paths(cfg: dict, results_dir: Path) -> list[Path]:
    feature_dir = results_dir / "feature_cache"
    backbones = cfg["backbones"]
    ext = cfg["extraction"]
    paths: list[Path] = []

    for ds in ext["global_clean_datasets"]:
        for bb in backbones:
            paths.append(feature_dir / f"{bb}_{ds}_original.npz")

    perturb_datasets = ext.get("global_perturbation_datasets", [])
    from wood_spatial.config import PERTURB_CONFIGS

    for ds in perturb_datasets:
        for pert_name, pcfg in PERTURB_CONFIGS.items():
            for value in pcfg["values"]:
                tag = cache_tag_for(pert_name, value)
                for bb in backbones:
                    paths.append(feature_dir / f"{bb}_{ds}_{tag}.npz")
    return paths


def expected_spatial_cache_paths(cfg: dict, results_dir: Path) -> list[Path]:
    spatial_dir = results_dir / "spatial_cache"
    backbones = cfg["backbones"]
    ext = cfg["extraction"]
    paths: list[Path] = []

    for ds in ext.get("spatial_clean_datasets", []):
        for bb in backbones:
            paths.append(spatial_dir / f"{bb}_{ds}_original.npz")

    for ds in ext.get("spatial_perturbation_datasets", []):
        for pert_name, value in ext.get("spatial_perturbations", []):
            tag = cache_tag_for(pert_name, value)
            for bb in backbones:
                paths.append(spatial_dir / f"{bb}_{ds}_{tag}.npz")
    return paths


def check_caches(cfg: dict, results_dir: Path) -> list[CheckResult]:
    out: list[CheckResult] = []
    global_paths = expected_global_cache_paths(cfg, results_dir)
    spatial_paths = expected_spatial_cache_paths(cfg, results_dir)

    for path in global_paths:
        ok, msg = _npz_ok(path, ("features", "labels", "paths"))
        out.append(result("OK" if ok else "FAIL", f"cache:global:{path.name}", msg))
    for path in spatial_paths:
        ok, msg = _npz_ok(path, ("features", "labels", "paths"))
        out.append(result("OK" if ok else "FAIL", f"cache:spatial:{path.name}", msg))
    return out


def stage_order(cfg: dict, selected: set[str] | None) -> list[str]:
    stages = list(cfg.get("experiments", []))
    known = [stage for wave in POST_EXTRACT_WAVES for stage in wave]
    stages = [stage for stage in known if stage in stages] + [stage for stage in stages if stage not in known]
    if selected:
        checkable = set(STAGE_CSVS) | set(STAGE_FIGURES)
        stages = [
            stage for stage in known
            if stage in selected and stage in checkable
        ]
        stages.extend(sorted(
            stage for stage in selected
            if stage in checkable and stage not in stages
        ))
    return stages


def check_logs(results_dir: Path, stages: Iterable[str]) -> list[CheckResult]:
    out: list[CheckResult] = []
    log_dir = results_dir / "logs"
    if not log_dir.exists():
        return [result("WARN", "logs", f"Missing log directory: {log_dir}")]

    stage_prefixes = tuple(stages)
    for path in sorted(log_dir.glob("*.log")):
        stem = path.stem
        if not stem.startswith(stage_prefixes):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            out.append(result("WARN", f"log:{path.name}", f"cannot read: {exc}"))
            continue
        hits = [line.strip() for line in text.splitlines() if LOG_ERROR_RE.search(line)]
        if hits:
            out.append(result("WARN", f"log:{path.name}", hits[-1][:240]))
        elif path.stat().st_size == 0:
            out.append(result("WARN", f"log:{path.name}", "empty log"))
    return out


def print_report(results: list[CheckResult], show_ok: bool) -> None:
    counts = {"OK": 0, "WARN": 0, "FAIL": 0}
    for item in results:
        counts[item.severity] = counts.get(item.severity, 0) + 1

    print("\n=== Wood Spatial output check ===")
    print(f"OK={counts.get('OK', 0)}  WARN={counts.get('WARN', 0)}  FAIL={counts.get('FAIL', 0)}")

    for severity in ("FAIL", "WARN", "OK"):
        items = [item for item in results if item.severity == severity]
        if severity == "OK" and not show_ok:
            continue
        if not items:
            continue
        print(f"\n[{severity}]")
        for item in items:
            print(f"  {item.item}: {item.message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether full-run experiment and figure outputs are complete.")
    parser.add_argument("--config", default="configs/full_colab_l4.json", help="Path to full-run config JSON.")
    parser.add_argument("--results-dir", default=None, help="Override results directory from config.")
    parser.add_argument("--only", nargs="+", default=None, help="Check only selected stages.")
    parser.add_argument("--no-state", action="store_true", help="Do not check full_run_state.json.")
    parser.add_argument("--no-paper-figures", action="store_true", help="Do not check figures from paper_figures.py.")
    parser.add_argument("--check-caches", action="store_true", help="Also validate expected feature/spatial cache npz files.")
    parser.add_argument("--check-logs", action="store_true", help="Scan log files for failure keywords.")
    parser.add_argument("--show-ok", action="store_true", help="Print successful checks, not just warnings/failures.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    results_dir = Path(args.results_dir or os.environ.get("WOOD_RESULTS_DIR") or cfg["paths"]["results_dir"])
    selected = set(args.only) if args.only else None
    stages = stage_order(cfg, selected)

    checks: list[CheckResult] = []
    if not args.no_state:
        checks.extend(check_state(cfg, results_dir, selected))
    checks.extend(check_csvs(results_dir, stages))
    checks.extend(check_figures(results_dir, stages, include_paper=not args.no_paper_figures))
    if args.check_caches:
        checks.extend(check_caches(cfg, results_dir))
    if args.check_logs:
        checks.extend(check_logs(results_dir, stages))

    if args.json:
        print(json.dumps([item.__dict__ for item in checks], indent=2))
    else:
        print(f"Config:  {Path(args.config).resolve()}")
        print(f"Results: {results_dir}")
        print_report(checks, show_ok=args.show_ok)

    has_fail = any(item.severity == "FAIL" for item in checks)
    return 1 if has_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
