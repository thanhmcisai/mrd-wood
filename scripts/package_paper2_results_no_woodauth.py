#!/usr/bin/env python3
"""Validate and package Paper-2 result artifacts after dropping WOODAUTH.

The default package contains the result files needed to update the MRD-Wood
Paper-2 cross-source/monitor sections after the FSDM41 label correction and
WOODAUTH removal. It also writes an audit report that checks for missing files,
stale WOODAUTH/DTSR14<->WOODAUTH content, and expected row/count invariants.

Example on Colab:

    cd /content/drive/MyDrive/Wood_Research_git
    python scripts/package_paper2_results_no_woodauth.py \
      --results-dir /content/drive/MyDrive/NCS/results_v5_k3_full \
      --download
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - script should fail loudly.
    raise SystemExit(f"pandas is required for validation: {exc}") from exc


STALE_TERMS = (
    "WOODAUTH",
    "DTSR14<->WOODAUTH",
    "DTSR14->WOODAUTH",
    "WOODAUTH->DTSR14",
    "TierC_DTSR14_WOODAUTH",
    "DTSR/WOODAUTH",
)


CORE_CSVS = [
    "exp_tierc_cross_source_summary.csv",
    "exp_tierc_cross_source_transfer.csv",
    "exp_tierc_cross_source_by_cell.csv",
    "exp_tierc_cross_source_by_pair.csv",
    "exp_source_vs_species_probe_summary.csv",
    "exp_source_vs_species_probe_by_backbone.csv",
    "exp_monitor_on_real_shift_scores.csv",
    "exp_monitor_on_real_shift_by_condition.csv",
    "exp_monitor_on_real_shift_summary.csv",
    "exp_monitor_severity_dissociation_records.csv",
    "exp_monitor_severity_dissociation_by_condition.csv",
    "exp_monitor_severity_dissociation_summary.csv",
    "exp_mmd_gamma_sensitivity_records.csv",
    "exp_mmd_gamma_sensitivity_by_condition.csv",
    "exp_mmd_gamma_sensitivity_summary.csv",
    "exp_mmd_confound_terms.csv",
    "exp_mmd_confound_regression.csv",
    "exp_mmd_confound_summary.csv",
    "exp_mmd_class_count_matched.csv",
    "exp_matched_class_dissociation_by_cell.csv",
    "exp_matched_class_dissociation_by_seed.csv",
    "exp_matched_class_dissociation_summary.csv",
    "exp5_crossmag_drift_drop.csv",
    "exp5_crossmag_asymmetry_summary.csv",
    "exp5_crossmag_asymmetry_by_pair.csv",
    "exp10_reference_monitor_auc.csv",
    "exp10_operating_points_fixed_fpr.csv",
]


CORE_FIGURES = [
    "tierc_cross_source_shift.png",
    "tierc_cross_source_shift.pdf",
    "source_vs_species_probe.png",
    "source_vs_species_probe.pdf",
    "monitor_on_real_shift.png",
    "monitor_on_real_shift.pdf",
    "monitor_severity_dissociation.png",
    "monitor_severity_dissociation.pdf",
    "mmd_gamma_sensitivity.png",
    "mmd_gamma_sensitivity.pdf",
    "cross_magnification_asymmetry.png",
    "cross_magnification_asymmetry.pdf",
]


OPTIONAL_FIGURES = [
    "figure1_mrd_overview.png",
    "mrd_wood_overview_figure1.svg",
    "mrd_wood_overview_figure1.png",
]


EXPECTED_EMPTY_CSVS = {
    "exp_mmd_class_count_matched.csv",
    "exp_matched_class_dissociation_by_cell.csv",
    "exp_matched_class_dissociation_by_seed.csv",
    "exp_matched_class_dissociation_summary.csv",
}


def default_results_dir() -> Path:
    candidates = [
        os.environ.get("WOOD_RESULTS_DIR"),
        "/content/drive/MyDrive/NCS/results_v5_k3_full",
        "results",
    ]
    for item in candidates:
        if item and Path(item).exists():
            return Path(item)
    return Path("results")


def read_text_limited(path: Path, max_bytes: int = 8_000_000) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def stale_hits(path: Path) -> list[str]:
    if path.suffix.lower() not in {".csv", ".json", ".txt", ".md", ".tex", ".py"}:
        return []
    text = read_text_limited(path)
    return [term for term in STALE_TERMS if term in text]


def add_check(report: dict[str, Any], ok: bool, name: str, detail: Any) -> None:
    report["checks"].append({"ok": bool(ok), "name": name, "detail": detail})
    if not ok:
        report["ok"] = False


def load_csv(csv_dir: Path, report: dict[str, Any], name: str) -> pd.DataFrame | None:
    path = csv_dir / name
    if not path.exists():
        add_check(report, False, f"required CSV exists: {name}", "missing")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        if name in EXPECTED_EMPTY_CSVS:
            try:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                text = ""
            if not text:
                return pd.DataFrame()
        add_check(report, False, f"read CSV: {name}", repr(exc))
        return None


def validate_core(results_dir: Path, report: dict[str, Any]) -> None:
    csv_dir = results_dir / "csv"
    fig_dir = results_dir / "figures"

    for name in CORE_CSVS:
        add_check(report, (csv_dir / name).exists(), f"required CSV exists: {name}", str(csv_dir / name))
    for name in CORE_FIGURES:
        add_check(report, (fig_dir / name).exists(), f"required figure exists: {name}", str(fig_dir / name))
    optional_missing = [name for name in OPTIONAL_FIGURES if not (fig_dir / name).exists()]
    report["optional_missing_figures"] = optional_missing

    tierc = load_csv(csv_dir, report, "exp_tierc_cross_source_transfer.csv")
    if tierc is not None:
        pairs = sorted(map(str, tierc.get("pair", pd.Series(dtype=str)).dropna().unique()))
        add_check(report, pairs == ["BFS46<->FSDM41"], "Tier-C transfer has only BFS46<->FSDM41", pairs)
        add_check(report, len(tierc) == 14, "Tier-C transfer row count is 14", len(tierc))
        directions = sorted(map(str, tierc.get("direction", pd.Series(dtype=str)).dropna().unique()))
        add_check(
            report,
            directions == ["BFS46->FSDM41", "FSDM41->BFS46"],
            "Tier-C directions are BFS/FSDM only",
            directions,
        )

    source_probe = load_csv(csv_dir, report, "exp_source_vs_species_probe_summary.csv")
    if source_probe is not None:
        pairs = sorted(map(str, source_probe.get("pair", pd.Series(dtype=str)).dropna().unique()))
        add_check(report, pairs == ["BFS46<->FSDM41"], "source-vs-species summary has only BFS46<->FSDM41", pairs)

    monitor = load_csv(csv_dir, report, "exp_monitor_on_real_shift_scores.csv")
    if monitor is not None:
        conditions = sorted(map(str, monitor.get("condition", pd.Series(dtype=str)).dropna().unique()))
        bad = [c for c in conditions if "WOODAUTH" in c or "DTSR14" in c]
        add_check(report, not bad, "monitor real-shift conditions have no WOODAUTH/DTSR14 pair", bad)
        counts = monitor["condition"].value_counts().to_dict() if "condition" in monitor else {}
        expected = {
            "TierA_clean": 21,
            "TierB_synth": 2128,
            "TierD_xmag": 42,
            "TierC_BFS46_FSDM41": 14,
        }
        add_check(report, counts == expected, "monitor real-shift condition counts match no-WOODAUTH run", counts)

    gamma = load_csv(csv_dir, report, "exp_mmd_gamma_sensitivity_records.csv")
    if gamma is not None:
        policy_counts = gamma["policy"].value_counts().to_dict() if "policy" in gamma else {}
        add_check(
            report,
            policy_counts == {
                "per_pair_median": 77,
                "global_median": 77,
                "global_median_x0.5": 77,
                "global_median_x2.0": 77,
            },
            "MMD gamma policy counts are 77 each",
            policy_counts,
        )
        stale_conditions = []
        if "condition" in gamma:
            stale_conditions = [
                c for c in sorted(map(str, gamma["condition"].dropna().unique()))
                if "WOODAUTH" in c or "DTSR14" in c
            ]
        add_check(report, not stale_conditions, "MMD gamma conditions have no stale WOODAUTH/DTSR14 pair", stale_conditions)

    confound = load_csv(csv_dir, report, "exp_mmd_confound_terms.csv")
    if confound is not None:
        pairs = sorted(map(str, confound.get("pair", pd.Series(dtype=str)).dropna().unique()))
        add_check(report, pairs == ["BFS46<->FSDM41"], "MMD confound terms have only BFS46<->FSDM41", pairs)
        add_check(report, len(confound) == 14, "MMD confound terms row count is 14", len(confound))

    for empty_name in [
        "exp_mmd_class_count_matched.csv",
        "exp_matched_class_dissociation_by_cell.csv",
        "exp_matched_class_dissociation_by_seed.csv",
        "exp_matched_class_dissociation_summary.csv",
    ]:
        df = load_csv(csv_dir, report, empty_name)
        if df is not None:
            add_check(report, len(df) == 0, f"{empty_name} is empty after removing second Tier-C pair", len(df))


def scan_results_tree(results_dir: Path, report: dict[str, Any]) -> None:
    csv_dir = results_dir / "csv"
    hits = []
    if csv_dir.exists():
        for path in sorted(csv_dir.glob("*")):
            if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
                found = stale_hits(path)
                if found:
                    hits.append({"file": str(path.relative_to(results_dir)), "terms": found})
    report["stale_term_hits_in_results_csv"] = hits


def collect_files(results_dir: Path, include_all_csv: bool) -> list[Path]:
    files: list[Path] = []
    csv_dir = results_dir / "csv"
    fig_dir = results_dir / "figures"
    names = sorted(p.name for p in csv_dir.glob("*.csv")) if include_all_csv and csv_dir.exists() else CORE_CSVS
    for name in names:
        path = csv_dir / name
        if path.exists():
            files.append(path)
    for name in CORE_FIGURES:
        path = fig_dir / name
        if path.exists():
            files.append(path)
    for name in OPTIONAL_FIGURES:
        path = fig_dir / name
        if path.exists():
            files.append(path)
    for name in [
        "configs/full_colab_l4.json",
        "wood_spatial/core/label_corrections.py",
    ]:
        path = Path(name)
        if path.exists():
            files.append(path)
    return files


def write_summary_tsv(report: dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["ok", "check", "detail"])
        for check in report["checks"]:
            writer.writerow([check["ok"], check["name"], json.dumps(check["detail"], ensure_ascii=False)])


def make_zip(files: list[Path], report_json: Path, report_tsv: Path, out_zip: Path, results_dir: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            if not path.exists() or not path.is_file():
                continue
            try:
                arcname = path.relative_to(results_dir.parent)
            except ValueError:
                arcname = Path("repo_context") / path
            zf.write(path, arcname.as_posix())
        zf.write(report_json, Path("audit") / report_json.name)
        zf.write(report_tsv, Path("audit") / report_tsv.name)


def maybe_download(path: Path) -> None:
    try:
        from google.colab import files  # type: ignore
        from IPython import get_ipython  # type: ignore
    except Exception:
        print(f"[download] not in Colab; zip is at {path}")
        return
    ip = get_ipython()
    if ip is None or getattr(ip, "kernel", None) is None:
        print(f"[download] Colab download unavailable in this process; zip is at {path}")
        return
    try:
        files.download(str(path))
    except Exception as exc:
        print(f"[download] Colab download failed ({exc}); zip is at {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--out", type=Path, default=None, help="Output zip path.")
    parser.add_argument("--include-all-csv", action="store_true", help="Include every top-level results/csv/*.csv in the zip.")
    parser.add_argument("--download", action="store_true", help="Call google.colab.files.download on the zip when available.")
    parser.add_argument("--allow-failed-audit", action="store_true", help="Create zip even if validation checks fail.")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_zip = args.out or (results_dir / f"paper2_no_woodauth_results_{stamp}.zip")
    audit_dir = results_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    report_json = audit_dir / f"paper2_no_woodauth_package_audit_{stamp}.json"
    report_tsv = audit_dir / f"paper2_no_woodauth_package_audit_{stamp}.tsv"

    report: dict[str, Any] = {
        "ok": True,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "results_dir": str(results_dir),
        "zip": str(out_zip),
        "checks": [],
    }
    validate_core(results_dir, report)
    scan_results_tree(results_dir, report)

    packaged_files = collect_files(results_dir, args.include_all_csv)
    packaged_hit_files = []
    for path in packaged_files:
        found = stale_hits(path)
        if found:
            packaged_hit_files.append({"file": str(path), "terms": found})
    report["stale_term_hits_in_packaged_files"] = packaged_hit_files
    add_check(report, not packaged_hit_files, "packaged files contain no stale WOODAUTH/DTSR14 terms", packaged_hit_files)

    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_tsv(report, report_tsv)

    if not report["ok"] and not args.allow_failed_audit:
        print(f"[FAIL] Audit failed. Report: {report_json}")
        print("Use --allow-failed-audit to create the zip anyway.")
        raise SystemExit(2)

    make_zip(packaged_files, report_json, report_tsv, out_zip, results_dir)
    print(f"[OK] package written: {out_zip}")
    print(f"[OK] audit report:    {report_json}")
    if report["stale_term_hits_in_results_csv"]:
        print("[WARN] Some non-packaged results/csv files still contain WOODAUTH/DTSR14 terms.")
        print("       See stale_term_hits_in_results_csv in the audit JSON.")
    if args.download:
        maybe_download(out_zip)


if __name__ == "__main__":
    main()
