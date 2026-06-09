#!/usr/bin/env python3
"""Direction-aware margin analysis for the MRD-Wood K1 validation."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception:  # pragma: no cover
    roc_auc_score = None
    roc_curve = None

from wood_spatial.experiments.exp_margin_crossing_validation import (
    _cache_tag_for,
    _load_cache_np,
    build_prototypes,
    l2_normalize,
)


def margin_competitor_pred(feats, labels, protos, classes):
    feats = l2_normalize(feats)
    col = {c: j for j, c in enumerate(classes)}
    scores = feats @ protos.T
    pred = np.array([classes[j] for j in scores.argmax(axis=1)])
    n = scores.shape[0]
    true_col = np.array([col[y] for y in labels])
    true_score = scores[np.arange(n), true_col]
    other = scores.copy()
    other[np.arange(n), true_col] = -np.inf
    comp_col = other.argmax(axis=1)
    margin = true_score - other[np.arange(n), comp_col]
    return pred, margin, comp_col


def firstorder_margin(clean_feats, pert_feats, labels, protos, classes):
    col = {c: j for j, c in enumerate(classes)}
    z_clean = l2_normalize(clean_feats)
    z_pert = l2_normalize(pert_feats)

    pred_clean, gamma, comp_col = margin_competitor_pred(
        clean_feats, labels, protos, classes
    )
    p_y = protos[[col[y] for y in labels]]
    p_comp = protos[comp_col]
    direction = p_y - p_comp
    aligned_erosion = np.sum((z_clean - z_pert) * direction, axis=1)
    pred_margin = gamma - aligned_erosion

    total_drift = 1.0 - np.sum(z_clean * z_pert, axis=1)
    pred_pert = np.array([classes[j] for j in (z_pert @ protos.T).argmax(axis=1)])
    flip = pred_pert != labels
    clean_correct = (pred_clean == labels) & (gamma > 0)
    return {
        "total_drift": total_drift,
        "gamma": gamma,
        "pred_margin": pred_margin,
        "flip": flip.astype(np.int8),
        "clean_correct": clean_correct,
    }


def summarize(total_drift, pred_margin, flip):
    out = {
        "auc_total_drift": float("nan"),
        "auc_pred_margin": float("nan"),
        "firstorder_recall": float("nan"),
    }
    if roc_auc_score is not None and 0 < int(flip.sum()) < int(flip.size):
        out["auc_total_drift"] = float(roc_auc_score(flip, total_drift))
        out["auc_pred_margin"] = float(roc_auc_score(flip, -pred_margin))
    if flip.sum() > 0:
        out["firstorder_recall"] = float(np.mean(pred_margin[flip == 1] <= 0))
    return out


def _run_pair(task):
    ds, bb, params = task
    if params["results_dir"]:
        os.environ["WOOD_RESULTS_DIR"] = params["results_dir"]

    from sklearn.model_selection import StratifiedShuffleSplit
    from wood_spatial.config import PERTURB_CONFIGS, V4_FEAT_CACHE

    try:
        clean, labels, clean_paths = _load_cache_np(V4_FEAT_CACHE, bb, ds, "original")
    except FileNotFoundError as exc:
        return {
            "dataset": ds,
            "backbone": bb,
            "ok": False,
            "missing": str(exc),
            "rows": [],
            "drift": np.array([], dtype=np.float64),
            "pred_margin": np.array([], dtype=np.float64),
            "flip": np.array([], dtype=np.int8),
            "path_warning_count": 0,
        }

    splitter = StratifiedShuffleSplit(
        n_splits=params["n_splits"],
        test_size=params["test_size"],
        random_state=params["seed"],
    )
    split_indices = list(splitter.split(clean, labels))
    classes = sorted(np.unique(labels).tolist())
    rows = []
    all_drift, all_pred_margin, all_flip = [], [], []
    path_warning_count = 0

    for pert_name, pcfg in PERTURB_CONFIGS.items():
        for value in pcfg["values"]:
            tag = _cache_tag_for(pert_name, value)
            try:
                shifted, shifted_labels, shifted_paths = _load_cache_np(
                    V4_FEAT_CACHE, bb, ds, tag
                )
            except FileNotFoundError:
                continue
            if len(shifted) != len(clean) or not np.array_equal(labels, shifted_labels):
                raise RuntimeError(f"Cache order/label mismatch for {bb}/{ds}/{tag}")
            if not np.array_equal(clean_paths, shifted_paths):
                path_warning_count += 1

            cond_drift, cond_pred_margin, cond_flip = [], [], []
            for _split_id, (train_idx, test_idx) in enumerate(split_indices):
                protos = build_prototypes(clean[train_idx], labels[train_idx], classes)
                result = firstorder_margin(
                    clean[test_idx],
                    shifted[test_idx],
                    labels[test_idx],
                    protos,
                    classes,
                )
                cc = result["clean_correct"]
                if not np.any(cc):
                    continue
                cond_drift.append(result["total_drift"][cc])
                cond_pred_margin.append(result["pred_margin"][cc])
                cond_flip.append(result["flip"][cc])

            if not cond_drift:
                continue
            d = np.concatenate(cond_drift)
            pm = np.concatenate(cond_pred_margin)
            f = np.concatenate(cond_flip)
            all_drift.append(d)
            all_pred_margin.append(pm)
            all_flip.append(f)
            s = summarize(d, pm, f)
            rows.append({
                "dataset": ds,
                "backbone": bb,
                "perturbation": pert_name,
                "severity": value,
                "tag": tag,
                "n_pairs": int(d.size),
                "n_flip": int(f.sum()),
                "flip_rate": float(f.mean()) if d.size else float("nan"),
                **s,
            })

    return {
        "dataset": ds,
        "backbone": bb,
        "ok": True,
        "missing": "",
        "rows": rows,
        "drift": np.concatenate(all_drift) if all_drift else np.array([], dtype=np.float64),
        "pred_margin": np.concatenate(all_pred_margin) if all_pred_margin else np.array([], dtype=np.float64),
        "flip": np.concatenate(all_flip) if all_flip else np.array([], dtype=np.int8),
        "path_warning_count": path_warning_count,
    }


def _make_roc_figure(total_drift, pred_margin, flip, fig_path):
    if roc_curve is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skipping figure: {exc}")
        return
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fpr_d, tpr_d, _ = roc_curve(flip, total_drift)
    fpr_m, tpr_m, _ = roc_curve(flip, -pred_margin)
    aucs = summarize(total_drift, pred_margin, flip)
    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=160)
    ax.plot(fpr_d, tpr_d, lw=1.5, color="#4C78A8",
            label=f"raw drift AUC={aucs['auc_total_drift']:.3f}")
    ax.plot(fpr_m, tpr_m, lw=1.5, color="#E45756",
            label=f"direction-aware AUC={aucs['auc_pred_margin']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("Direction-aware margin prediction")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] figure saved: {fig_path}")


def run_real(args):
    if args.results_dir:
        os.environ["WOOD_RESULTS_DIR"] = args.results_dir

    import pandas as pd
    from wood_spatial.config import BB_ORDER, TIER_A, V4_CSV

    datasets = args.datasets.split(",") if args.datasets else list(TIER_A)
    backbones = args.backbones.split(",") if args.backbones else list(BB_ORDER)
    V4_CSV.mkdir(parents=True, exist_ok=True)
    params = {
        "results_dir": args.results_dir,
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "seed": args.seed,
    }
    tasks = [(ds, bb, params) for ds in datasets for bb in backbones]
    all_drift, all_pred_margin, all_flip = [], [], []
    by_condition, missing = [], []

    if args.jobs > 1:
        print(f"[info] running {len(tasks)} dataset/backbone tasks with jobs={args.jobs}", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_run_pair, task): task for task in tasks}
            for fut in as_completed(futs):
                result = fut.result()
                ds, bb = result["dataset"], result["backbone"]
                if not result["ok"]:
                    missing.append(result["missing"])
                    print(f"[skip] {ds} / {bb}: missing clean cache", flush=True)
                    continue
                if result["path_warning_count"]:
                    print(f"[warn] {ds} / {bb}: {result['path_warning_count']} path-order warnings; assuming row alignment", flush=True)
                if result["drift"].size:
                    all_drift.append(result["drift"])
                    all_pred_margin.append(result["pred_margin"])
                    all_flip.append(result["flip"])
                    by_condition.extend(result["rows"])
                print(f"[done] {ds} / {bb}: conditions={len(result['rows'])}", flush=True)
    else:
        for task in tasks:
            ds, bb, _ = task
            print(f"[run] {ds} / {bb}", flush=True)
            result = _run_pair(task)
            if not result["ok"]:
                missing.append(result["missing"])
                continue
            if result["drift"].size:
                all_drift.append(result["drift"])
                all_pred_margin.append(result["pred_margin"])
                all_flip.append(result["flip"])
                by_condition.extend(result["rows"])
            print(f"[done] {ds} / {bb}: cumulative_conditions={len(by_condition)}", flush=True)

    if not all_drift:
        raise RuntimeError("No usable perturbation feature caches found.")

    drift = np.concatenate(all_drift)
    pred_margin = np.concatenate(all_pred_margin)
    flip = np.concatenate(all_flip)
    s = summarize(drift, pred_margin, flip)
    summary = {
        "protocol": "nearest_centroid_firstorder_margin_v1",
        "datasets": ",".join(datasets),
        "backbones": ",".join(backbones),
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "split_seed": args.seed,
        "n_pairs": int(drift.size),
        "n_flip": int(flip.sum()),
        "flip_rate": float(flip.mean()),
        **s,
        "auc_gain_pred_margin_vs_drift": s["auc_pred_margin"] - s["auc_total_drift"],
    }

    out_summary = Path(args.out_summary) if args.out_summary else V4_CSV / "exp_margin_aligned_summary.csv"
    out_csv = Path(args.out_csv) if args.out_csv else V4_CSV / "exp_margin_aligned_by_condition.csv"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    pd.DataFrame(by_condition).to_csv(out_csv, index=False)
    if args.fig:
        _make_roc_figure(drift, pred_margin, flip, args.fig)
    if missing:
        print(f"[warn] missing clean caches: {len(missing)} (first: {missing[0]})")
    print(f"[ok] wrote {out_summary}")
    print(f"[ok] wrote {out_csv}")
    print("\n=== direction-aware margin analysis ===")
    print(f"  records={summary['n_pairs']}  flips={summary['n_flip']} ({100*summary['flip_rate']:.1f}%)")
    print(f"  AUC total drift     : {summary['auc_total_drift']:.3f}")
    print(f"  AUC first-order     : {summary['auc_pred_margin']:.3f}")
    print(f"  AUC gain            : {summary['auc_gain_pred_margin_vs_drift']:.3f}")
    print(f"  first-order recall  : {summary['firstorder_recall']:.3f}\n")
    return summary


def _demo():
    rng = np.random.default_rng(0)
    c, d, npc = 40, 128, 150
    means = l2_normalize(rng.normal(size=(c, d)))
    scales = rng.uniform(0.05, 0.30, size=c)

    def sample(n):
        labels = rng.integers(0, c, size=n)
        x = means[labels] + scales[labels][:, None] * rng.normal(size=(n, d))
        return l2_normalize(x), labels

    xtr, ytr = sample(c * npc)
    xte, yte = sample(c * npc)
    classes = sorted(set(ytr.tolist()))
    protos = build_prototypes(xtr, ytr, classes)

    drift_all, predm_all, flip_all = [], [], []
    for sev in (0.05, 0.12, 0.22, 0.35, 0.55):
        xp = l2_normalize(xte + sev * rng.normal(size=xte.shape))
        result = firstorder_margin(xte, xp, yte, protos, classes)
        cc = result["clean_correct"]
        drift_all.append(result["total_drift"][cc])
        predm_all.append(result["pred_margin"][cc])
        flip_all.append(result["flip"][cc])
    drift = np.concatenate(drift_all)
    predm = np.concatenate(predm_all)
    flip = np.concatenate(flip_all)
    s = summarize(drift, predm, flip)
    print("\n=== aligned (direction-aware) margin analysis [demo] ===")
    print(f"  records={drift.size}  flip_rate={flip.mean():.3f}")
    print(f"  AUC total drift (loose magnitude signal)   : {s['auc_total_drift']:.3f}")
    print(f"  AUC predicted margin (direction-aware)     : {s['auc_pred_margin']:.3f}")
    print(f"  first-order recall (flips with hatGamma<=0): {s['firstorder_recall']:.3f}\n")


def main():
    parser = argparse.ArgumentParser(description="Run direction-aware margin analysis.")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--backbones", default="")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out-summary", default="")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--fig", default="results/figures/fig_margin_direction.png")
    args = parser.parse_args()
    if args.real:
        run_real(args)
    else:
        _demo()


if __name__ == "__main__":
    main()
