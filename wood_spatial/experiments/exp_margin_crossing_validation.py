#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
margin_crossing_validation.py
=============================
K1 experiment for MRD-Wood: empirical validation of the margin-crossing
mechanism (Proposition 1 / Corollary 1 of Section 3.5 "Analytical account").

WHAT THIS TESTS
---------------
Under the prototype (cosine nearest-centroid) rule, theory says a previously
correct sample flips only if its paired feature drift exceeds a per-sample
margin gate:

        flip(x)  ==>  Delta_f(x) >= Gamma(x)^2 / 8                 (Corollary 1)

Equivalently, with the margin-normalized drift

        Delta_tilde(x) = 8 * Delta_f(x) / Gamma(x)^2,              (Eq. 19)

failures must concentrate at Delta_tilde(x) >= 1.

This script reports the three numbers that fill the \todo placeholders in the
PART B paragraph, plus the scatter figure figures/fig_margin_crossing.png:

  (a) frac_necessary : fraction of FLIPPED samples that satisfy
                       Delta_f >= Gamma^2/8.
                       This is THEOREM-GUARANTEED to be ~1.0 for the
                       nearest-centroid rule; it is a CONSISTENCY/SANITY check.
                       A value clearly below 1.0 means a bug (wrong prototypes,
                       features not L2-normalized, or flip defined with a
                       different rule than the margin).
  (b) auc_raw, auc_norm : per-sample flip ROC-AUC using raw drift vs
                       margin-normalized drift. The MECHANISTIC prediction is
                       auc_norm > auc_raw (margin normalization is the gate the
                       theory predicts). This is the real empirical evidence.
  (c) figure : scatter of Delta_f (y) vs Gamma^2/8 (x), coloured by
                       preserved/flipped, with the predicted line y = x.

IMPORTANT
---------
* Use the NEAREST-CENTROID rule here, NOT the main cosine-kNN rule. The bound is
  exact for nearest-centroid; kNN obeys the same logic only with a local margin.
* Prototypes are built from the CLEAN TRAIN split; margins/drift/flips are
  evaluated on the CLEAN/PERTURBED TEST split (disjoint), matching the paper.
* Drift is PAIRED: pert_test_feats[i] must be the perturbed version of
  clean_test_feats[i] (same image, same row order).
* All features must be L2-normalized (the script re-normalizes defensively).

USAGE
-----
  python margin_crossing_validation.py --demo
  python margin_crossing_validation.py --real --results-dir /path/to/results
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None


# ----------------------------------------------------------------------------- #
# Core math (rule-agnostic; depends only on features + labels)
# ----------------------------------------------------------------------------- #
def l2_normalize(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def build_prototypes(train_feats, train_labels, classes):
    """p_c = normalized mean of normalized clean-train features of class c."""
    train_feats = l2_normalize(train_feats)
    protos = np.zeros((len(classes), train_feats.shape[1]), dtype=np.float64)
    for j, c in enumerate(classes):
        mask = (train_labels == c)
        if not np.any(mask):
            raise ValueError(f"class {c} has no training samples")
        mu = train_feats[mask].mean(axis=0)
        protos[j] = mu
    return l2_normalize(protos)  # p_c = mu_c / ||mu_c||


def scores_and_margin(feats, labels, protos, classes):
    """
    Return (pred, margin) for the cosine nearest-centroid rule.
      scores[i,j] = <z_i, p_{classes[j]}>
      pred[i]     = argmax_j scores
      margin[i]   = score(true class) - max score(other classes)   (Eq. 16)
    margin > 0  <=>  correctly classified.
    """
    feats = l2_normalize(feats)
    class_to_col = {c: j for j, c in enumerate(classes)}
    S = feats @ protos.T                                  # (n, C)
    pred_col = S.argmax(axis=1)
    pred = np.array([classes[j] for j in pred_col])
    true_col = np.array([class_to_col[y] for y in labels])
    n = S.shape[0]
    true_score = S[np.arange(n), true_col]
    S_other = S.copy()
    S_other[np.arange(n), true_col] = -np.inf
    max_other = S_other.max(axis=1)
    margin = true_score - max_other
    return pred, margin


def paired_drift(clean_feats, pert_feats):
    """Delta_f = 1 - cos(clean, pert), paired row-by-row.  (Eq. 3)"""
    a = l2_normalize(clean_feats)
    b = l2_normalize(pert_feats)
    cos = np.sum(a * b, axis=1)
    return 1.0 - cos


# ----------------------------------------------------------------------------- #
# Experiment driver
# ----------------------------------------------------------------------------- #
def run_validation(clean_train_feats, clean_train_labels,
                   clean_test_feats, clean_test_labels,
                   perturbed_conditions, fig_path=None, tol=1e-9):
    """
    perturbed_conditions: iterable of (name, pert_test_feats) where pert_test_feats
        is aligned row-by-row with clean_test_feats (paired, same images).
    Returns a results dict with the three reportable numbers.
    """
    classes = sorted(np.unique(clean_train_labels).tolist())
    protos = build_prototypes(clean_train_feats, clean_train_labels, classes)

    # Clean-side margin and clean correctness (prototype rule).
    pred_clean, margin_clean = scores_and_margin(
        clean_test_feats, clean_test_labels, protos, classes)
    clean_correct = (pred_clean == clean_test_labels) & (margin_clean > 0)

    # Accumulate over all perturbation conditions, restricted to clean-correct.
    all_drift, all_gate, all_flip, all_normdrift = [], [], [], []
    for name, pert_feats in perturbed_conditions:
        pert_feats = np.asarray(pert_feats)
        if pert_feats.shape[0] != clean_test_feats.shape[0]:
            raise ValueError(f"[{name}] pert/clean row mismatch (drift must be paired)")
        df = paired_drift(clean_test_feats, pert_feats)        # Delta_f
        pred_pert, _ = scores_and_margin(pert_feats, clean_test_labels, protos, classes)
        flip = clean_correct & (pred_pert != clean_test_labels)

        g = margin_clean[clean_correct]                        # Gamma > 0
        gate = (g ** 2) / 8.0                                  # Gamma^2/8
        df_cc = df[clean_correct]
        normdrift = 8.0 * df_cc / (g ** 2)                     # Delta_tilde

        all_drift.append(df_cc)
        all_gate.append(gate)
        all_flip.append(flip[clean_correct].astype(int))
        all_normdrift.append(normdrift)

    drift = np.concatenate(all_drift)
    gate = np.concatenate(all_gate)
    flip = np.concatenate(all_flip)
    normdrift = np.concatenate(all_normdrift)

    n_pairs = drift.size
    n_flip = int(flip.sum())

    # (a) Necessary-condition check (theorem-guaranteed ~1.0).
    if n_flip > 0:
        frac_necessary = float(np.mean(drift[flip == 1] >= gate[flip == 1] - tol))
    else:
        frac_necessary = float("nan")

    certified = drift < gate
    coverage_certified = float(np.mean(certified))
    flip_rate_in_certified = float(np.mean(flip[certified])) if certified.any() else float("nan")
    preserved = flip == 0
    frac_preserved_certified = (
        float(np.mean(certified[preserved])) if preserved.any() else float("nan")
    )

    # (b) Does the MARGIN add predictive value beyond drift?  This is the robust,
    #     theory-motivated test: the gate Gamma^2/8 should carry information about
    #     flips that raw drift alone misses (two samples with equal drift but
    #     different margins have different flip risk).
    #       * auc_raw         : flip predicted by raw drift Delta_f
    #       * auc_gatedist    : flip predicted by signed gate distance Delta_f - Gamma^2/8
    #       * auc_drift_logit / auc_driftmargin_logit : nested logistic models
    #         flip ~ Delta_f   vs   flip ~ Delta_f + Gamma  (margin adds value if higher)
    #     (margin-normalized drift is also reported but is only necessary-not-
    #      sufficient, so it is a weaker, data-dependent signal.)
    auc_raw = auc_gatedist = auc_norm = float("nan")
    auc_drift_logit = auc_driftmargin_logit = float("nan")
    gamma_cc = np.sqrt(8.0 * gate)  # recover Gamma from gate = Gamma^2/8
    if roc_auc_score is not None and 0 < n_flip < n_pairs:
        auc_raw = float(roc_auc_score(flip, drift))
        auc_gatedist = float(roc_auc_score(flip, drift - gate))
        auc_norm = float(roc_auc_score(flip, normdrift))
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_predict
            def cv_auc(X):
                Xs = StandardScaler().fit_transform(X)
                p = cross_val_predict(
                    LogisticRegression(max_iter=1000), Xs, flip,
                    cv=5, method="predict_proba")[:, 1]
                return float(roc_auc_score(flip, p))
            auc_drift_logit = cv_auc(drift.reshape(-1, 1))
            auc_driftmargin_logit = cv_auc(np.column_stack([drift, gamma_cc]))
        except Exception as e:  # pragma: no cover
            print(f"[warn] logistic nested-model check skipped: {e}")

    results = {
        "n_pairs": n_pairs,
        "n_flip": n_flip,
        "flip_rate": n_flip / n_pairs if n_pairs else float("nan"),
        "frac_necessary": frac_necessary,            # -> (a)  \todo{X.X\%}
        "coverage_certified_safe": coverage_certified,
        "flip_rate_in_certified": flip_rate_in_certified,
        "frac_preserved_certified": frac_preserved_certified,
        "auc_raw_drift": auc_raw,                    # -> (b)  raw drift
        "auc_gate_distance": auc_gatedist,           # -> (b)  drift - Gamma^2/8
        "auc_margin_normalized": auc_norm,           #          (weaker, optional)
        "auc_drift_logit": auc_drift_logit,          # -> (b)  flip ~ drift
        "auc_drift_plus_margin_logit": auc_driftmargin_logit,  # -> (b) flip ~ drift+margin
    }

    if fig_path is not None:
        _make_figure(drift, gate, flip, fig_path)

    return results


def _make_figure(drift, gate, flip, fig_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skipping figure: {e}")
        return
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.4, 4.0), dpi=160)

    upper = float(np.percentile(drift, 99.5)) + 1e-9
    edges = np.linspace(0, upper, 21)
    centers, rates = [], []
    for i in range(len(edges) - 1):
        m = (drift >= edges[i]) & (drift < edges[i + 1])
        if m.sum() >= 30:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            rates.append(float(flip[m].mean()))
    ax_l.plot(centers, rates, "o-", color="#E45756", ms=4, lw=1.4)
    ax_l.set_xlabel(r"feature drift  $\Delta_f(x)$")
    ax_l.set_ylabel("empirical flip rate")
    ax_l.set_ylim(0, 1)
    ax_l.set_title("(a) Drift magnitude vs. flips")

    idx = np.arange(drift.size)
    if drift.size > 60000:
        rng = np.random.default_rng(42)
        idx = rng.choice(drift.size, 60000, replace=False)
    d, g, f = drift[idx], np.maximum(gate[idx], 1e-6), flip[idx]
    ax_r.scatter(g[f == 0], d[f == 0], s=4, alpha=0.20, label="preserved",
                 color="#4C78A8", linewidths=0)
    ax_r.scatter(g[f == 1], d[f == 1], s=5, alpha=0.40, label="flipped",
                 color="#E45756", linewidths=0)
    hi = float(np.percentile(np.concatenate([d, g]), 99.7)) if d.size else 1.0
    hi = max(hi, 1e-6)
    xs = np.linspace(1e-6, hi, 200)
    ax_r.plot(xs, xs, "k--", lw=1.2, label=r"$\Delta_f=\Gamma^2/8$")
    ax_r.set_xscale("log")
    ax_r.set_xlim(1e-6, hi)
    ax_r.set_ylim(0, hi)
    ax_r.set_xlabel(r"margin gate  $\Gamma(x)^2/8$  (log scale)")
    ax_r.set_ylabel(r"feature drift  $\Delta_f(x)$")
    cov = float(np.mean(drift < gate))
    ax_r.set_title(f"(b) Bound certifies {100 * cov:.1f}% safe")
    ax_r.legend(loc="lower right", frameon=False, fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] figure saved: {fig_path}")


def print_report(results):
    print("\n========== K1 MARGIN-CROSSING VALIDATION ==========")
    print(f"  paired (sample x condition) records : {results['n_pairs']}")
    print(f"  flips                               : {results['n_flip']} "
          f"({100*results['flip_rate']:.1f}% of clean-correct)")
    print("  ---- numbers to paste into PART B \\todo ----")
    fn = results["frac_necessary"]
    print(f"  (a) frac flips with Delta_f >= Gamma^2/8 : {100*fn:.1f}%   "
          f"[theorem says ~100%; <100% => bug]")
    cov = results.get("coverage_certified_safe", float("nan"))
    fric = results.get("flip_rate_in_certified", float("nan"))
    fpc = results.get("frac_preserved_certified", float("nan"))
    print(f"  (a') bound tightness: certifies {100*cov:.1f}% of records as safe "
          f"(flip rate inside = {100*fric:.2f}%, must be ~0)")
    print(f"       => explains {100*fpc:.1f}% of PRESERVED samples; "
          f"the rest are preserved despite exceeding the conservative gate")
    print(f"  (b) flip ROC-AUC, raw drift only         : {results['auc_raw_drift']:.3f}")
    print(f"      flip ROC-AUC, signed gate distance   : {results['auc_gate_distance']:.3f}")
    print(f"      flip ROC-AUC, logistic drift         : {results['auc_drift_logit']:.3f}")
    print(f"      flip ROC-AUC, logistic drift+margin  : {results['auc_drift_plus_margin_logit']:.3f}")
    margin_adds = results['auc_drift_plus_margin_logit'] > results['auc_drift_logit']
    print(f"      prediction: margin adds info (drift+margin > drift) : {margin_adds}")
    print(f"      [optional] margin-normalized drift AUC : {results['auc_margin_normalized']:.3f}")
    print("===================================================\n")


# ----------------------------------------------------------------------------- #
# REAL DATA: cached-feature runner used by the paper experiment.
# ----------------------------------------------------------------------------- #
def _load_cache_np(cache_dir, backbone: str, dataset: str, tag: str):
    path = Path(cache_dir) / f"{backbone}_{dataset}_{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    feats = l2_normalize(data["features"].astype(np.float64))
    return feats, data["labels"], data["paths"]


def _cache_tag_for(pert_name: str, value) -> str:
    """Local copy to avoid importing torch-backed perturbation transforms."""
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


def _auc_summary(drift, gate, flip, max_auc_records=None, seed=42, skip_logit=False):
    drift = np.asarray(drift, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)
    flip = np.asarray(flip, dtype=np.int32)
    normdrift = 8.0 * drift / np.maximum((np.sqrt(8.0 * gate) ** 2), 1e-12)

    idx = np.arange(drift.size)
    if max_auc_records is not None and drift.size > max_auc_records:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_auc_records, replace=False)
        print(f"[info] AUC checks subsampled from {drift.size} to {len(idx)} records")
    else:
        print(f"[info] AUC checks using {len(idx)} records")

    d = drift[idx]
    g = gate[idx]
    f = flip[idx]
    gamma = np.sqrt(8.0 * g)
    nd = normdrift[idx]

    out = {
        "n_auc_records": int(len(idx)),
        "auc_raw_drift": float("nan"),
        "auc_gate_distance": float("nan"),
        "auc_margin_normalized": float("nan"),
        "auc_drift_logit": float("nan"),
        "auc_drift_plus_margin_logit": float("nan"),
    }
    if roc_auc_score is None or len(np.unique(f)) < 2:
        return out

    out["auc_raw_drift"] = float(roc_auc_score(f, d))
    out["auc_gate_distance"] = float(roc_auc_score(f, d - g))
    out["auc_margin_normalized"] = float(roc_auc_score(f, nd))
    if skip_logit:
        print("[info] skipped logistic nested-model AUC (--skip-logit)")
        return out

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_predict, StratifiedKFold

        counts = np.bincount(f, minlength=2)
        n_splits = int(min(5, counts.min()))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

            def cv_auc(X):
                Xs = StandardScaler().fit_transform(X)
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                p = cross_val_predict(clf, Xs, f, cv=cv, method="predict_proba")[:, 1]
                return float(roc_auc_score(f, p))

            out["auc_drift_logit"] = cv_auc(d.reshape(-1, 1))
            out["auc_drift_plus_margin_logit"] = cv_auc(np.column_stack([d, gamma]))
    except Exception as e:  # pragma: no cover
        print(f"[warn] logistic nested-model check skipped: {e}")
    return out


def _run_real_pair_task(task):
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
            "gate": np.array([], dtype=np.float64),
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
    all_drift, all_gate, all_flip = [], [], []
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

            cond_drift, cond_gate, cond_flip = [], [], []
            for _split_id, (train_idx, test_idx) in enumerate(split_indices):
                protos = build_prototypes(clean[train_idx], labels[train_idx], classes)
                pred_clean, margin_clean = scores_and_margin(
                    clean[test_idx], labels[test_idx], protos, classes
                )
                clean_correct = (pred_clean == labels[test_idx]) & (margin_clean > 0)
                if not np.any(clean_correct):
                    continue

                df = paired_drift(clean[test_idx], shifted[test_idx])
                pred_pert, _ = scores_and_margin(shifted[test_idx], labels[test_idx], protos, classes)
                flip = clean_correct & (pred_pert != labels[test_idx])
                gamma = margin_clean[clean_correct]
                gate = (gamma ** 2) / 8.0
                df_cc = df[clean_correct]
                fl_cc = flip[clean_correct].astype(np.int8)

                cond_drift.append(df_cc)
                cond_gate.append(gate)
                cond_flip.append(fl_cc)

            if not cond_drift:
                continue
            d = np.concatenate(cond_drift)
            g = np.concatenate(cond_gate)
            f = np.concatenate(cond_flip)
            all_drift.append(d)
            all_gate.append(g)
            all_flip.append(f)
            rows.append({
                "dataset": ds,
                "backbone": bb,
                "perturbation": pert_name,
                "severity": value,
                "tag": tag,
                "n_pairs": int(d.size),
                "n_flip": int(f.sum()),
                "flip_rate": float(f.mean()) if d.size else float("nan"),
                "frac_necessary": float(np.mean(d[f == 1] >= g[f == 1] - params["tol"])) if f.sum() else float("nan"),
                "coverage_certified_safe": float(np.mean(d < g)) if d.size else float("nan"),
                "flip_rate_in_certified": float(np.mean(f[d < g])) if np.any(d < g) else float("nan"),
                "frac_preserved_certified": float(np.mean((d < g)[f == 0])) if np.any(f == 0) else float("nan"),
                "mean_drift": float(np.mean(d)),
                "mean_gate": float(np.mean(g)),
            })

    return {
        "dataset": ds,
        "backbone": bb,
        "ok": True,
        "missing": "",
        "rows": rows,
        "drift": np.concatenate(all_drift) if all_drift else np.array([], dtype=np.float64),
        "gate": np.concatenate(all_gate) if all_gate else np.array([], dtype=np.float64),
        "flip": np.concatenate(all_flip) if all_flip else np.array([], dtype=np.int8),
        "path_warning_count": path_warning_count,
    }


def run_real_from_caches(args):
    if args.results_dir:
        os.environ["WOOD_RESULTS_DIR"] = args.results_dir

    import pandas as pd
    from wood_spatial.config import BB_ORDER, TIER_A, V4_CSV, V4_FEAT_CACHE

    datasets = args.datasets.split(",") if args.datasets else list(TIER_A)
    backbones = args.backbones.split(",") if args.backbones else list(BB_ORDER)
    V4_CSV.mkdir(parents=True, exist_ok=True)

    all_drift, all_gate, all_flip = [], [], []
    by_condition = []
    missing = []
    params = {
        "results_dir": args.results_dir,
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "seed": args.seed,
        "tol": args.tol,
    }
    tasks = [(ds, bb, params) for ds in datasets for bb in backbones]

    if args.jobs > 1:
        print(f"[info] running {len(tasks)} dataset/backbone tasks with jobs={args.jobs}", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_run_real_pair_task, task): task for task in tasks}
            for fut in as_completed(futs):
                result = fut.result()
                ds = result["dataset"]
                bb = result["backbone"]
                if not result["ok"]:
                    missing.append(result["missing"])
                    print(f"[skip] {ds} / {bb}: missing clean cache", flush=True)
                    continue
                if result["path_warning_count"]:
                    print(f"[warn] {ds} / {bb}: {result['path_warning_count']} path-order warnings; assuming row alignment", flush=True)
                if result["drift"].size:
                    all_drift.append(result["drift"])
                    all_gate.append(result["gate"])
                    all_flip.append(result["flip"])
                    by_condition.extend(result["rows"])
                print(f"[done] {ds} / {bb}: conditions={len(result['rows'])}", flush=True)
    else:
        for task in tasks:
            ds, bb, _params = task
            print(f"[run] {ds} / {bb}", flush=True)
            result = _run_real_pair_task(task)
            if not result["ok"]:
                missing.append(result["missing"])
                continue
            if result["path_warning_count"]:
                print(f"[warn] {ds} / {bb}: {result['path_warning_count']} path-order warnings; assuming row alignment", flush=True)
            if result["drift"].size:
                all_drift.append(result["drift"])
                all_gate.append(result["gate"])
                all_flip.append(result["flip"])
                by_condition.extend(result["rows"])
            print(f"[done] {ds} / {bb}: cumulative_conditions={len(by_condition)}", flush=True)

    if not all_drift:
        raise RuntimeError(
            f"No usable perturbation feature caches found in {V4_FEAT_CACHE}. "
            "Run the full feature extraction/experiment pipeline first."
        )

    drift = np.concatenate(all_drift)
    gate = np.concatenate(all_gate)
    flip = np.concatenate(all_flip)
    n_pairs = int(drift.size)
    n_flip = int(flip.sum())
    frac_necessary = float(np.mean(drift[flip == 1] >= gate[flip == 1] - args.tol)) if n_flip else float("nan")
    certified = drift < gate
    preserved = flip == 0

    summary = {
        "protocol": "nearest_centroid_margin_crossing_v1",
        "datasets": ",".join(datasets),
        "backbones": ",".join(backbones),
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "split_seed": args.seed,
        "n_pairs": n_pairs,
        "n_flip": n_flip,
        "flip_rate": n_flip / n_pairs,
        "frac_necessary": frac_necessary,
        "coverage_certified_safe": float(np.mean(certified)),
        "flip_rate_in_certified": float(np.mean(flip[certified])) if certified.any() else float("nan"),
        "frac_preserved_certified": float(np.mean(certified[preserved])) if preserved.any() else float("nan"),
    }
    summary.update(_auc_summary(
        drift, gate, flip,
        max_auc_records=args.max_auc_records,
        seed=args.seed,
        skip_logit=args.skip_logit,
    ))

    out_csv = Path(args.out_csv) if args.out_csv else V4_CSV / "exp_margin_crossing_by_condition.csv"
    out_summary = Path(args.out_summary) if args.out_summary else V4_CSV / "exp_margin_crossing_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(by_condition).to_csv(out_csv, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    _make_figure(drift, gate, flip, args.fig)

    if missing:
        print(f"[warn] missing clean caches: {len(missing)} (first: {missing[0]})")
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_summary}")
    print_report(summary)
    return summary


# ----------------------------------------------------------------------------- #
# DEMO: synthetic separable features so you can verify the pipeline + theorem.
# ----------------------------------------------------------------------------- #
def make_demo(seed=0, C=12, D=64, n_per_class=120):
    rng = np.random.default_rng(seed)
    # class means with DELIBERATELY VARYING separation so margins differ
    means = rng.normal(size=(C, D))
    means = l2_normalize(means)
    scales = rng.uniform(0.05, 0.30, size=C)  # per-class intra-class spread

    def sample(n):
        labels = rng.integers(0, C, size=n)
        X = means[labels] + scales[labels][:, None] * rng.normal(size=(n, D))
        return l2_normalize(X), labels

    Xtr, ytr = sample(C * n_per_class)
    Xte, yte = sample(C * n_per_class)

    # perturbations = increasing isotropic noise added to TEST features (paired)
    conditions = []
    for sev in (0.05, 0.12, 0.22, 0.35, 0.55):
        Xp = l2_normalize(Xte + sev * rng.normal(size=Xte.shape))
        conditions.append((f"noise_sev_{sev}", Xp))
    return Xtr, ytr, Xte, yte, conditions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run synthetic self-test")
    ap.add_argument("--real", action="store_true", help="run on cached Wood Spatial feature_cache/*.npz files")
    ap.add_argument("--backbone", default="dinov2_b")
    ap.add_argument("--backbones", default="", help="comma-separated backbones; default = config BB_ORDER")
    ap.add_argument("--datasets", default="", help="comma-separated datasets; default = config TIER_A")
    ap.add_argument("--results-dir", default="", help="override WOOD_RESULTS_DIR before importing wood_spatial.config")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel dataset/backbone workers for --real; use 2-4 on Colab CPU")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--max-auc-records", type=int, default=300000,
                    help="subsample cap for logistic AUC checks; use 0 for all records")
    ap.add_argument("--skip-logit", action="store_true",
                    help="skip slow cross-validated logistic AUC; still reports raw drift and signed gate-distance AUC")
    ap.add_argument("--out-csv", default="", help="by-condition CSV path; default = results/csv/exp_margin_crossing_by_condition.csv")
    ap.add_argument("--out-summary", default="", help="summary CSV path; default = results/csv/exp_margin_crossing_summary.csv")
    ap.add_argument("--fig", default="figures/fig_margin_crossing.png")
    args = ap.parse_args()
    if args.max_auc_records <= 0:
        args.max_auc_records = None

    if args.real:
        run_real_from_caches(args)
        return
    else:  # default to demo
        if not args.demo:
            print("[info] no mode given; running --demo. Use --real for your data.")
        Xtr, ytr, Xte, yte, conds = make_demo()

    res = run_validation(Xtr, ytr, Xte, yte, conds, fig_path=args.fig)
    print_report(res)


if __name__ == "__main__":
    main()
