#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
competitor_switching_analysis.py   (TIER 2)
===========================================
Answers reviewer Q1: the margin-direction check (aligned_margin_analysis.py) showed
that the region hatGamma(z')<=0 deterministically explains ~53.3% of flips (the
true class loses to its CLEAN nearest competitor c*). This script characterizes
the REMAINING ~46.7% of flips, where a DIFFERENT class overtakes the true class
("competitor switching"), to show those failures are still structured rather than
random.

For each clean-correct test sample and perturbation, using the prototype
(nearest-centroid) rule with prototypes from the clean TRAIN split:
  - winner   = argmax_c <z'_hat, p_c>          (predicted class at the shifted feature)
  - flip     = winner != y
  - clean_rank(winner) = rank of the winning class in the CLEAN score ranking of
                         this sample (rank 1 = true class, rank 2 = clean nearest
                         competitor c*, >=3 = a further class).
      * flip with clean_rank(winner)==2  -> "first-order" flip (caught by hatGamma<=0)
      * flip with clean_rank(winner)>=3  -> "competitor-switching" flip

REPORTED (fills the \todo in the new TIER-2 results paragraph):
  - first_order_frac           : share of flips with winner == c* (~ matches 53.3%)
  - switch_frac                : 1 - first_order_frac (~46.7%)
  - switch_winner_in_top3/top5 : among switching flips, share whose winner was
                                  already in the clean top-3 / top-5 neighbours
                                  (HIGH => failures stay local/among confusable
                                   classes => structured, not random)
  - median_clean_rank_switch   : median clean rank of the winner for switching flips
  - first_order_frac_by_family : first-order share broken down by perturbation
                                  family (which stressors cause more chaotic,
                                  multi-competitor failures)
  - figure: histogram of winner clean-rank for flips + per-family bar.

Verify the logic with `--demo`; run on cached features with `--real` or by
omitting the mode flag.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

def l2n(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)

def build_prototypes(train_feats, train_labels, classes):
    Z = l2n(train_feats)
    P = np.zeros((len(classes), Z.shape[1]))
    for j, c in enumerate(classes):
        P[j] = Z[train_labels == c].mean(0)
    return l2n(P)

def clean_scores_ranks(clean_feats, P):
    """S[i,j] = <z_i, p_j>; rank of each class (1=best)."""
    S = l2n(clean_feats) @ P.T
    order = np.argsort(-S, axis=1)                 # columns sorted best->worst
    ranks = np.empty_like(order)
    rows = np.arange(S.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, S.shape[1] + 1)[None, :]  # rank per column
    return S, ranks

def analyze(clean_train_feats, clean_train_labels,
            clean_test_feats, clean_test_labels,
            perturbed_conditions, fig_path=None):
    classes = sorted(np.unique(clean_train_labels).tolist())
    col = {c: j for j, c in enumerate(classes)}
    P = build_prototypes(clean_train_feats, clean_train_labels, classes)

    S_clean, ranks_clean = clean_scores_ranks(clean_test_feats, P)
    pred_clean = np.array([classes[j] for j in S_clean.argmax(1)])
    y = clean_test_labels
    clean_correct = pred_clean == y

    winner_ranks, families = [], []
    for name, pert in perturbed_conditions:
        Sp = l2n(pert) @ P.T
        win_col = Sp.argmax(1)
        winner = np.array([classes[j] for j in win_col])
        flip = clean_correct & (winner != y)
        if not flip.any():
            continue
        idx = np.where(flip)[0]
        wr = ranks_clean[idx, win_col[idx]]        # clean rank of the winning class
        winner_ranks.append(wr)
        fam = name.split("_")[0] if "_" in name else name
        families.append(np.array([fam] * len(idx)))
    if not winner_ranks:
        print("[warn] no flips found"); return {}
    wr = np.concatenate(winner_ranks)
    fam = np.concatenate(families)

    n = wr.size
    first_order = (wr == 2)
    switch = (wr >= 3)
    res = {
        "n_flips": int(n),
        "first_order_frac": float(first_order.mean()),
        "switch_frac": float(switch.mean()),
        "switch_winner_in_top3": float(np.mean(wr[switch] <= 3)) if switch.any() else float("nan"),
        "switch_winner_in_top5": float(np.mean(wr[switch] <= 5)) if switch.any() else float("nan"),
        "median_clean_rank_switch": float(np.median(wr[switch])) if switch.any() else float("nan"),
    }
    # per-family first-order share
    by_fam = {}
    for f in sorted(set(fam.tolist())):
        m = fam == f
        by_fam[f] = float((wr[m] == 2).mean())
    res["first_order_frac_by_family"] = by_fam

    if fig_path:
        _figure(wr, by_fam, fig_path)
    return res

def _figure(wr, by_fam, fig_path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] no matplotlib: {e}"); return
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=160)
    capped = np.minimum(wr, 11)
    axL.hist(capped, bins=np.arange(1.5, 12.5, 1), color="#E45756",
             edgecolor="white", rwidth=0.9)
    axL.axvline(2, ls="--", c="k", lw=1)
    axL.set_xlabel("clean rank of the winning class (2 = nearest competitor)")
    axL.set_ylabel("number of flips")
    axL.set_title("(a) Low-rank confusions with a long tail")
    axL.set_xticks(list(range(2, 11)) + [11]); axL.set_xticklabels([str(i) for i in range(2, 11)] + ["11+"])
    fams = list(by_fam.keys()); vals = [by_fam[f] for f in fams]
    axR.barh(fams, vals, color="#4C78A8")
    axR.set_xlim(0, 1); axR.set_xlabel("nearest-competitor winner share")
    axR.set_title("(b) Single-competitor share varies by stressor")
    fig.tight_layout(); fig.savefig(fig_path, bbox_inches="tight"); plt.close(fig)
    print(f"[ok] figure saved: {fig_path}")

def report(res):
    print("\n=== TIER-2 competitor-switching analysis ===")
    print(f"  flips analyzed: {res.get('n_flips')}")
    print(f"  first-order (winner=c*)      : {100*res['first_order_frac']:.1f}%  (-> matches 53.3%)")
    print(f"  competitor-switching         : {100*res['switch_frac']:.1f}%")
    print(f"   .. winner in clean top-3    : {100*res['switch_winner_in_top3']:.1f}%")
    print(f"   .. winner in clean top-5    : {100*res['switch_winner_in_top5']:.1f}%")
    print(f"   .. median clean rank        : {res['median_clean_rank_switch']:.1f}")
    print("  first-order share by family:")
    for f, v in res["first_order_frac_by_family"].items():
        print(f"     {f:<16s}: {100*v:.1f}%")

def _run_real_pair(task):
    ds, bb, params = task
    if params["results_dir"]:
        os.environ["WOOD_RESULTS_DIR"] = params["results_dir"]

    import pandas as pd
    from sklearn.model_selection import StratifiedShuffleSplit
    from wood_spatial.config import PERTURB_CONFIGS, V4_FEAT_CACHE
    from wood_spatial.experiments.exp_margin_crossing_validation import (
        _cache_tag_for,
        _load_cache_np,
    )

    checkpoint_dir = Path(params["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_npz = checkpoint_dir / f"{ds}__{bb}.npz"
    ckpt_csv = checkpoint_dir / f"{ds}__{bb}.csv"
    if not params["force"] and ckpt_npz.exists() and ckpt_csv.exists():
        data = np.load(ckpt_npz, allow_pickle=False)
        print(f"[skip] {ds} / {bb}: loaded checkpoint", flush=True)
        return {
            "dataset": ds,
            "backbone": bb,
            "ok": True,
            "missing": "",
            "rows": pd.read_csv(ckpt_csv).to_dict("records"),
            "winner_ranks": data["winner_ranks"],
            "families": data["families"].astype(str),
            "path_warning_count": int(data["path_warning_count"][0]),
            "cached": True,
        }

    try:
        clean, labels, clean_paths = _load_cache_np(V4_FEAT_CACHE, bb, ds, "original")
    except FileNotFoundError as exc:
        return {"dataset": ds, "backbone": bb, "ok": False, "missing": str(exc)}

    splitter = StratifiedShuffleSplit(
        n_splits=params["n_splits"],
        test_size=params["test_size"],
        random_state=params["seed"],
    )
    split_indices = list(splitter.split(clean, labels))
    classes = sorted(np.unique(labels).tolist())
    split_cache = []
    for split_id, (train_idx, test_idx) in enumerate(split_indices):
        P = build_prototypes(clean[train_idx], labels[train_idx], classes)
        S_clean, ranks_clean = clean_scores_ranks(clean[test_idx], P)
        pred_clean = np.array([classes[j] for j in S_clean.argmax(axis=1)])
        y = labels[test_idx]
        split_cache.append({
            "split_id": split_id,
            "test_idx": test_idx,
            "P": P,
            "ranks_clean": ranks_clean,
            "clean_correct": pred_clean == y,
            "y": y,
        })
    rows = []
    all_ranks, all_families = [], []
    path_warning_count = 0
    print(f"[run] {ds} / {bb}: splits={len(split_cache)}", flush=True)

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

            cond_ranks = []
            for sc in split_cache:
                test_idx = sc["test_idx"]
                Sp = l2n(shifted[test_idx]) @ sc["P"].T
                win_col = Sp.argmax(axis=1)
                winner = np.array([classes[j] for j in win_col])
                y = sc["y"]
                clean_correct = sc["clean_correct"]
                flip = clean_correct & (winner != y)
                if not flip.any():
                    continue
                idx = np.where(flip)[0]
                cond_ranks.append(sc["ranks_clean"][idx, win_col[idx]])

            if not cond_ranks:
                continue
            wr = np.concatenate(cond_ranks)
            switch = wr >= 3
            row = {
                "dataset": ds,
                "backbone": bb,
                "perturbation": pert_name,
                "severity": value,
                "tag": tag,
                "n_flip": int(wr.size),
                "first_order_frac": float(np.mean(wr == 2)),
                "switch_frac": float(np.mean(switch)),
                "switch_winner_in_top3": float(np.mean(wr[switch] <= 3)) if switch.any() else float("nan"),
                "switch_winner_in_top5": float(np.mean(wr[switch] <= 5)) if switch.any() else float("nan"),
                "median_clean_rank_switch": float(np.median(wr[switch])) if switch.any() else float("nan"),
            }
            rows.append(row)
            all_ranks.append(wr)
            all_families.append(np.array([pert_name] * wr.size, dtype="<U64"))
        print(f"[progress] {ds} / {bb}: finished {pert_name}, rows={len(rows)}", flush=True)

    winner_ranks = np.concatenate(all_ranks) if all_ranks else np.array([], dtype=int)
    families = np.concatenate(all_families) if all_families else np.array([], dtype="<U64")
    if rows and winner_ranks.size:
        pd.DataFrame(rows).to_csv(ckpt_csv, index=False)
        np.savez_compressed(
            ckpt_npz,
            winner_ranks=winner_ranks.astype(np.int16),
            families=families.astype("<U64"),
            path_warning_count=np.array([path_warning_count], dtype=np.int32),
        )
        print(f"[ok] checkpoint saved: {ckpt_npz}", flush=True)

    return {
        "dataset": ds,
        "backbone": bb,
        "ok": True,
        "missing": "",
        "rows": rows,
        "winner_ranks": winner_ranks,
        "families": families,
        "path_warning_count": path_warning_count,
        "cached": False,
    }


def run_real(args):
    if args.results_dir:
        os.environ["WOOD_RESULTS_DIR"] = args.results_dir

    import pandas as pd
    from wood_spatial.config import BB_ORDER, TIER_A, V4_CSV

    out_summary = Path(args.out_summary) if args.out_summary else V4_CSV / "exp_competitor_switching_summary.csv"
    out_csv = Path(args.out_csv) if args.out_csv else V4_CSV / "exp_competitor_switching_by_condition.csv"
    fig_path = Path(args.fig) if args.fig else None
    if (
        not args.force
        and out_summary.exists()
        and out_csv.exists()
        and (fig_path is None or fig_path.exists())
    ):
        print(f"[skip] outputs already exist: {out_summary}, {out_csv}", flush=True)
        summary = pd.read_csv(out_summary).iloc[0].to_dict()
        by_fam = {}
        for item in str(summary.get("first_order_frac_by_family", "")).split(";"):
            if ":" in item:
                key, value = item.split(":", 1)
                by_fam[key] = float(value)
        report({**summary, "first_order_frac_by_family": by_fam})
        return summary

    datasets = args.datasets.split(",") if args.datasets else list(TIER_A)
    backbones = args.backbones.split(",") if args.backbones else (
        [args.backbone] if args.backbone else list(BB_ORDER)
    )
    params = {
        "results_dir": args.results_dir,
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "seed": args.seed,
        "force": args.force,
        "checkpoint_dir": str(V4_CSV / "exp_competitor_switching_checkpoints"),
    }
    tasks = [(ds, bb, params) for ds in datasets for bb in backbones]
    all_ranks, all_families, rows, missing = [], [], [], []

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_run_real_pair, task): task for task in tasks}
            for fut in as_completed(futs):
                res = fut.result()
                if not res["ok"]:
                    missing.append(res["missing"])
                    print(f"[skip] {res['dataset']} / {res['backbone']}: missing clean cache", flush=True)
                    continue
                if res.get("path_warning_count"):
                    print(f"[warn] {res['dataset']} / {res['backbone']}: path-order warnings; assuming row alignment", flush=True)
                rows.extend(res["rows"])
                if res["winner_ranks"].size:
                    all_ranks.append(res["winner_ranks"])
                    all_families.append(res["families"])
                print(f"[done] {res['dataset']} / {res['backbone']}: conditions={len(res['rows'])}", flush=True)
    else:
        for task in tasks:
            print(f"[run] {task[0]} / {task[1]}", flush=True)
            res = _run_real_pair(task)
            if not res["ok"]:
                missing.append(res["missing"])
                print(f"[skip] {task[0]} / {task[1]}: missing clean cache", flush=True)
                continue
            rows.extend(res["rows"])
            if res["winner_ranks"].size:
                all_ranks.append(res["winner_ranks"])
                all_families.append(res["families"])
            print(f"[done] {task[0]} / {task[1]}: cumulative_conditions={len(rows)}", flush=True)

    if not all_ranks:
        raise RuntimeError("No usable perturbation feature caches found.")

    wr = np.concatenate(all_ranks)
    fam = np.concatenate(all_families)
    switch = wr >= 3
    by_fam = {}
    for f in sorted(set(fam.tolist())):
        m = fam == f
        by_fam[f] = float(np.mean(wr[m] == 2))
    summary = {
        "protocol": "nearest_centroid_competitor_switching_v1",
        "datasets": ",".join(datasets),
        "backbones": ",".join(backbones),
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "split_seed": args.seed,
        "n_flips": int(wr.size),
        "first_order_frac": float(np.mean(wr == 2)),
        "switch_frac": float(np.mean(switch)),
        "switch_winner_in_top3": float(np.mean(wr[switch] <= 3)) if switch.any() else float("nan"),
        "switch_winner_in_top5": float(np.mean(wr[switch] <= 5)) if switch.any() else float("nan"),
        "median_clean_rank_switch": float(np.median(wr[switch])) if switch.any() else float("nan"),
        "first_order_frac_by_family": ";".join(f"{k}:{v:.6f}" for k, v in by_fam.items()),
    }

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    if args.fig:
        _figure(wr, by_fam, args.fig)
    if missing:
        print(f"[warn] missing clean caches: {len(missing)} (first: {missing[0]})")
    print(f"[ok] wrote {out_summary}")
    print(f"[ok] wrote {out_csv}")
    report({**summary, "first_order_frac_by_family": by_fam})
    return summary

def make_demo(seed=0, C=40, D=128, npc=150):
    rng = np.random.default_rng(seed)
    means = l2n(rng.normal(size=(C, D))); scales = rng.uniform(0.05, 0.30, C)
    def samp(n):
        y = rng.integers(0, C, n)
        return l2n(means[y] + scales[y][:, None] * rng.normal(size=(n, D))), y
    Xtr, ytr = samp(C*npc); Xte, yte = samp(C*npc)
    conds = [(f"noise_{s}", l2n(Xte + s*rng.normal(size=Xte.shape)))
             for s in (0.05, 0.12, 0.22, 0.35, 0.55)]
    # add a structured "blur"-like family (low-rank shift) to vary determinism
    U = l2n(rng.normal(size=(8, D)))
    conds += [(f"blur_{s}", l2n(Xte + s*(rng.normal(size=(len(Xte),8))@U)))
              for s in (0.2, 0.5)]
    return Xtr, ytr, Xte, yte, conds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--real", action="store_true")
    ap.add_argument("--backbone", default="", help="single backbone; default with --real = all backbones")
    ap.add_argument("--backbones", default="", help="comma-separated backbones; overrides --backbone")
    ap.add_argument("--datasets", default="", help="comma-separated datasets; default = Tier-A")
    ap.add_argument("--results-dir", default="", help="override WOOD_RESULTS_DIR")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out-summary", default="")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--fig", default="results/figures/fig_competitor_switching.png")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if final outputs or per-pair checkpoints exist")
    a = ap.parse_args()
    if a.demo:
        Xtr, ytr, Xte, yte, conds = make_demo()
    else:
        if not a.real:
            print("[info] no mode given; running --real. Use --demo for a synthetic self-test.")
        run_real(a)
        return
    report(analyze(Xtr, ytr, Xte, yte, conds, fig_path=a.fig))

if __name__ == "__main__":
    main()
