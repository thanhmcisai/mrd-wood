#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augmentation_probe_experiment.py   (TIER 3)
===========================================
Answers reviewer concern #4 / Q3: "does training with the perturbations as
augmentation close the gap, and does the drift->drop relationship persist?"

KEY POINT (why this needs NO new data and NO backbone retraining): the backbone is
frozen, so input-space augmentation training of the head is EXACTLY training the
head on the cached perturbed features. We therefore train the decision head two
ways on cached features only:
  - BASELINE  : head trained on clean TRAIN features.
  - AUGMENTED : head trained on clean TRAIN + perturbed TRAIN features
                (= input-augmentation training under a frozen backbone).
Both are evaluated on clean TEST and perturbed TEST features.

REPORTED (fills the \todo in the new TIER-3 results subsection):
  - clean_acc / pert_acc / gap for BASELINE and AUGMENTED  (does augmentation
    raise perturbed accuracy and shrink the clean->perturbed gap?)
  - drift_drop_r_baseline / drift_drop_r_augmented : per-condition Pearson r
    between mean feature drift and accuracy drop under each head (does the
    drift->drop relationship persist after augmentation?)

Needs perturbed TRAIN features. If you only cached perturbed TEST features,
generate perturbed TRAIN features once with the same pipeline (no new data, just
the existing perturbation transforms applied to the train split). Verify with
`--demo`; run on cached features with `--real` or by omitting the mode flag.
"""
import argparse
import os
from pathlib import Path
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr
except Exception:
    LogisticRegression = None

def l2n(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)

def _fit(X, y, head="ridge", max_iter=300):
    sc = StandardScaler().fit(X)
    if head == "logistic":
        clf = LogisticRegression(
            max_iter=max_iter,
            C=1.0,
            solver="saga",
            multi_class="auto",
            n_jobs=-1,
            verbose=0,
        )
    else:
        clf = RidgeClassifier(solver="lsqr")
    clf.fit(sc.transform(X), y)
    return sc, clf

def _acc(sc, clf, X, y):
    return float((clf.predict(sc.transform(X)) == y).mean())

def run(clean_train, ytr, pert_train_conds,
        clean_test, yte, pert_test_conds, head="ridge", max_iter=300):
    Xtr = l2n(clean_train); Xte = l2n(clean_test)

    # BASELINE head: clean train only
    sc_b, clf_b = _fit(Xtr, ytr, head=head, max_iter=max_iter)

    # AUGMENTED head: clean train + perturbed train (frozen-backbone augmentation)
    Xaug = [Xtr]; yaug = [ytr]
    for _, Xp in pert_train_conds:
        Xaug.append(l2n(Xp)); yaug.append(ytr)
    sc_a, clf_a = _fit(np.vstack(Xaug), np.concatenate(yaug), head=head, max_iter=max_iter)

    def eval_head(sc, clf):
        clean_acc = _acc(sc, clf, Xte, yte)
        drift_list, drop_list, accs = [], [], []
        for _, Xp in pert_test_conds:
            Xp = l2n(Xp)
            a = _acc(sc, clf, Xp, yte)
            accs.append(a)
            drop_list.append(clean_acc - a)
            drift_list.append(float((1 - np.sum(Xte * Xp, axis=1)).mean()))
        pert_acc = float(np.mean(accs))
        r = float(pearsonr(drift_list, drop_list)[0]) if len(drop_list) > 2 else float("nan")
        return clean_acc, pert_acc, clean_acc - pert_acc, r

    cb, pb, gb, rb = eval_head(sc_b, clf_b)
    ca, pa, ga, ra = eval_head(sc_a, clf_a)
    return {
        "baseline":  {"clean_acc": cb, "pert_acc": pb, "gap": gb, "drift_drop_r": rb},
        "augmented": {"clean_acc": ca, "pert_acc": pa, "gap": ga, "drift_drop_r": ra},
        "gap_closed_abs": gb - ga,
        "pert_acc_gain": pa - pb,
    }

def report(res):
    b, a = res["baseline"], res["augmented"]
    print("\n=== TIER-3 augmentation-probe experiment (frozen backbone) ===")
    print(f"  {'':10s} clean_acc  pert_acc   gap    drift->drop r")
    print(f"  baseline  : {b['clean_acc']:.3f}     {b['pert_acc']:.3f}   {b['gap']:.3f}   {b['drift_drop_r']:.3f}")
    print(f"  augmented : {a['clean_acc']:.3f}     {a['pert_acc']:.3f}   {a['gap']:.3f}   {a['drift_drop_r']:.3f}")
    print(f"  -> perturbed-acc gain   : {res['pert_acc_gain']:+.3f}")
    print(f"  -> gap closed (abs)     : {res['gap_closed_abs']:+.3f}")
    print(f"  -> drift->drop persists : {('yes' if a['drift_drop_r']>0.4 else 'check')} "
          f"(r={a['drift_drop_r']:.3f} after augmentation)")

def run_real(args):
    if args.results_dir:
        os.environ["WOOD_RESULTS_DIR"] = args.results_dir

    import pandas as pd
    from sklearn.model_selection import StratifiedShuffleSplit
    from scipy.stats import pearsonr
    from wood_spatial.config import PERTURB_CONFIGS, V4_CSV, V4_FEAT_CACHE
    from wood_spatial.experiments.exp_margin_crossing_validation import (
        _cache_tag_for,
        _load_cache_np,
    )

    ds = args.dataset or "WRD25"
    bb = args.backbone
    out_summary = Path(args.out_summary) if args.out_summary else V4_CSV / "exp_augmentation_probe_summary.csv"
    out_csv = Path(args.out_csv) if args.out_csv else V4_CSV / "exp_augmentation_probe_by_condition.csv"
    if not args.force and out_summary.exists() and out_csv.exists():
        print(f"[skip] outputs already exist: {out_summary}, {out_csv}", flush=True)
        s = pd.read_csv(out_summary).iloc[0].to_dict()
        report({
            "baseline": {
                "clean_acc": s["baseline_clean_acc"],
                "pert_acc": s["baseline_pert_acc"],
                "gap": s["baseline_gap"],
                "drift_drop_r": s["baseline_drift_drop_r"],
            },
            "augmented": {
                "clean_acc": s["augmented_clean_acc"],
                "pert_acc": s["augmented_pert_acc"],
                "gap": s["augmented_gap"],
                "drift_drop_r": s["augmented_drift_drop_r"],
            },
            "gap_closed_abs": s["gap_closed_abs"],
            "pert_acc_gain": s["pert_acc_gain"],
        })
        return s

    clean, labels, clean_paths = _load_cache_np(V4_FEAT_CACHE, bb, ds, "original")
    shifted_all = []
    print(f"[run] augmentation probe: dataset={ds} backbone={bb} head={args.head}", flush=True)
    for pert_name, pcfg in PERTURB_CONFIGS.items():
        for value in pcfg["values"]:
            tag = _cache_tag_for(pert_name, value)
            try:
                shifted, shifted_labels, shifted_paths = _load_cache_np(V4_FEAT_CACHE, bb, ds, tag)
            except FileNotFoundError:
                continue
            if len(shifted) != len(clean) or not np.array_equal(labels, shifted_labels):
                raise RuntimeError(f"Cache order/label mismatch for {bb}/{ds}/{tag}")
            if not np.array_equal(clean_paths, shifted_paths):
                print(f"[warn] {bb}/{ds}/{tag}: path order differs; assuming row alignment", flush=True)
            shifted_all.append((pert_name, value, tag, shifted))
    if not shifted_all:
        raise RuntimeError(f"No perturbation caches found for {bb}/{ds} in {V4_FEAT_CACHE}")
    print(f"[info] loaded {len(shifted_all)} perturbation conditions", flush=True)

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.seed,
    )
    rows = []
    split_summaries = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(clean, labels)):
        Xtr = l2n(clean[train_idx])
        Xte = l2n(clean[test_idx])
        ytr = labels[train_idx]
        yte = labels[test_idx]

        sc_b, clf_b = _fit(Xtr, ytr, head=args.head, max_iter=args.max_iter)
        Xaug = [Xtr]
        yaug = [ytr]
        for _, _, _, shifted in shifted_all:
            Xaug.append(l2n(shifted[train_idx]))
            yaug.append(ytr)
        print(f"[fit] split {split_id}: augmented rows={sum(len(x) for x in Xaug)}", flush=True)
        sc_a, clf_a = _fit(
            np.vstack(Xaug),
            np.concatenate(yaug),
            head=args.head,
            max_iter=args.max_iter,
        )

        clean_b = _acc(sc_b, clf_b, Xte, yte)
        clean_a = _acc(sc_a, clf_a, Xte, yte)
        pert_b, pert_a = [], []
        for pert_name, value, tag, shifted in shifted_all:
            Xp = l2n(shifted[test_idx])
            drift = float((1 - np.sum(Xte * Xp, axis=1)).mean())
            acc_b = _acc(sc_b, clf_b, Xp, yte)
            acc_a = _acc(sc_a, clf_a, Xp, yte)
            pert_b.append(acc_b)
            pert_a.append(acc_a)
            rows.append({
                "dataset": ds,
                "backbone": bb,
                "split": split_id,
                "perturbation": pert_name,
                "severity": value,
                "tag": tag,
                "mean_drift": drift,
                "baseline_clean_acc": clean_b,
                "baseline_pert_acc": acc_b,
                "baseline_drop": clean_b - acc_b,
                "augmented_clean_acc": clean_a,
                "augmented_pert_acc": acc_a,
                "augmented_drop": clean_a - acc_a,
            })
        split_summaries.append({
            "baseline_clean_acc": clean_b,
            "baseline_pert_acc": float(np.mean(pert_b)),
            "baseline_gap": clean_b - float(np.mean(pert_b)),
            "augmented_clean_acc": clean_a,
            "augmented_pert_acc": float(np.mean(pert_a)),
            "augmented_gap": clean_a - float(np.mean(pert_a)),
        })
        print(f"[done] split {split_id}: baseline pert={np.mean(pert_b):.3f}, augmented pert={np.mean(pert_a):.3f}", flush=True)

    df = pd.DataFrame(rows)
    sb = pd.DataFrame(split_summaries)
    rb = float(pearsonr(df["mean_drift"], df["baseline_drop"])[0]) if len(df) > 2 else float("nan")
    ra = float(pearsonr(df["mean_drift"], df["augmented_drop"])[0]) if len(df) > 2 else float("nan")
    summary = {
        "protocol": f"frozen_{args.head}_probe_clean_vs_augmented_features_v1",
        "dataset": ds,
        "backbone": bb,
        "head": args.head,
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "split_seed": args.seed,
        "n_conditions": int(len(shifted_all)),
        "baseline_clean_acc": float(sb["baseline_clean_acc"].mean()),
        "baseline_pert_acc": float(sb["baseline_pert_acc"].mean()),
        "baseline_gap": float(sb["baseline_gap"].mean()),
        "baseline_drift_drop_r": rb,
        "augmented_clean_acc": float(sb["augmented_clean_acc"].mean()),
        "augmented_pert_acc": float(sb["augmented_pert_acc"].mean()),
        "augmented_gap": float(sb["augmented_gap"].mean()),
        "augmented_drift_drop_r": ra,
    }
    summary["pert_acc_gain"] = summary["augmented_pert_acc"] - summary["baseline_pert_acc"]
    summary["gap_closed_abs"] = summary["baseline_gap"] - summary["augmented_gap"]

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    df.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_summary}")
    print(f"[ok] wrote {out_csv}")
    report({
        "baseline": {
            "clean_acc": summary["baseline_clean_acc"],
            "pert_acc": summary["baseline_pert_acc"],
            "gap": summary["baseline_gap"],
            "drift_drop_r": summary["baseline_drift_drop_r"],
        },
        "augmented": {
            "clean_acc": summary["augmented_clean_acc"],
            "pert_acc": summary["augmented_pert_acc"],
            "gap": summary["augmented_gap"],
            "drift_drop_r": summary["augmented_drift_drop_r"],
        },
        "gap_closed_abs": summary["gap_closed_abs"],
        "pert_acc_gain": summary["pert_acc_gain"],
    })
    return summary

def make_demo(seed=0, C=20, D=128, npc=200):
    rng = np.random.default_rng(seed)
    means = l2n(rng.normal(size=(C, D))); scales = rng.uniform(0.05, 0.25, C)
    def samp(n):
        y = rng.integers(0, C, n)
        return l2n(means[y] + scales[y][:, None]*rng.normal(size=(n, D))), y
    Xtr, ytr = samp(C*npc); Xte, yte = samp(C*npc)
    def perturb(X, s): return l2n(X + s*rng.normal(size=X.shape))
    sevs = [0.05, 0.12, 0.22, 0.35, 0.55]
    ptr = [(f"noise_{s}", perturb(Xtr, s)) for s in sevs]
    pte = [(f"noise_{s}", perturb(Xte, s)) for s in sevs]
    return Xtr, ytr, ptr, Xte, yte, pte

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--real", action="store_true")
    ap.add_argument("--backbone", default="dinov2_b"); ap.add_argument("--dataset", default=None)
    ap.add_argument("--results-dir", default="", help="override WOOD_RESULTS_DIR")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--head", choices=["ridge", "logistic"], default="ridge",
                    help="linear head to train; ridge is the fast default")
    ap.add_argument("--max-iter", type=int, default=300,
                    help="max iterations for --head logistic")
    ap.add_argument("--out-summary", default="")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if output CSV files already exist")
    a = ap.parse_args()
    if a.demo:
        Xtr, ytr, ptr, Xte, yte, pte = make_demo()
    else:
        if not a.real:
            print("[info] no mode given; running --real. Use --demo for a synthetic self-test.")
        run_real(a)
        return
    report(run(Xtr, ytr, ptr, Xte, yte, pte, head=a.head, max_iter=a.max_iter))

if __name__ == "__main__":
    main()
