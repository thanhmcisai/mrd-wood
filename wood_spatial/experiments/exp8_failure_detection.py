"""
Wood Spatial — Experiment 8: Unsupervised Failure Detection
============================================================
Feature drift as an unsupervised signal for detecting recognition failure
WITHOUT requiring ground truth labels on the shifted domain.

Key insight: Feature drift is computed purely from feature spaces (no labels),
yet AUC=0.984 for detecting whether accuracy drop exceeds threshold.

Practical implication: paired/reference-based drift and reference-bank
embedding distances can monitor deployment shift before observing errors.

Usage:
    python -m wood_spatial.experiments.exp8_failure_detection
"""
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score,
)

from wood_spatial.config import BB_ORDER, V4_CSV

logger = logging.getLogger(__name__)

# Failure threshold: accuracy drop > this value is considered "failure"
FAILURE_THRESHOLD = 0.20


def _binary_label(drop_series: pd.Series, threshold: float = FAILURE_THRESHOLD) -> pd.Series:
    return (drop_series > threshold).astype(int)


def _auc_metrics(y_true, scores, label: str):
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    fpr, tpr, thresh = roc_curve(y_true, scores)
    prec, rec, pt = precision_recall_curve(y_true, scores)
    # Best F1 threshold
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s)
    return {
        'detector': label,
        'auc_roc': float(auc),
        'avg_precision': float(ap),
        'best_f1': float(f1s[best_idx]),
        'best_threshold': float(pt[best_idx]) if best_idx < len(pt) else np.nan,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'prec': prec.tolist(),
        'rec': rec.tolist(),
    }


def run_exp8(
    failure_threshold: float = FAILURE_THRESHOLD,
    save: bool = True,
) -> dict:
    df = pd.read_csv(V4_CSV / 'exp6_multilevel_table.csv').dropna(subset=['drop', 'feature_drift'])
    y = _binary_label(df['drop'], failure_threshold)
    logger.info('Failure threshold=%.2f | failure rate=%.1f%% (n=%d)',
                failure_threshold, 100 * y.mean(), len(y))

    # ── 1. ROC / AUC for each candidate detector ─────────────────────────────
    logger.info('=== Exp 8 Step 1: Detector comparison ===')
    detectors = {
        'feature_drift': df['feature_drift'],
        'delta_fgcs': df['delta_fgcs'],
        'inter_collapse': df['inter_collapse'],
        '1 - csi_bo': 1 - df['csi'].fillna(0.5),
        'cam_shift_js_distance': df.get('cam_shift_js_distance', df['cam_shift_jsd']).fillna(0),
    }
    if 'csi_hungarian' in df.columns:
        detectors['1 - csi_hungarian'] = 1 - df['csi_hungarian'].fillna(0.5)
    if 'csi_permutation_gap' in df.columns:
        detectors['csi_permutation_gap'] = df['csi_permutation_gap'].fillna(0)
    if 'cam_shift_js_divergence' in df.columns:
        detectors['cam_shift_js_divergence'] = df['cam_shift_js_divergence'].fillna(0)

    ref_path = V4_CSV / 'exp8_additional_baselines.csv'
    if ref_path.exists():
        ref = pd.read_csv(ref_path).rename(columns={'accuracy_drop': 'drop'})
        ref['severity_key'] = ref['perturbation'].astype(str)
        # exp8_additional_baselines stores perturbation as cache tag. It is kept
        # as a separate detector summary by exp8_baselines.py; avoid lossy merge here.
    auc_rows = []
    roc_curves = {}
    pr_curves = {}
    for name, scores in detectors.items():
        valid = scores.notna()
        if valid.sum() < 30:
            continue
        try:
            metrics = _auc_metrics(y[valid].values, scores[valid].values, name)
            auc_rows.append({
                'detector': name,
                'auc_roc': metrics['auc_roc'],
                'avg_precision': metrics['avg_precision'],
                'best_f1': metrics['best_f1'],
                'best_threshold': metrics['best_threshold'],
                'n': int(valid.sum()),
            })
            roc_curves[name] = {'fpr': metrics['fpr'], 'tpr': metrics['tpr']}
            pr_curves[name] = {'prec': metrics['prec'], 'rec': metrics['rec']}
            logger.info('  %-22s AUC=%.4f  AP=%.4f  F1=%.4f',
                        name, metrics['auc_roc'], metrics['avg_precision'], metrics['best_f1'])
        except Exception as e:
            logger.warning('  %s: failed (%s)', name, e)

    df_auc = pd.DataFrame(auc_rows).sort_values('auc_roc', ascending=False)

    # ── 2. Threshold sensitivity analysis ────────────────────────────────────
    logger.info('=== Exp 8 Step 2: Threshold sensitivity ===')
    thresh_rows = []
    scores_fd = df['feature_drift'].values
    for t_fail in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        y_t = _binary_label(df['drop'], t_fail)
        if y_t.sum() < 5 or (1 - y_t).sum() < 5:
            continue
        auc = roc_auc_score(y_t.values, scores_fd)
        ap = average_precision_score(y_t.values, scores_fd)
        thresh_rows.append({
            'failure_threshold': t_fail,
            'failure_rate': float(y_t.mean()),
            'auc_roc': float(auc),
            'avg_precision': float(ap),
        })
        logger.info('  drop>%.2f (%.0f%% failure): AUC=%.4f  AP=%.4f',
                    t_fail, 100 * y_t.mean(), auc, ap)
    df_thresh = pd.DataFrame(thresh_rows)

    # ── 3. Per-backbone detection performance ─────────────────────────────────
    logger.info('=== Exp 8 Step 3: Per-backbone AUC ===')
    bb_auc_rows = []
    for bb, sub in df.groupby('backbone'):
        sub = sub.dropna(subset=['drop', 'feature_drift'])
        y_bb = _binary_label(sub['drop'], failure_threshold)
        if y_bb.sum() < 3 or (1 - y_bb).sum() < 3:
            continue
        try:
            auc = roc_auc_score(y_bb.values, sub['feature_drift'].values)
            ap = average_precision_score(y_bb.values, sub['feature_drift'].values)
            bb_auc_rows.append({'backbone': bb, 'auc_roc': float(auc), 'avg_precision': float(ap), 'n': len(sub)})
            logger.info('  %-20s AUC=%.4f  AP=%.4f', bb, auc, ap)
        except Exception:
            pass
    df_bb_auc = pd.DataFrame(bb_auc_rows)

    # ── 4. Per-perturbation detection performance ─────────────────────────────
    logger.info('=== Exp 8 Step 4: Per-perturbation AUC ===')
    pert_auc_rows = []
    for pert, sub in df.groupby('perturbation'):
        sub = sub.dropna(subset=['drop', 'feature_drift'])
        y_p = _binary_label(sub['drop'], failure_threshold)
        if y_p.sum() < 3 or (1 - y_p).sum() < 3:
            continue
        try:
            auc = roc_auc_score(y_p.values, sub['feature_drift'].values)
            pert_auc_rows.append({'perturbation': pert, 'auc_roc': float(auc), 'failure_rate': float(y_p.mean()), 'n': len(sub)})
            logger.info('  %-20s AUC=%.4f  (failure rate=%.1f%%)', pert, auc, 100 * y_p.mean())
        except Exception:
            pass
    df_pert_auc = pd.DataFrame(pert_auc_rows)

    # ── 5. Operating point analysis ───────────────────────────────────────────
    logger.info('=== Exp 8 Step 5: Operating points ===')
    op_rows = []
    for t_detect in np.arange(0.05, 0.71, 0.05):
        pred = (df['feature_drift'] > t_detect).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        op_rows.append({
            'threshold': round(t_detect, 2),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'flagged_rate': round(float(pred.mean()), 4),
        })
    df_op = pd.DataFrame(op_rows)
    logger.info('\n%s', df_op[['threshold', 'precision', 'recall', 'f1', 'flagged_rate']].to_string(index=False))

    results = {
        'detector_auc': df_auc,
        'threshold_sensitivity': df_thresh,
        'backbone_auc': df_bb_auc,
        'perturbation_auc': df_pert_auc,
        'operating_points': df_op,
        'roc_curves': roc_curves,
        'pr_curves': pr_curves,
    }

    if save:
        df_auc.to_csv(V4_CSV / 'exp8_detector_auc.csv', index=False)
        df_thresh.to_csv(V4_CSV / 'exp8_threshold_sensitivity.csv', index=False)
        df_bb_auc.to_csv(V4_CSV / 'exp8_backbone_auc.csv', index=False)
        df_pert_auc.to_csv(V4_CSV / 'exp8_perturbation_auc.csv', index=False)
        df_op.to_csv(V4_CSV / 'exp8_operating_points.csv', index=False)
        logger.info('Saved Exp 8 results to %s', V4_CSV)

    return results


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    t0 = time.time()
    results = run_exp8()
    logger.info('Exp 8 done in %.1f min', (time.time() - t0) / 60)

    print('\n=== Detector AUC Summary ===')
    print(results['detector_auc'][['detector', 'auc_roc', 'avg_precision', 'best_f1']].to_string(index=False))
    print('\n=== Best Operating Point (feature_drift) ===')
    df_op = results['operating_points']
    best = df_op.loc[df_op['f1'].idxmax()]
    print(f"  threshold={best['threshold']:.2f}: precision={best['precision']:.3f} recall={best['recall']:.3f} F1={best['f1']:.3f}")


if __name__ == '__main__':
    main()
