"""Metric helpers shared by training and the dashboard."""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def tune_threshold(y_true, proba, beta: float = 2.0, min_precision: float = 0.0) -> float:
    """Pick the threshold maximising F-beta on the PR curve, subject to a precision floor."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall have one more element than thresholds; drop the final (recall=0) point.
    p, r = precision[:-1], recall[:-1]
    b2 = beta ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where((b2 * p + r) > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
    fbeta = np.where(p >= min_precision, fbeta, -1.0)
    if fbeta.max() < 0:  # no threshold meets the floor; fall back to unconstrained
        fbeta = np.where((b2 * p + r) > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
    return float(thresholds[int(np.argmax(fbeta))])
