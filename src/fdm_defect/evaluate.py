"""Evaluation: metrics, bootstrap confidence intervals over prints, grouped CV.

Recall (the share of under-extruded frames that are caught) is the primary
metric, because a missed defect wastes the print; precision, F1, specificity
and ROC-AUC are reported alongside.

Frames of one print are near-duplicates, so the independent unit is the
print. Confidence intervals therefore resample *prints* (with all their
frames), and cross-validation folds are made of whole prints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from . import config
from .baseline import fit_baseline, predict_proba

#: Metrics returned by :func:`binary_metrics`, in order.
METRIC_NAMES = ("recall", "precision", "f1", "specificity", "accuracy", "roc_auc")


def _check_labels_proba(y_true, proba):
    """Validate 0/1 labels and matching probabilities."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    if y_true.ndim != 1 or y_true.shape != proba.shape or len(y_true) == 0:
        raise ValueError(
            f"y_true and proba must be non-empty 1-D arrays of equal length, "
            f"got {y_true.shape} and {proba.shape}"
        )
    if not set(np.unique(y_true).tolist()) <= {0, 1}:
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(proba).all() or proba.min() < 0 or proba.max() > 1:
        raise ValueError("proba must be finite and in [0, 1]")
    return y_true.astype(int), proba


def binary_metrics(y_true, proba, threshold=0.5):
    """Classification metrics at a probability threshold.

    Parameters
    ----------
    y_true : array_like
        0/1 labels (1 = under-extrusion).
    proba : array_like
        Predicted probability of under-extrusion, in [0, 1].
    threshold : float, default 0.5
        A frame is predicted defective when ``proba >= threshold``.

    Returns
    -------
    dict
        ``recall = TP / (TP + FN)``, ``precision = TP / (TP + FP)``,
        ``f1`` (harmonic mean of precision and recall),
        ``specificity = TN / (TN + FP)``, ``accuracy`` and ``roc_auc``.
        A ratio with a zero denominator is NaN (precision and F1 are 0 when
        nothing is predicted defective but defects exist). ``roc_auc`` is NaN
        when only one class is present.

    Raises
    ------
    ValueError
        If the inputs are malformed or ``threshold`` is outside [0, 1].
    """
    y, p = _check_labels_proba(y_true, proba)
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def ratio(a, b):
        return a / b if b else float("nan")

    recall = ratio(tp, tp + fn)
    precision = ratio(tp, tp + fp)
    if tp + fp == 0 and tp + fn > 0:
        precision = 0.0  # defects exist but none were flagged
    if np.isnan(precision) or np.isnan(recall):
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both_classes = 0 < y.sum() < len(y)
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "specificity": ratio(tn, tn + fp),
        "accuracy": (tp + tn) / len(y),
        "roc_auc": roc_auc_score(y, p) if both_classes else float("nan"),
    }


def bootstrap_ci(y_true, proba, groups, n_boot=1000, ci=0.95, threshold=0.5, seed=config.DEFAULT_SEED):
    """Percentile bootstrap confidence intervals, resampling whole groups (prints).

    Each resample draws as many prints as there are, with replacement, and
    keeps all frames of each drawn print. Resamples containing only one class
    are skipped, because recall, specificity and ROC-AUC are undefined there.

    Parameters
    ----------
    y_true, proba : array_like
        Labels and probabilities, as for :func:`binary_metrics`.
    groups : array_like
        Group (print) id of each frame.
    n_boot : int, default 1000
        Number of resamples attempted.
    ci : float, default 0.95
        Confidence level.
    threshold : float, default 0.5
    seed : int, default ``config.DEFAULT_SEED``

    Returns
    -------
    pandas.DataFrame
        One row per metric: ``metric``, ``estimate`` (on the original data),
        ``lo``, ``hi`` and ``n_valid`` (resamples used).

    Raises
    ------
    ValueError
        If the inputs are malformed, ``groups`` has a different length, there
        are fewer than two groups, or ``ci`` is not in (0, 1).
    """
    y, p = _check_labels_proba(y_true, proba)
    groups = np.asarray(groups)
    if groups.shape != y.shape:
        raise ValueError("groups must have the same length as y_true")
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("need at least two groups to bootstrap")

    index_of = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = np.concatenate([index_of[g] for g in rng.choice(unique, size=len(unique))])
        if 0 < y[idx].sum() < len(idx):
            samples.append(binary_metrics(y[idx], p[idx], threshold))
    samples = pd.DataFrame(samples, columns=list(METRIC_NAMES))
    estimate = binary_metrics(y, p, threshold)
    alpha = (1 - ci) / 2
    return pd.DataFrame({
        "metric": list(METRIC_NAMES),
        "estimate": [estimate[m] for m in METRIC_NAMES],
        "lo": [samples[m].quantile(alpha) for m in METRIC_NAMES],
        "hi": [samples[m].quantile(1 - alpha) for m in METRIC_NAMES],
        "n_valid": len(samples),
    })


def group_level(y_true, proba, groups):
    """Aggregate frame predictions to one prediction per group (print).

    Parameters
    ----------
    y_true, proba, groups : array_like
        Frame labels, probabilities and group ids.

    Returns
    -------
    pandas.DataFrame
        Indexed by group, with ``label`` (the group's label, which must be the
        same for all its frames) and ``proba`` (mean frame probability).

    Raises
    ------
    ValueError
        If a group has frames with different labels.
    """
    y, p = _check_labels_proba(y_true, proba)
    df = pd.DataFrame({"group": np.asarray(groups), "label": y, "proba": p})
    if (df.groupby("group")["label"].nunique() > 1).any():
        raise ValueError("a group has frames with different labels")
    return df.groupby("group").agg(label=("label", "first"), proba=("proba", "mean"))


def _fold_rows(x, y, groups, splits, model_name, seed, threshold):
    """Fit and score one baseline model per (train, test) index pair."""
    rows = []
    for fold, (tr, te) in enumerate(splits):
        if set(groups[tr]) & set(groups[te]):
            raise RuntimeError("a group appears on both sides of a fold")
        model = fit_baseline(model_name, x[tr], y[tr], seed=seed)
        metrics = binary_metrics(y[te], predict_proba(model, x[te]), threshold)
        rows.append({"fold": fold, "n_train": len(tr), "n_test": len(te), **metrics})
    return rows


def grouped_cv(x, y, groups, model_name, n_splits=5, seed=config.DEFAULT_SEED, threshold=0.5):
    """Stratified grouped k-fold cross-validation of a baseline model.

    Folds are built from whole groups (prints) with ``StratifiedGroupKFold``,
    which balances the label across folds as far as whole groups allow.

    Parameters
    ----------
    x : numpy.ndarray
        Feature matrix, one row per frame.
    y : array_like
        0/1 labels.
    groups : array_like
        Print id of each frame.
    model_name : {"logreg", "random_forest"}
    n_splits : int, default 5
    seed : int, default ``config.DEFAULT_SEED``
        Seed for fold assignment and the model.
    threshold : float, default 0.5

    Returns
    -------
    pandas.DataFrame
        One row per fold with ``fold``, ``n_train``, ``n_test``,
        ``n_test_groups`` and the metrics of :func:`binary_metrics`.
    """
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(x, y, groups))
    rows = _fold_rows(x, y, groups, splits, model_name, seed, threshold)
    for row, (_, te) in zip(rows, splits):
        row["n_test_groups"] = len(np.unique(groups[te]))
    return pd.DataFrame(rows)


def leave_one_group_out(x, y, groups, model_name, seed=config.DEFAULT_SEED, threshold=0.5):
    """Leave-one-group-out evaluation, e.g. hold out each printer in turn.

    Parameters
    ----------
    x, y : array_like
        Features and 0/1 labels, one row per frame.
    groups : array_like
        Group of each frame (e.g. ``printer_id``); one fold per unique value.
    model_name : {"logreg", "random_forest"}
    seed : int, default ``config.DEFAULT_SEED``
    threshold : float, default 0.5

    Returns
    -------
    pandas.DataFrame
        One row per held-out group with ``held_out``, ``n_train``,
        ``n_test`` and the metrics of :func:`binary_metrics`.
    """
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    splits = list(LeaveOneGroupOut().split(x, y, groups))
    rows = _fold_rows(x, y, groups, splits, model_name, seed, threshold)
    for row, (_, te) in zip(rows, splits):
        row["held_out"] = groups[te][0]
    df = pd.DataFrame(rows)
    return df[["held_out"] + [c for c in df.columns if c != "held_out"]]
