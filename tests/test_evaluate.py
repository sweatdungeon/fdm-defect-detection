"""Tests for fdm_defect.evaluate: metrics against hand calculations, bootstrap, CV."""

import numpy as np
import pandas as pd
import pytest

from fdm_defect.evaluate import (
    METRIC_NAMES,
    binary_metrics,
    bootstrap_ci,
    group_level,
    grouped_cv,
    leave_one_group_out,
)

# --------------------------------------------------------------------------
# binary_metrics vs hand counts
# --------------------------------------------------------------------------


def test_metrics_hand_calculation():
    # Predictions at 0.5: TP = 2, FN = 1, FP = 1, TN = 2.
    y = [1, 1, 1, 0, 0, 0]
    p = [0.9, 0.6, 0.2, 0.7, 0.1, 0.3]
    m = binary_metrics(y, p)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    # ROC-AUC = share of (positive, negative) pairs ranked correctly:
    # 0.9 beats all 3, 0.6 beats 0.1 and 0.3, 0.2 beats 0.1 -> 6 of 9.
    assert m["roc_auc"] == pytest.approx(6 / 9)


def test_metrics_threshold_is_inclusive():
    assert binary_metrics([1, 0], [0.5, 0.4])["recall"] == 1.0


def test_metrics_perfect():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert all(m[k] == 1.0 for k in METRIC_NAMES)


def test_metrics_nothing_flagged():
    m = binary_metrics([1, 0], [0.1, 0.2])
    assert m["recall"] == 0.0 and m["precision"] == 0.0 and m["f1"] == 0.0


def test_metrics_single_class_auc_is_nan():
    m = binary_metrics([1, 1], [0.7, 0.2])
    assert np.isnan(m["roc_auc"]) and np.isnan(m["specificity"])
    assert m["recall"] == 0.5


@pytest.mark.parametrize(
    "y, p, kwargs",
    [([0, 1], [0.5], {}), ([0, 2], [0.1, 0.9], {}), ([0, 1], [0.1, 1.5], {}), ([], [], {}), ([0, 1], [0.1, 0.9], {"threshold": 2})],
)
def test_metrics_invalid(y, p, kwargs):
    with pytest.raises(ValueError):
        binary_metrics(y, p, **kwargs)


# --------------------------------------------------------------------------
# bootstrap_ci and group_level
# --------------------------------------------------------------------------


def make_groups(n_groups=10, frames=5, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_groups) % 2, frames)
    groups = np.repeat(np.arange(n_groups), frames)
    proba = np.clip(labels * 0.4 + 0.3 + rng.normal(0, 0.15, len(labels)), 0, 1)
    return labels, proba, groups


def test_bootstrap_interval_contains_estimate():
    y, p, g = make_groups()
    ci = bootstrap_ci(y, p, g, n_boot=300, seed=0).set_index("metric")
    assert list(ci.index) == list(METRIC_NAMES)
    assert ((ci["lo"] <= ci["estimate"] + 1e-12) & (ci["estimate"] <= ci["hi"] + 1e-12)).all()
    assert 0 < ci["n_valid"].iloc[0] <= 300


def test_bootstrap_deterministic():
    y, p, g = make_groups()
    pd.testing.assert_frame_equal(bootstrap_ci(y, p, g, n_boot=100, seed=1), bootstrap_ci(y, p, g, n_boot=100, seed=1))


def test_bootstrap_skips_single_class_resamples():
    # With 2 groups, a resample is one class whenever both draws are the same
    # group: probability 1/2. Roughly half the resamples must be skipped.
    y, p, g = make_groups(n_groups=2)
    n_valid = bootstrap_ci(y, p, g, n_boot=400, seed=0)["n_valid"].iloc[0]
    assert 140 < n_valid < 260


def test_bootstrap_perfect_separation_has_zero_width():
    y = np.repeat([0, 1, 0, 1], 3)
    p = np.where(y == 1, 0.9, 0.1)
    ci = bootstrap_ci(y, p, np.repeat(np.arange(4), 3), n_boot=100).set_index("metric")
    assert (ci.loc["recall", ["lo", "hi"]] == 1.0).all()


def test_bootstrap_invalid():
    y, p, g = make_groups()
    with pytest.raises(ValueError):
        bootstrap_ci(y, p, g[:-1])
    with pytest.raises(ValueError):
        bootstrap_ci(y, p, np.zeros_like(g))
    with pytest.raises(ValueError):
        bootstrap_ci(y, p, g, ci=1.5)


def test_group_level_means():
    out = group_level([1, 1, 0, 0], [0.8, 0.6, 0.2, 0.4], ["a", "a", "b", "b"])
    assert out.loc["a", "proba"] == pytest.approx(0.7)
    assert out.loc["b", "label"] == 0


def test_group_level_mixed_labels():
    with pytest.raises(ValueError, match="different labels"):
        group_level([1, 0], [0.5, 0.5], ["a", "a"])


# --------------------------------------------------------------------------
# grouped_cv and leave_one_group_out
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def separable():
    """Two-feature data where the label is easy to predict, 20 groups x 5 frames."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 5)
    y = np.repeat(np.arange(20) % 2, 5)
    x = rng.normal(size=(100, 2)) + 3 * y[:, None]
    # Consecutive blocks of 5 groups per printer, so each printer has both labels.
    printers = groups // 5
    return x, y, groups, printers


def test_grouped_cv_folds(separable):
    x, y, groups, _ = separable
    folds = grouped_cv(x, y, groups, "logreg", n_splits=5, seed=0)
    assert len(folds) == 5
    assert folds["n_test"].sum() == 100  # every frame tested exactly once
    assert (folds["n_test_groups"] == 4).all()
    assert (folds["roc_auc"] > 0.9).all()


def test_leave_one_group_out(separable):
    x, y, _, printers = separable
    folds = leave_one_group_out(x, y, printers, "random_forest", seed=0)
    assert sorted(folds["held_out"]) == [0, 1, 2, 3]
    assert (folds["n_test"] == 25).all()
    assert (folds["recall"] > 0.8).all()
