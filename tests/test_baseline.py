"""Tests for fdm_defect.baseline: model construction, fitting and prediction."""

import numpy as np
import pytest

from fdm_defect.baseline import MODEL_NAMES, compute_features, fit_baseline, make_model, predict_proba
from fdm_defect.data import load_metadata
from fdm_defect.features import extract_features
from fdm_defect.synthetic import make_bead_image


def synthetic_features(n_per_class, seed):
    """Features and labels for n_per_class good and defective synthetic images."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], n_per_class)
    x = np.stack([extract_features(make_bead_image(lab, size=48, rng=rng) / 255.0) for lab in labels])
    return x, labels


@pytest.fixture(scope="module")
def train_test():
    return synthetic_features(30, seed=0), synthetic_features(30, seed=1)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_separate_synthetic_classes(train_test, name):
    (x_train, y_train), (x_test, y_test) = train_test
    model = fit_baseline(name, x_train, y_train, seed=0)
    proba = predict_proba(model, x_test)
    assert proba.shape == (len(y_test),)
    assert ((proba >= 0) & (proba <= 1)).all()
    accuracy = ((proba >= 0.5) == y_test).mean()
    assert accuracy >= 0.9


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fit_is_reproducible(train_test, name):
    (x_train, y_train), (x_test, _) = train_test
    a = predict_proba(fit_baseline(name, x_train, y_train, seed=5), x_test)
    b = predict_proba(fit_baseline(name, x_train, y_train, seed=5), x_test)
    # The trees are identical, but with n_jobs=-1 the per-tree probabilities
    # are summed in thread-completion order, which can change the last bit.
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)


def test_models_use_balanced_class_weights():
    assert make_model("random_forest").class_weight == "balanced"
    assert make_model("logreg")[-1].class_weight == "balanced"


def test_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        make_model("svm")


@pytest.mark.parametrize(
    "x, y, match",
    [
        (np.zeros((4, 3)), np.array([0, 1, 0]), "shape"),
        (np.zeros((4, 3)), np.array([0, 1, 2, 1]), "0 or 1"),
        (np.zeros((4, 3)), np.array([1, 1, 1, 1]), "one class"),
        (np.full((4, 3), np.nan), np.array([0, 1, 0, 1]), "NaN"),
        (np.zeros(4), np.array([0, 1, 0, 1]), "2-D"),
    ],
)
def test_fit_invalid_inputs(x, y, match):
    with pytest.raises(ValueError, match=match):
        fit_baseline("logreg", x, y)


def test_compute_features_from_metadata(synthetic_dir):
    df = load_metadata(synthetic_dir / "metadata.csv", synthetic_dir / "images").head(4)
    assert compute_features(df, size=32).shape == (4, 43)


def test_compute_features_needs_path_column(synthetic_df):
    with pytest.raises(ValueError, match="path"):
        compute_features(synthetic_df)
