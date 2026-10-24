"""Baseline classifiers on the hand-crafted features.

Two models, both with ``class_weight="balanced"`` so each class contributes
equally to the loss however many frames it has:

- ``"logreg"``: standardisation followed by L2-regularised logistic
  regression. Standardising matters because the features have very different
  scales (GLCM contrast can reach 49, histogram bins are below 1) and the L2
  penalty would otherwise shrink some coefficients far more than others.
- ``"random_forest"``: 300 trees with at least 5 frames per leaf, which stops
  trees memorising individual near-duplicate frames.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import extract_features_from_paths

#: Names accepted by :func:`make_model`.
MODEL_NAMES = ("logreg", "random_forest")


def make_model(name, seed=config.DEFAULT_SEED):
    """Build an unfitted baseline model.

    Parameters
    ----------
    name : {"logreg", "random_forest"}
        Which model to build.
    seed : int, default ``config.DEFAULT_SEED``
        Random state, so fits are reproducible.

    Returns
    -------
    sklearn estimator
        Unfitted classifier with ``fit`` and ``predict_proba``.

    Raises
    ------
    ValueError
        If ``name`` is not in :data:`MODEL_NAMES`.
    """
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def _check_xy(x, y=None):
    """Validate a feature matrix and, optionally, binary labels."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"features must be a non-empty 2-D array, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("features contain NaN or infinite values")
    if y is None:
        return x
    y = np.asarray(y)
    if y.shape != (x.shape[0],):
        raise ValueError(f"labels must have shape ({x.shape[0]},), got {y.shape}")
    values = set(np.unique(y).tolist())
    if not values <= {0, 1}:
        raise ValueError(f"labels must be 0 or 1, found {sorted(values)}")
    if len(values) < 2:
        raise ValueError("training labels contain only one class")
    return x, y.astype(int)


def fit_baseline(name, x, y, seed=config.DEFAULT_SEED):
    """Fit a baseline model.

    Parameters
    ----------
    name : {"logreg", "random_forest"}
        Which model to fit.
    x : array_like
        Feature matrix of shape ``(n_samples, n_features)``.
    y : array_like
        Labels of shape ``(n_samples,)``, 0 or 1, with both classes present.
    seed : int, default ``config.DEFAULT_SEED``
        Random state.

    Returns
    -------
    sklearn estimator
        The fitted model.

    Raises
    ------
    ValueError
        If ``name`` is unknown, or ``x`` / ``y`` are malformed.
    """
    model = make_model(name, seed)
    x, y = _check_xy(x, y)
    return model.fit(x, y)


def predict_proba(model, x):
    """Probability of under-extrusion for each sample.

    Parameters
    ----------
    model : sklearn estimator
        A model returned by :func:`fit_baseline`.
    x : array_like
        Feature matrix with the same columns as in training.

    Returns
    -------
    numpy.ndarray
        Probabilities in [0, 1], shape ``(n_samples,)``.
    """
    return model.predict_proba(_check_xy(x))[:, 1]


def compute_features(df, size=config.DEFAULT_IMAGE_SIZE, **kwargs):
    """Feature matrix for every image in a metadata table.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata with a ``path`` column, as returned by
        :func:`fdm_defect.data.load_metadata` with ``image_dir`` set.
    size : int, default ``config.DEFAULT_IMAGE_SIZE``
        Image side length used for feature extraction.
    **kwargs
        Passed to :func:`fdm_defect.features.extract_features`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(df), n_features)``, rows in ``df`` order.
    """
    if "path" not in df.columns:
        raise ValueError("df has no 'path' column; load it with image_dir set")
    return extract_features_from_paths(df["path"], size=size, **kwargs)
