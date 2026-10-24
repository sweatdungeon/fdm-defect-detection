"""Shared fixtures: a small synthetic dataset written to a temporary folder."""

import pytest

from fdm_defect.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    """Folder with ``metadata.csv`` and ``images/`` for 10 good + 10 defective prints."""
    out = tmp_path_factory.mktemp("synthetic")
    make_synthetic_dataset(out, n_good_prints=10, n_defective_prints=10, frames_per_print=3)
    return out


@pytest.fixture
def synthetic_df(synthetic_dir):
    """Raw (unvalidated) synthetic metadata, a fresh copy per test."""
    import pandas as pd

    return pd.read_csv(synthetic_dir / "metadata.csv")
