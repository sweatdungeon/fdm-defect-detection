"""Tests for fdm_defect.synthetic: the bead-stripe image generator."""

import numpy as np
import pytest

from fdm_defect.data import load_metadata, validate_metadata
from fdm_defect.synthetic import make_bead_image, make_synthetic_dataset


@pytest.mark.parametrize("label", [0, 1])
def test_bead_image_shape_and_dtype(label):
    img = make_bead_image(label, size=48, rng=0)
    assert img.shape == (48, 48, 3)
    assert img.dtype == np.uint8


def test_bead_image_deterministic():
    np.testing.assert_array_equal(make_bead_image(1, rng=7), make_bead_image(1, rng=7))


def test_good_image_is_periodic_at_n_beads():
    # Known answer: with size a multiple of n_beads and no noise, a good image's
    # row-mean profile repeats exactly n_beads times, so the strongest non-DC
    # frequency in its FFT is n_beads.
    img = make_bead_image(0, size=128, n_beads=8, noise_std=0.0, rng=0)
    profile = img.mean(axis=(1, 2))
    power = np.abs(np.fft.rfft(profile - profile.mean())) ** 2
    assert np.argmax(power) == 8


def test_good_image_has_no_gaps_along_beads():
    # Columns of a good image are identical apart from noise: beads are continuous.
    img = make_bead_image(0, size=64, noise_std=0.0, rng=0).astype(int)
    assert (img == img[:, :1, :]).all()


def test_defect_images_cover_less_area():
    # Under-extruded beads are thinner and gapped, so on average the image is
    # closer to the background. Compare the bright-pixel fraction over 20 seeds.
    def bright_fraction(label, seed):
        img = make_bead_image(label, size=64, noise_std=0.0, rng=seed).mean(axis=2)
        return (img > (img.min() + img.max()) / 2).mean()

    good = np.mean([bright_fraction(0, s) for s in range(20)])
    bad = np.mean([bright_fraction(1, s) for s in range(20)])
    assert bad < 0.6 * good


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label": 2},
        {"label": 0, "size": 0},
        {"label": 0, "n_beads": 0},
        {"label": 0, "size": 4, "n_beads": 8},
        {"label": 0, "noise_std": -1.0},
    ],
)
def test_bead_image_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        make_bead_image(**kwargs)


def test_synthetic_dataset_layout(tmp_path):
    df = make_synthetic_dataset(tmp_path, n_good_prints=2, n_defective_prints=3, frames_per_print=2)
    assert len(df) == 10
    assert df["print_id"].nunique() == 5
    assert df["has_under_extrusion"].sum() == 6
    # What was written passes the same validation as the real data.
    loaded = load_metadata(tmp_path / "metadata.csv", tmp_path / "images")
    assert len(loaded) == 10
    validate_metadata(df, check_files=False)


def test_synthetic_dataset_deterministic(tmp_path):
    make_synthetic_dataset(tmp_path / "a", seed=3, frames_per_print=1)
    make_synthetic_dataset(tmp_path / "b", seed=3, frames_per_print=1)
    for f in (tmp_path / "a" / "images").rglob("*.jpg"):
        twin = tmp_path / "b" / "images" / f.relative_to(tmp_path / "a" / "images")
        assert f.read_bytes() == twin.read_bytes()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_good_prints": -1},
        {"n_good_prints": 0, "n_defective_prints": 0},
        {"frames_per_print": 0},
        {"printers": ()},
    ],
)
def test_synthetic_dataset_invalid_inputs(tmp_path, kwargs):
    with pytest.raises(ValueError):
        make_synthetic_dataset(tmp_path, **kwargs)
