"""Tests for fdm_defect.features, including known-answer tests."""

import numpy as np
import pytest

from fdm_defect.features import (
    LBP_BINS,
    extract_features,
    extract_features_from_paths,
    feature_names,
    fft_features,
    glcm,
    glcm_features,
    gradient_features,
    grey_histogram,
    lbp_histogram,
    lbp_uniform,
    sobel,
    to_grey,
)
from fdm_defect.synthetic import make_bead_image


def checkerboard(n=16):
    """n x n checkerboard of 0.0 and 1.0."""
    return (np.indices((n, n)).sum(axis=0) % 2).astype(float)


# --------------------------------------------------------------------------
# Sobel: known answer on a linear ramp
# --------------------------------------------------------------------------


@pytest.mark.parametrize("slope", [0.01, 0.05, -0.03])
def test_sobel_horizontal_ramp_is_8_times_slope(slope):
    grey = slope * np.tile(np.arange(12.0), (10, 1))  # g = slope * x
    gx, gy = sobel(grey)
    assert gx.shape == (8, 10)
    np.testing.assert_allclose(gx, 8 * slope)
    np.testing.assert_allclose(gy, 0.0, atol=1e-12)


def test_sobel_vertical_ramp_is_8_times_slope():
    grey = 0.02 * np.arange(10.0)[:, None] * np.ones((1, 12))  # g = 0.02 * y
    gx, gy = sobel(grey)
    np.testing.assert_allclose(gy, 8 * 0.02)
    np.testing.assert_allclose(gx, 0.0, atol=1e-12)


def test_gradient_features_constant_image_are_zero():
    np.testing.assert_array_equal(gradient_features(np.full((8, 8), 0.3)), np.zeros(5))


def test_gradient_features_step_edge_density():
    # Vertical step from 0 to 1 at column 4 of an 8x8 image: the Sobel
    # magnitude is 4 in the two interior columns either side of the step and 0
    # elsewhere, so 2 of the 6 interior columns are edges.
    grey = np.zeros((8, 8))
    grey[:, 4:] = 1.0
    feats = gradient_features(grey, edge_threshold=1.0)
    assert feats[4] == pytest.approx(2 / 6)
    assert feats[2] == pytest.approx(4 * 2 / 6)  # mean |gx|
    assert feats[3] == 0.0  # mean |gy|


def test_gradient_features_negative_threshold():
    with pytest.raises(ValueError):
        gradient_features(np.zeros((5, 5)), edge_threshold=-1)


# --------------------------------------------------------------------------
# FFT: a pure cosine puts half the power in each of its two peaks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 5, 13])
def test_fft_pure_cosine_peak_fraction_is_half(k):
    n = 64
    x = np.arange(n)
    grey = np.tile(0.5 + 0.5 * np.cos(2 * np.pi * k * x / n), (n, 1))
    peak_fraction, peak_freq = fft_features(grey)
    assert peak_fraction == pytest.approx(0.5)
    assert peak_freq == pytest.approx(k / n)


def test_fft_constant_image():
    np.testing.assert_array_equal(fft_features(np.full((16, 16), 0.7)), [0.0, 0.0])


# --------------------------------------------------------------------------
# GLCM: two-level checkerboard has contrast (L - 1)^2
# --------------------------------------------------------------------------


@pytest.mark.parametrize("levels", [2, 8, 16])
def test_glcm_checkerboard_known_values(levels):
    contrast, homogeneity, energy, correlation = glcm_features(glcm(checkerboard(), levels))
    assert contrast == pytest.approx((levels - 1) ** 2)
    assert homogeneity == pytest.approx(1 / (1 + (levels - 1) ** 2))
    # Only cells (0, L-1) and (L-1, 0) are filled, each with 0.5.
    assert energy == pytest.approx(np.sqrt(0.5))
    assert correlation == pytest.approx(-1.0)


def test_glcm_is_symmetric_and_normalised():
    rng = np.random.default_rng(0)
    p = glcm(rng.random((20, 20)), levels=8)
    assert p.shape == (8, 8)
    np.testing.assert_allclose(p, p.T)
    assert p.sum() == pytest.approx(1.0)


def test_glcm_constant_image():
    contrast, homogeneity, energy, correlation = glcm_features(glcm(np.full((10, 10), 0.4)))
    assert (contrast, homogeneity, energy, correlation) == pytest.approx((0, 1, 1, 1))


def test_glcm_maps_one_to_top_level():
    p = glcm(np.ones((5, 5)), levels=4)
    assert p[3, 3] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs", [{"levels": 1}, {"offsets": ()}, {"offsets": ((0, 0),)}, {"offsets": ((-1, 0),)}, {"offsets": ((20, 0),)}]
)
def test_glcm_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        glcm(checkerboard(), **kwargs)


def test_glcm_features_non_square():
    with pytest.raises(ValueError):
        glcm_features(np.ones((2, 3)))


# --------------------------------------------------------------------------
# LBP
# --------------------------------------------------------------------------


def test_lbp_constant_image_all_neighbours_equal():
    # Every neighbour >= centre -> eight 1 bits -> uniform label 8.
    labels = lbp_uniform(np.full((6, 6), 0.5))
    assert (labels == 8).all()
    hist = lbp_histogram(np.full((6, 6), 0.5))
    assert hist[8] == 1.0 and hist.sum() == pytest.approx(1.0)


def test_lbp_bright_spot_is_label_zero():
    grey = np.zeros((3, 3))
    grey[1, 1] = 1.0  # centre brighter than all neighbours -> no 1 bits
    assert lbp_uniform(grey)[0, 0] == 0


def test_lbp_edge_is_uniform():
    # Top row brighter than the centre, rest darker: bits 1,1,1,0,0,0,0,0
    # around the circle -> 2 transitions, 3 ones -> label 3.
    grey = np.array([[1.0, 1.0, 1.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
    assert lbp_uniform(grey)[0, 0] == 3


def test_lbp_alternating_is_non_uniform():
    # Neighbours alternate 1, 0, ... around the circle -> 8 transitions.
    grey = np.array([[1.0, 0.0, 1.0], [0.0, 0.5, 0.0], [1.0, 0.0, 1.0]])
    assert lbp_uniform(grey)[0, 0] == LBP_BINS - 1


# --------------------------------------------------------------------------
# Histogram and grey conversion
# --------------------------------------------------------------------------


def test_grey_histogram_constant():
    hist = grey_histogram(np.full((4, 4), 0.5), bins=16)
    assert hist[8] == 1.0  # 0.5 * 16 = 8 -> bin 8
    assert hist.sum() == pytest.approx(1.0)


def test_grey_histogram_includes_endpoints():
    hist = grey_histogram(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), bins=4)
    np.testing.assert_allclose(hist, [5 / 9, 0, 0, 4 / 9])


def test_to_grey_weights():
    img = np.zeros((3, 3, 3))
    img[..., 1] = 1.0  # pure green
    np.testing.assert_allclose(to_grey(img), 0.587)


# --------------------------------------------------------------------------
# extract_features
# --------------------------------------------------------------------------


def test_extract_features_length_matches_names():
    img = make_bead_image(0, size=32, rng=0) / 255.0
    feats = extract_features(img)
    names = feature_names()
    assert feats.shape == (len(names),) == (43,)
    assert len(set(names)) == len(names)
    assert np.isfinite(feats).all()


def test_extract_features_custom_bins():
    img = make_bead_image(1, size=32, rng=0) / 255.0
    assert len(extract_features(img, hist_bins=8)) == len(feature_names(hist_bins=8))


def test_extract_features_deterministic():
    img = make_bead_image(1, size=32, rng=3) / 255.0
    np.testing.assert_array_equal(extract_features(img), extract_features(img))


@pytest.mark.parametrize(
    "img, err",
    [
        (np.zeros((8, 8, 3), dtype=np.uint8), TypeError),
        (np.zeros((8, 8)), ValueError),
        (np.zeros((8, 8, 4)), ValueError),
        (np.zeros((2, 8, 3)), ValueError),
        (np.full((8, 8, 3), 1.5), ValueError),
        (np.full((8, 8, 3), np.nan), ValueError),
    ],
)
def test_extract_features_invalid_images(img, err):
    with pytest.raises(err):
        extract_features(img)


def test_extract_features_from_paths(synthetic_dir):
    paths = sorted((synthetic_dir / "images").rglob("*.jpg"))[:3]
    x = extract_features_from_paths(paths, size=32)
    assert x.shape == (3, 43)


def test_extract_features_from_paths_empty():
    with pytest.raises(ValueError):
        extract_features_from_paths([])
