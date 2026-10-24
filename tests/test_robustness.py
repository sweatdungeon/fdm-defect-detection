"""Tests for fdm_defect.robustness: image corruptions."""

import numpy as np
import pytest

from fdm_defect.robustness import (
    CORRUPTIONS,
    adjust_brightness,
    add_gaussian_noise,
    corrupt,
    gaussian_blur,
    gaussian_kernel,
    jpeg_compress,
)


@pytest.fixture
def img():
    return np.random.default_rng(0).random((16, 20, 3)).astype(np.float32)


@pytest.mark.parametrize("kind", list(CORRUPTIONS))
def test_every_corruption_keeps_shape_dtype_and_range(img, kind):
    for level in CORRUPTIONS[kind]:
        out = corrupt(img, kind, level, rng=0)
        assert out.shape == img.shape and out.dtype == img.dtype
        assert out.min() >= 0 and out.max() <= 1


def test_gaussian_kernel_sums_to_one_and_is_symmetric():
    k = gaussian_kernel(1.5)
    assert len(k) == 2 * 5 + 1  # radius ceil(4.5) = 5
    assert k.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(k, k[::-1])


def test_blur_leaves_constant_image_unchanged():
    flat = np.full((10, 10, 3), 0.3, dtype=np.float32)
    np.testing.assert_allclose(gaussian_blur(flat, 2.0), flat, atol=1e-6)


def test_blur_matches_dense_2d_convolution(img):
    # Known answer: the separable blur equals the direct 2-D convolution with
    # the outer-product kernel at an interior pixel.
    sigma = 1.0
    k = gaussian_kernel(sigma)
    r = len(k) // 2
    k2 = np.outer(k, k)
    y, x = 8, 10
    expected = (img[y - r : y + r + 1, x - r : x + r + 1, :] * k2[..., None]).sum(axis=(0, 1))
    np.testing.assert_allclose(gaussian_blur(img, sigma)[y, x], expected, rtol=1e-5)


def test_blur_reduces_variance(img):
    assert gaussian_blur(img, 2.0).var() < 0.2 * img.var()


def test_brightness_known_values():
    x = np.array([[[0.2, 0.5, 0.8]]], dtype=np.float32)
    np.testing.assert_allclose(adjust_brightness(x, 1.5), [[[0.3, 0.75, 1.0]]])  # 1.2 clips to 1


def test_noise_has_requested_std():
    flat = np.full((200, 200, 3), 0.5, dtype=np.float32)
    out = add_gaussian_noise(flat, 0.05, rng=0)
    assert (out - flat).std() == pytest.approx(0.05, rel=0.02)


def test_noise_is_reproducible(img):
    np.testing.assert_array_equal(add_gaussian_noise(img, 0.1, rng=3), add_gaussian_noise(img, 0.1, rng=3))


def test_jpeg_lower_quality_means_larger_error(img):
    err_hi = np.abs(jpeg_compress(img, 90) - img).mean()
    err_lo = np.abs(jpeg_compress(img, 10) - img).mean()
    assert err_lo > err_hi


@pytest.mark.parametrize(
    "fn, arg", [(gaussian_blur, 0), (adjust_brightness, -1), (add_gaussian_noise, -0.1), (jpeg_compress, 0)]
)
def test_invalid_levels(img, fn, arg):
    with pytest.raises(ValueError):
        fn(img, arg)


def test_unknown_corruption(img):
    with pytest.raises(ValueError, match="unknown corruption"):
        corrupt(img, "rotate", 10)


def test_rejects_uint8():
    with pytest.raises(TypeError):
        adjust_brightness(np.zeros((4, 4, 3), dtype=np.uint8), 1.0)
