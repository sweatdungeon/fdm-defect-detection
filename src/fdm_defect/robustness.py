"""Image corruptions for robustness testing without retraining.

Each corruption takes a float RGB image in [0, 1] (as returned by
:func:`fdm_defect.data.load_image`) and returns one of the same shape and
range. They are applied at the model's input resolution and mimic realistic
camera problems: defocus (blur), lighting changes (brightness), sensor noise
and heavy compression (JPEG).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

#: Corruptions and the levels tested, mild then severe.
#: blur: Gaussian sigma in px; brightness: multiplicative factor;
#: noise: Gaussian std on a [0, 1] scale; jpeg: quality (lower = worse).
CORRUPTIONS = {
    "blur": (1.0, 2.0),
    "brightness": (0.6, 1.4),
    "noise": (0.05, 0.1),
    "jpeg": (30, 10),
}


def _check_image(img):
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {img.shape}")
    if not np.issubdtype(img.dtype, np.floating):
        raise TypeError(f"expected a float image in [0, 1], got {img.dtype}")
    if not np.isfinite(img).all() or img.min() < 0 or img.max() > 1:
        raise ValueError("image values must be finite and in [0, 1]")
    return img


def gaussian_kernel(sigma):
    """Normalised 1-D Gaussian kernel with radius ``ceil(3 sigma)``.

    Parameters
    ----------
    sigma : float
        Standard deviation in pixels, > 0.

    Returns
    -------
    numpy.ndarray
        Odd-length kernel summing to 1.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    radius = int(np.ceil(3 * sigma))
    t = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(img, sigma):
    """Separable Gaussian blur with reflected borders.

    The 2-D Gaussian factorises into a horizontal and a vertical 1-D pass,
    which costs ``2 (2r + 1)`` instead of ``(2r + 1)^2`` multiplies per pixel.

    Parameters
    ----------
    img : numpy.ndarray
        Float ``(H, W, 3)`` image in [0, 1].
    sigma : float
        Standard deviation in pixels, > 0.

    Returns
    -------
    numpy.ndarray
        Blurred image, same shape and dtype.
    """
    img = _check_image(img)
    k = gaussian_kernel(sigma)
    r = len(k) // 2
    out = img.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * 3
        pad[axis] = (r, r)
        padded = np.pad(out, pad, mode="reflect")
        n = out.shape[axis]
        # Weighted sum of shifted copies = 1-D convolution along this axis.
        out = sum(
            w * np.take(padded, np.arange(i, i + n), axis=axis) for i, w in enumerate(k)
        )
    return np.clip(out, 0.0, 1.0).astype(img.dtype)


def adjust_brightness(img, factor):
    """Multiply all values by ``factor`` and clip to [0, 1]."""
    img = _check_image(img)
    if factor < 0:
        raise ValueError(f"factor must be non-negative, got {factor}")
    return np.clip(img * factor, 0.0, 1.0).astype(img.dtype)


def add_gaussian_noise(img, std, rng=None):
    """Add i.i.d. Gaussian noise with standard deviation ``std`` and clip to [0, 1]."""
    img = _check_image(img)
    if std < 0:
        raise ValueError(f"std must be non-negative, got {std}")
    rng = np.random.default_rng(rng)
    return np.clip(img + rng.normal(0.0, std, size=img.shape), 0.0, 1.0).astype(img.dtype)


def jpeg_compress(img, quality):
    """Round-trip the image through JPEG encoding at ``quality`` (1-95)."""
    img = _check_image(img)
    if not 1 <= quality <= 95:
        raise ValueError(f"quality must be in 1-95, got {quality}")
    buf = io.BytesIO()
    Image.fromarray(np.round(img * 255).astype(np.uint8)).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as decoded:
        return (np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0).astype(img.dtype)


def corrupt(img, kind, level, rng=None):
    """Apply one corruption from :data:`CORRUPTIONS`.

    Parameters
    ----------
    img : numpy.ndarray
        Float ``(H, W, 3)`` image in [0, 1].
    kind : {"blur", "brightness", "noise", "jpeg"}
    level : float
        Sigma, factor, std or quality respectively.
    rng : numpy.random.Generator or int, optional
        Used by ``"noise"`` only.

    Returns
    -------
    numpy.ndarray
        Corrupted image, same shape and dtype.

    Raises
    ------
    ValueError
        If ``kind`` is unknown or ``level`` invalid for it.
    """
    if kind == "blur":
        return gaussian_blur(img, level)
    if kind == "brightness":
        return adjust_brightness(img, level)
    if kind == "noise":
        return add_gaussian_noise(img, level, rng)
    if kind == "jpeg":
        return jpeg_compress(img, level)
    raise ValueError(f"unknown corruption {kind!r}; choose from {tuple(CORRUPTIONS)}")
