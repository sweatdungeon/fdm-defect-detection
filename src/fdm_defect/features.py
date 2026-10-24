"""Hand-crafted image features for the baseline classifiers, in NumPy.

Each image is reduced to a fixed-length vector built from six groups:

=====================  =========  =============================================
group                  length     what it captures
=====================  =========  =============================================
grey histogram         hist_bins  overall brightness distribution
RGB mean / std         6          filament and bed colour
Sobel gradients        5          edge strength and direction, edge density
uniform LBP histogram  10         local micro-texture
GLCM statistics        4          grey-level co-occurrence texture
2-D FFT periodicity    2          how regular the bead stripes are
=====================  =========  =============================================

Under-extrusion shows up as thinner, gapped beads, so it should change edge
density, texture and periodicity. All functions take float images in [0, 1],
as returned by :func:`fdm_defect.data.load_image`.
"""

from __future__ import annotations

import numpy as np

from . import config
from .data import load_image

#: ITU-R BT.601 luma weights used to convert RGB to grey.
GREY_WEIGHTS = np.array([0.299, 0.587, 0.114])

#: Default number of grey-histogram bins.
DEFAULT_HIST_BINS = 16

#: Default number of grey levels the GLCM quantises to.
DEFAULT_GLCM_LEVELS = 8

#: Default GLCM pixel offsets (dy, dx): right neighbour and lower neighbour.
DEFAULT_GLCM_OFFSETS = ((0, 1), (1, 0))

#: Default Sobel magnitude above which a pixel counts as an edge. A step of
#: 0.25 in grey value gives a magnitude of 1.0 (see :func:`sobel`).
DEFAULT_EDGE_THRESHOLD = 1.0

#: Uniform LBP with 8 neighbours: labels 0-8 (number of brighter neighbours)
#: plus one bin for all non-uniform patterns.
LBP_BINS = 10

# Neighbours (dy, dx) in circular order, clockwise from top-left. The order
# matters: uniformity counts 0/1 transitions around this circle.
_LBP_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


def _check_rgb(img):
    """Validate an RGB image and return it as float64."""
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {img.shape}")
    if img.shape[0] < 3 or img.shape[1] < 3:
        raise ValueError(f"image must be at least 3x3, got {img.shape[:2]}")
    if not np.issubdtype(img.dtype, np.floating):
        raise TypeError(
            f"expected a float image in [0, 1], got {img.dtype} (divide uint8 images by 255)"
        )
    if not np.isfinite(img).all():
        raise ValueError("image contains NaN or infinite values")
    if img.min() < 0 or img.max() > 1:
        raise ValueError(f"image values must be in [0, 1], got [{img.min()}, {img.max()}]")
    return img.astype(np.float64, copy=False)


def _check_grey(grey, unit_range=False):
    """Validate a 2-D grey image and return it as float64."""
    grey = np.asarray(grey)
    if grey.ndim != 2:
        raise ValueError(f"expected a 2-D grey image, got shape {grey.shape}")
    if grey.shape[0] < 3 or grey.shape[1] < 3:
        raise ValueError(f"image must be at least 3x3, got {grey.shape}")
    if not np.issubdtype(grey.dtype, np.floating):
        raise TypeError(f"expected a float image, got {grey.dtype}")
    if not np.isfinite(grey).all():
        raise ValueError("image contains NaN or infinite values")
    if unit_range and (grey.min() < 0 or grey.max() > 1):
        raise ValueError(f"image values must be in [0, 1], got [{grey.min()}, {grey.max()}]")
    return grey.astype(np.float64, copy=False)


def to_grey(img):
    """Convert an RGB image to grey with the BT.601 luma weights.

    Parameters
    ----------
    img : numpy.ndarray
        Float array of shape ``(H, W, 3)`` with values in [0, 1].

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(H, W)`` with values in [0, 1].
    """
    return _check_rgb(img) @ GREY_WEIGHTS


def grey_histogram(grey, bins=DEFAULT_HIST_BINS):
    """Normalised histogram of grey values over [0, 1].

    Parameters
    ----------
    grey : numpy.ndarray
        Float array of shape ``(H, W)`` with values in [0, 1].
    bins : int, default 16
        Number of equal-width bins.

    Returns
    -------
    numpy.ndarray
        Length-``bins`` array summing to 1.
    """
    grey = _check_grey(grey, unit_range=True)
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    counts, _ = np.histogram(grey, bins=bins, range=(0.0, 1.0))
    return counts / grey.size


def colour_stats(img):
    """Per-channel mean and standard deviation.

    Parameters
    ----------
    img : numpy.ndarray
        Float array of shape ``(H, W, 3)`` with values in [0, 1].

    Returns
    -------
    numpy.ndarray
        ``[mean_r, mean_g, mean_b, std_r, std_g, std_b]``.
    """
    img = _check_rgb(img)
    return np.concatenate([img.mean(axis=(0, 1)), img.std(axis=(0, 1))])


def sobel(grey):
    """Sobel gradients over the image interior.

    Uses the unnormalised kernels::

        Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
              [-2, 0, 2],            [ 0,  0,  0],
              [-1, 0, 1]]            [ 1,  2,  1]]

    so ``gx`` is positive where brightness increases to the right and ``gy``
    where it increases downwards. On a linear ramp ``g = s * x`` every
    interior pixel gives ``gx = 8 s``: each row of the kernel takes a central
    difference of ``2 s``, weighted 1 + 2 + 1 = 4. Border pixels are dropped
    (a "valid" convolution) instead of padded, so no edge artefacts appear.

    Parameters
    ----------
    grey : numpy.ndarray
        Float array of shape ``(H, W)``, at least 3x3.

    Returns
    -------
    gx, gy : numpy.ndarray
        Arrays of shape ``(H - 2, W - 2)``.
    """
    g = _check_grey(grey)
    # Shifted views of the image stand in for the kernel taps.
    left = g[:-2, :-2] + 2 * g[1:-1, :-2] + g[2:, :-2]
    right = g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]
    top = g[:-2, :-2] + 2 * g[:-2, 1:-1] + g[:-2, 2:]
    bottom = g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]
    return right - left, bottom - top


def gradient_features(grey, edge_threshold=DEFAULT_EDGE_THRESHOLD):
    """Summary statistics of the Sobel gradient.

    Parameters
    ----------
    grey : numpy.ndarray
        Float array of shape ``(H, W)``.
    edge_threshold : float, default 1.0
        Gradient magnitude above which a pixel counts as an edge.

    Returns
    -------
    numpy.ndarray
        ``[mean magnitude, std magnitude, mean |gx|, mean |gy|, edge density]``,
        where edge density is the fraction of interior pixels above the
        threshold. Horizontal beads give large ``|gy|`` and small ``|gx|``.
    """
    if edge_threshold < 0:
        raise ValueError(f"edge_threshold must be non-negative, got {edge_threshold}")
    gx, gy = sobel(grey)
    mag = np.hypot(gx, gy)
    return np.array(
        [mag.mean(), mag.std(), np.abs(gx).mean(), np.abs(gy).mean(), (mag > edge_threshold).mean()]
    )


def lbp_uniform(grey):
    """Uniform local binary pattern labels with 8 neighbours at radius 1.

    Each interior pixel gets one bit per neighbour (1 if the neighbour is at
    least as bright as the centre), read in circular order. A pattern is
    *uniform* if it has at most two 0/1 transitions around the circle; uniform
    patterns are labelled by their number of 1 bits (0-8) and all others get
    label 9. The 8 neighbours are the square 3x3 ring, not interpolated
    points on a circle.

    Parameters
    ----------
    grey : numpy.ndarray
        Float array of shape ``(H, W)``, at least 3x3.

    Returns
    -------
    numpy.ndarray
        Integer labels in 0-9, shape ``(H - 2, W - 2)``.
    """
    g = _check_grey(grey)
    h, w = g.shape
    centre = g[1:-1, 1:-1]
    bits = np.stack(
        [g[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx] >= centre for dy, dx in _LBP_OFFSETS]
    )
    ones = bits.sum(axis=0)
    # Compare each bit with the previous one around the circle.
    transitions = (bits != np.roll(bits, 1, axis=0)).sum(axis=0)
    return np.where(transitions <= 2, ones, LBP_BINS - 1)


def lbp_histogram(grey):
    """Normalised histogram of uniform LBP labels (see :func:`lbp_uniform`).

    Returns
    -------
    numpy.ndarray
        Length-10 array summing to 1.
    """
    labels = lbp_uniform(grey)
    return np.bincount(labels.ravel(), minlength=LBP_BINS) / labels.size


def glcm(grey, levels=DEFAULT_GLCM_LEVELS, offsets=DEFAULT_GLCM_OFFSETS):
    """Symmetric, normalised grey-level co-occurrence matrix.

    The image is quantised to ``levels`` grey levels (``floor(g * levels)``,
    with 1.0 mapped to the top level). For each offset ``(dy, dx)`` the pairs
    (pixel, pixel shifted by the offset) are counted, the counts are made
    symmetric by adding the transpose, and normalised to sum to 1. The result
    is the mean over offsets.

    Parameters
    ----------
    grey : numpy.ndarray
        Float array of shape ``(H, W)`` with values in [0, 1].
    levels : int, default 8
        Number of grey levels, at least 2.
    offsets : sequence of (int, int), default ((0, 1), (1, 0))
        Non-negative pixel offsets ``(dy, dx)``, not both zero, smaller than
        the image.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(levels, levels)`` array summing to 1.
    """
    g = _check_grey(grey, unit_range=True)
    if levels < 2:
        raise ValueError(f"levels must be at least 2, got {levels}")
    if len(offsets) == 0:
        raise ValueError("offsets is empty")
    h, w = g.shape
    q = np.minimum((g * levels).astype(int), levels - 1)
    p = np.zeros((levels, levels))
    for dy, dx in offsets:
        if dy < 0 or dx < 0 or (dy == 0 and dx == 0) or dy >= h or dx >= w:
            raise ValueError(f"invalid offset {(dy, dx)} for image of shape {g.shape}")
        a = q[: h - dy, : w - dx]
        b = q[dy:, dx:]
        # Encode each (a, b) pair as one integer so bincount counts pairs.
        counts = np.bincount((a * levels + b).ravel(), minlength=levels * levels)
        counts = counts.reshape(levels, levels)
        counts = counts + counts.T
        p += counts / counts.sum()
    return p / len(offsets)


def glcm_features(p):
    """Contrast, homogeneity, energy and correlation of a GLCM.

    With ``P`` the normalised symmetric GLCM and ``i, j`` the level indices:

    - contrast = sum P(i, j) (i - j)^2
    - homogeneity = sum P(i, j) / (1 + (i - j)^2)
    - energy = sqrt(sum P(i, j)^2)  (the scikit-image convention)
    - correlation = sum P(i, j) (i - mu)(j - mu) / sigma^2, defined as 1 when
      sigma = 0 (a constant image)

    A two-level checkerboard using levels 0 and L - 1 has every neighbour pair
    differing by L - 1, so its contrast is (L - 1)^2 and its correlation -1.

    Parameters
    ----------
    p : numpy.ndarray
        Square, symmetric GLCM summing to 1, e.g. from :func:`glcm`.

    Returns
    -------
    numpy.ndarray
        ``[contrast, homogeneity, energy, correlation]``.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(f"GLCM must be square, got shape {p.shape}")
    i, j = np.indices(p.shape)
    contrast = (p * (i - j) ** 2).sum()
    homogeneity = (p / (1 + (i - j) ** 2)).sum()
    energy = np.sqrt((p**2).sum())
    # P is symmetric, so the row and column means and variances are equal.
    mu = (i * p).sum()
    var = ((i - mu) ** 2 * p).sum()
    correlation = ((i - mu) * (j - mu) * p).sum() / var if var > 1e-12 else 1.0
    return np.array([contrast, homogeneity, energy, correlation])


def fft_features(grey):
    """Periodicity of the image from its 2-D power spectrum.

    The mean is subtracted and the zero-frequency term dropped, so only
    variation counts. Note the full (two-sided) FFT is used: a real cosine has
    two mirror-image peaks at +f and -f, so a pure cosine gives a peak
    fraction of exactly 0.5.

    Parameters
    ----------
    grey : numpy.ndarray
        Float array of shape ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        ``[peak fraction, peak frequency]``: the share of non-DC power in the
        single strongest frequency bin, and that bin's radial frequency in
        cycles per pixel. Both are 0 for a constant image. Regular, evenly
        spaced beads give a high peak fraction.
    """
    g = _check_grey(grey)
    power = np.abs(np.fft.fft2(g - g.mean())) ** 2
    power[0, 0] = 0.0
    total = power.sum()
    # A constant image leaves only rounding error in the spectrum.
    if total < 1e-12 * g.size:
        return np.zeros(2)
    ky, kx = np.unravel_index(np.argmax(power), power.shape)
    fy = np.fft.fftfreq(g.shape[0])[ky]
    fx = np.fft.fftfreq(g.shape[1])[kx]
    return np.array([power[ky, kx] / total, np.hypot(fx, fy)])


def feature_names(hist_bins=DEFAULT_HIST_BINS):
    """Names of the entries returned by :func:`extract_features`, in order.

    Parameters
    ----------
    hist_bins : int, default 16
        Must match the value passed to :func:`extract_features`.

    Returns
    -------
    list of str
    """
    return (
        [f"hist_{k:02d}" for k in range(hist_bins)]
        + ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]
        + ["grad_mean", "grad_std", "grad_abs_x", "grad_abs_y", "edge_density"]
        + [f"lbp_{k}" for k in range(LBP_BINS - 1)]
        + ["lbp_nonuniform"]
        + ["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]
        + ["fft_peak_fraction", "fft_peak_freq"]
    )


def extract_features(
    img,
    hist_bins=DEFAULT_HIST_BINS,
    glcm_levels=DEFAULT_GLCM_LEVELS,
    edge_threshold=DEFAULT_EDGE_THRESHOLD,
):
    """Compute the full hand-crafted feature vector for one image.

    Parameters
    ----------
    img : numpy.ndarray
        Float array of shape ``(H, W, 3)`` with values in [0, 1], at least 3x3.
    hist_bins : int, default 16
        Grey-histogram bins.
    glcm_levels : int, default 8
        GLCM grey levels.
    edge_threshold : float, default 1.0
        Sobel magnitude threshold for edge density.

    Returns
    -------
    numpy.ndarray
        Float64 vector, named by :func:`feature_names`
        (``hist_bins + 27`` entries).

    Raises
    ------
    TypeError
        If ``img`` is not a float array.
    ValueError
        If ``img`` has the wrong shape, is smaller than 3x3, contains
        non-finite values or values outside [0, 1], or a parameter is invalid.
    """
    img = _check_rgb(img)
    grey = img @ GREY_WEIGHTS
    return np.concatenate(
        [
            grey_histogram(grey, hist_bins),
            colour_stats(img),
            gradient_features(grey, edge_threshold),
            lbp_histogram(grey),
            glcm_features(glcm(grey, glcm_levels)),
            fft_features(grey),
        ]
    )


def extract_features_from_paths(paths, size=config.DEFAULT_IMAGE_SIZE, **kwargs):
    """Load images one at a time and compute their feature vectors.

    Images are loaded one by one rather than stacked, so memory use stays
    small (6,227 images at 128 px would be ~1.2 GB as float32).

    Parameters
    ----------
    paths : sequence of str or Path
        Image files, e.g. the ``path`` column from
        :func:`fdm_defect.data.load_metadata`.
    size : int, default ``config.DEFAULT_IMAGE_SIZE``
        Side length images are resized to before feature extraction.
    **kwargs
        Passed to :func:`extract_features`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(paths), n_features)``.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("paths is empty")
    return np.stack([extract_features(load_image(p, size), **kwargs) for p in paths])
