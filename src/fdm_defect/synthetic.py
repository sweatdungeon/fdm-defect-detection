"""Synthetic bead-stripe images that mimic the nozzle-camera view.

A real frame shows parallel extruded beads beside the nozzle. Here a *good*
image is a set of evenly spaced, wide, continuous horizontal stripes; an
*under-extruded* image has the same spacing but thinner stripes with gaps
along their length, so less of the bed is covered by plastic.

These images are not a model of the real data. They exist so the tests can run
quickly, without the real dataset, on inputs whose properties are known.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .data import LABEL_COLUMN, REQUIRED_COLUMNS

#: Bead width as a fraction of the bead spacing for good prints.
GOOD_WIDTH = 0.85

#: Range the bead-width fraction is drawn from for under-extruded prints.
DEFECT_WIDTH_RANGE = (0.3, 0.5)


def make_bead_image(label, size=128, n_beads=8, noise_std=0.02, rng=None):
    """Draw one synthetic bead-stripe image.

    Parameters
    ----------
    label : {0, 1}
        0 for a good print (wide continuous beads), 1 for under-extrusion
        (thin beads with gaps).
    size : int, default 128
        Side length in pixels.
    n_beads : int, default 8
        Number of bead periods across the image height. If ``size`` is a
        multiple of ``n_beads`` a good image is exactly periodic vertically.
    noise_std : float, default 0.02
        Standard deviation of additive Gaussian noise, on a [0, 1] scale.
    rng : numpy.random.Generator or int, optional
        Random generator or seed. The same seed always gives the same image.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(size, size, 3)``, dtype uint8.

    Raises
    ------
    ValueError
        If ``label`` is not 0 or 1, ``size`` or ``n_beads`` is not positive,
        ``n_beads`` exceeds ``size``, or ``noise_std`` is negative.
    """
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label}")
    if size <= 0 or n_beads <= 0:
        raise ValueError("size and n_beads must be positive")
    if n_beads > size:
        raise ValueError(f"n_beads ({n_beads}) cannot exceed size ({size})")
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    rng = np.random.default_rng(rng)

    period = size / n_beads
    y = np.arange(size)[:, None]

    # Position within the bead period, t in [0, 1), with the bead centred at 0.5.
    t = (y % period) / period
    width = GOOD_WIDTH if label == 0 else rng.uniform(*DEFECT_WIDTH_RANGE)
    # Raised-cosine profile: 1 at the bead centre, falling to 0 at its edges.
    d = np.abs(t - 0.5) / (width / 2)
    bead = np.broadcast_to(np.where(d < 1, 0.5 * (1 + np.cos(np.pi * d)), 0.0), (size, size))

    if label == 1:
        # Knock 1-3 gaps out of each bead to mimic missing extrusion.
        present = np.ones((n_beads, size), dtype=bool)
        for k in range(n_beads):
            for _ in range(rng.integers(1, 4)):
                start = rng.integers(0, size)
                length = rng.integers(max(1, size // 10), max(2, size // 4))
                present[k, start : start + length] = False
        # Row r belongs to bead r // period; look up that bead's gap mask.
        bead_index = np.minimum((y[:, 0] // period).astype(int), n_beads - 1)
        bead = bead * present[bead_index]

    background = rng.uniform(0.1, 0.25)
    colour = rng.uniform(0.5, 1.0, size=3)
    img = background + bead[..., None] * (colour - background)
    img = img + rng.normal(0.0, noise_std, size=img.shape)
    return np.round(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def make_synthetic_dataset(
    out_dir,
    n_good_prints=4,
    n_defective_prints=4,
    frames_per_print=3,
    size=64,
    printers=(22, 101),
    seed=0,
):
    """Write a small synthetic dataset laid out like the real one.

    Images go to ``<out_dir>/images/<printer:03d>/<print_id>/<timestamp>.jpg``
    and the metadata to ``<out_dir>/metadata.csv``, with the same columns as
    the real CSV, so the result can be read with
    :func:`fdm_defect.data.load_metadata`.

    Parameters
    ----------
    out_dir : str or Path
        Output folder; created if needed.
    n_good_prints, n_defective_prints : int, default 4
        Number of prints of each class.
    frames_per_print : int, default 3
        Images per print, all sharing the print's label.
    size : int, default 64
        Image side length in pixels.
    printers : sequence of int, default (22, 101)
        Printer ids, assigned to prints in turn.
    seed : int, default 0
        Seed for all randomness.

    Returns
    -------
    pandas.DataFrame
        The metadata that was written (``img_path`` relative to ``images/``).

    Raises
    ------
    ValueError
        If a count is negative, there are no prints or frames, or
        ``printers`` is empty.
    """
    if min(n_good_prints, n_defective_prints) < 0:
        raise ValueError("print counts must be non-negative")
    if n_good_prints + n_defective_prints == 0 or frames_per_print < 1:
        raise ValueError("need at least one print and one frame per print")
    if len(printers) == 0:
        raise ValueError("printers is empty")

    rng = np.random.default_rng(seed)
    image_dir = Path(out_dir) / "images"
    labels = [0] * n_good_prints + [1] * n_defective_prints
    rows = []
    for i, label in enumerate(labels):
        printer_id = int(printers[i % len(printers)])
        # Fake unix start times, one print every ~3 hours.
        print_id = 1_700_000_000 + i * 10_000
        for j in range(frames_per_print):
            # Real frames are ~0.41 s apart; mirror that in the file names.
            rel = f"{printer_id:03d}/{print_id}/{print_id + 1 + 0.41 * j:.6f}.jpg"
            target = image_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(make_bead_image(label, size=size, rng=rng)).save(target, quality=95)
            rows.append((rel, printer_id, print_id, label))

    df = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype("int64")
    df.to_csv(Path(out_dir) / "metadata.csv", index=False)
    return df
