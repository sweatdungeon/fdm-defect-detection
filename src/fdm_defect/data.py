"""Load, validate and split the FDM under-extrusion dataset.

The metadata CSV has one row per image with columns

========================  =====  ==============================================
column                    type   meaning
========================  =====  ==============================================
``img_path``              str    ``<printer_id>/<print_id>/<unix_timestamp>.jpg``,
                                 relative to the image folder
``printer_id``            int    printer that made the print (e.g. 22, 101)
``print_id``              int    print the frame belongs to
``has_under_extrusion``   int    label: 1 = under-extrusion, 0 = good
========================  =====  ==============================================

Every frame of a print carries the same label, and consecutive frames are
near-duplicates, so the independent unit is the *print*, not the image. The
split in :func:`split_by_print` therefore assigns whole prints to train,
validation or test, never individual images.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from . import config

#: Columns the metadata CSV must contain.
REQUIRED_COLUMNS = ("img_path", "printer_id", "print_id", "has_under_extrusion")

#: Name of the label column (1 = under-extrusion, 0 = good).
LABEL_COLUMN = "has_under_extrusion"

# <printer_id>/<print_id>/<timestamp>.<ext>, e.g. 022/1672709609/1672709790.713084.jpg
_IMG_PATH_RE = re.compile(r"^(\d+)/(\d+)/\d+(?:\.\d+)?\.(?:jpe?g|png)$", re.IGNORECASE)


class DataSplit(NamedTuple):
    """Train, validation and test metadata, each with whole prints only."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_metadata(csv_path, image_dir=None, check_files=True):
    """Read and validate the metadata CSV.

    Parameters
    ----------
    csv_path : str or Path
        CSV with the columns in :data:`REQUIRED_COLUMNS`.
    image_dir : str or Path, optional
        Folder that ``img_path`` is relative to. Required when
        ``check_files`` is True. When given, an absolute ``path`` column is
        added.
    check_files : bool, default True
        Check that every image file exists.

    Returns
    -------
    pandas.DataFrame
        Validated copy of the metadata (see :func:`validate_metadata`).

    Raises
    ------
    FileNotFoundError
        If the CSV, the image folder or any image is missing.
    TypeError, ValueError
        If the contents are invalid (see :func:`validate_metadata`).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"metadata CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return validate_metadata(df, image_dir=image_dir, check_files=check_files)


def validate_metadata(df, image_dir=None, check_files=True):
    """Check metadata for missing, malformed or inconsistent values.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata with the columns in :data:`REQUIRED_COLUMNS`.
    image_dir : str or Path, optional
        Folder that ``img_path`` is relative to. Required when
        ``check_files`` is True.
    check_files : bool, default True
        Check that every image file exists under ``image_dir``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with ``img_path`` normalised to forward slashes and,
        if ``image_dir`` is given, a ``path`` column with absolute paths.

    Raises
    ------
    TypeError
        If ``df`` is not a DataFrame, or an id or label column is not integer.
    ValueError
        If columns are missing, the table is empty, values are null, labels
        are not 0/1, ``img_path`` is malformed or disagrees with the id
        columns, paths are duplicated, a print has more than one label or
        printer, or ``check_files`` is True without ``image_dir``.
    FileNotFoundError
        If ``image_dir`` or any image file does not exist.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"metadata is missing columns: {missing}")
    if df.empty:
        raise ValueError("metadata has no rows")

    df = df.copy()
    null_cols = [c for c in REQUIRED_COLUMNS if df[c].isna().any()]
    if null_cols:
        raise ValueError(f"metadata has missing values in columns: {null_cols}")

    for col in ("printer_id", "print_id", LABEL_COLUMN):
        if not pd.api.types.is_integer_dtype(df[col]):
            raise TypeError(f"column {col!r} must be integer, got {df[col].dtype}")
    bad_labels = sorted(set(df[LABEL_COLUMN].unique()) - {0, 1})
    if bad_labels:
        raise ValueError(f"{LABEL_COLUMN!r} must be 0 or 1, found {bad_labels}")

    # Accept Windows-style separators but store paths in one canonical form.
    df["img_path"] = df["img_path"].astype(str).str.replace("\\", "/", regex=False)
    parts = df["img_path"].str.extract(_IMG_PATH_RE)
    malformed = parts[0].isna()
    if malformed.any():
        examples = df.loc[malformed, "img_path"].head(3).tolist()
        raise ValueError(
            f"{malformed.sum()} img_path values do not match "
            f"'<printer_id>/<print_id>/<timestamp>.jpg', e.g. {examples}"
        )
    # The folder names must agree with the id columns ("022" == 22).
    mismatch = (parts[0].astype(int) != df["printer_id"]) | (
        parts[1].astype(int) != df["print_id"]
    )
    if mismatch.any():
        examples = df.loc[mismatch, "img_path"].head(3).tolist()
        raise ValueError(
            f"{mismatch.sum()} img_path values disagree with printer_id/print_id, "
            f"e.g. {examples}"
        )

    dupes = df["img_path"].duplicated()
    if dupes.any():
        examples = df.loc[dupes, "img_path"].head(3).tolist()
        raise ValueError(f"{dupes.sum()} duplicated img_path values, e.g. {examples}")

    # The print-level split relies on each print having one label and one printer.
    per_print = df.groupby("print_id").agg(
        n_labels=(LABEL_COLUMN, "nunique"), n_printers=("printer_id", "nunique")
    )
    mixed = per_print.index[per_print["n_labels"] > 1].tolist()
    if mixed:
        raise ValueError(f"prints with more than one label: {mixed[:5]}")
    multi = per_print.index[per_print["n_printers"] > 1].tolist()
    if multi:
        raise ValueError(f"prints recorded on more than one printer: {multi[:5]}")

    if check_files and image_dir is None:
        raise ValueError("image_dir is required when check_files=True")
    if image_dir is not None:
        image_dir = Path(image_dir).resolve()
        if check_files and not image_dir.is_dir():
            raise FileNotFoundError(f"image folder not found: {image_dir}")
        df["path"] = [str(image_dir / p) for p in df["img_path"]]
        if check_files:
            absent = [p for p in df["path"] if not Path(p).is_file()]
            if absent:
                raise FileNotFoundError(
                    f"{len(absent)} of {len(df)} images are missing, e.g. {absent[:3]}"
                )
    return df


def load_image(path, size=config.DEFAULT_IMAGE_SIZE):
    """Load one image as a square RGB array scaled to [0, 1].

    Parameters
    ----------
    path : str or Path
        JPEG or PNG file. Greyscale and RGBA images are converted to RGB.
    size : int, default ``config.DEFAULT_IMAGE_SIZE``
        Output side length in pixels; the image is resized (bilinear) to
        ``size x size`` without preserving aspect ratio.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(size, size, 3)``, dtype float32, values in [0, 1].

    Raises
    ------
    TypeError
        If ``size`` is not an integer.
    ValueError
        If ``size`` is not positive or the file is not a readable image.
    FileNotFoundError
        If ``path`` does not exist.
    """
    # bool is a subclass of int, but size=True is certainly a mistake.
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise TypeError(f"size must be an integer, got {type(size).__name__}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
    except UnidentifiedImageError as err:
        raise ValueError(f"not a readable image: {path}") from err
    return arr / 255.0


def load_images(paths, size=config.DEFAULT_IMAGE_SIZE):
    """Load several images into one array.

    Parameters
    ----------
    paths : sequence of str or Path
        Image files, e.g. the ``path`` column from :func:`load_metadata`.
    size : int, default ``config.DEFAULT_IMAGE_SIZE``
        Output side length in pixels.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(paths), size, size, 3)``, float32 in [0, 1].

    Raises
    ------
    ValueError
        If ``paths`` is empty, or as for :func:`load_image`.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("paths is empty")
    return np.stack([load_image(p, size) for p in paths])


def print_table(df):
    """Summarise metadata to one row per print.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated metadata.

    Returns
    -------
    pandas.DataFrame
        Columns ``print_id``, ``printer_id``, ``has_under_extrusion`` and
        ``n_frames``, sorted by ``print_id``.
    """
    return (
        df.groupby("print_id")
        .agg(
            printer_id=("printer_id", "first"),
            has_under_extrusion=(LABEL_COLUMN, "first"),
            n_frames=("img_path", "size"),
        )
        .reset_index()
        .sort_values("print_id", ignore_index=True)
    )


def split_by_print(df, val_frac=0.15, test_frac=0.15, seed=config.DEFAULT_SEED):
    """Split metadata into train/val/test by print, stratified by label.

    Whole prints are assigned to one split, so near-duplicate frames of the
    same print can never appear on both sides of an evaluation. Good and
    defective prints are shuffled and divided separately, so each split gets
    about the same fraction of each class and at least one print of each.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated metadata (see :func:`validate_metadata`).
    val_frac, test_frac : float, default 0.15
        Fraction of the prints of each class sent to validation and test.
        Each must be in (0, 1) and together less than 1. Counts are rounded
        to the nearest whole print (halves round up), with a minimum of one.
    seed : int, default ``config.DEFAULT_SEED``
        Seed for the shuffle; the same seed always gives the same split.

    Returns
    -------
    DataSplit
        ``(train, val, test)`` DataFrames with reset indexes.

    Raises
    ------
    ValueError
        If the fractions are out of range, or a class has fewer than three
        prints (one is needed per split).
    """
    for name, frac in (("val_frac", val_frac), ("test_frac", test_frac)):
        if not 0 < frac < 1:
            raise ValueError(f"{name} must be in (0, 1), got {frac}")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be less than 1")

    prints = print_table(df)
    rng = np.random.default_rng(seed)
    split_ids = {"train": [], "val": [], "test": []}
    for label in (0, 1):
        ids = prints.loc[prints[LABEL_COLUMN] == label, "print_id"].to_numpy()
        n = len(ids)
        if n < 3:
            raise ValueError(
                f"need at least 3 prints with label {label} to fill every split, got {n}"
            )
        ids = rng.permutation(ids)
        # Round half up; Python's round() rounds half to even (round(4.5) == 4).
        n_test = max(1, int(np.floor(n * test_frac + 0.5)))
        n_val = max(1, int(np.floor(n * val_frac + 0.5)))
        if n - n_test - n_val < 1:
            raise ValueError(f"no label-{label} prints left for training; lower the fractions")
        split_ids["test"].extend(ids[:n_test])
        split_ids["val"].extend(ids[n_test : n_test + n_val])
        split_ids["train"].extend(ids[n_test + n_val :])

    split = DataSplit(
        *(
            df[df["print_id"].isin(split_ids[name])].reset_index(drop=True)
            for name in DataSplit._fields
        )
    )
    check_no_print_leakage(split)
    return split


def check_no_print_leakage(split):
    """Check that no print appears in more than one split.

    Parameters
    ----------
    split : DataSplit or sequence of pandas.DataFrame
        DataFrames with a ``print_id`` column.

    Raises
    ------
    ValueError
        If any ``print_id`` is shared between two of the DataFrames.
    """
    seen = {}
    for i, part in enumerate(split):
        for pid in part["print_id"].unique():
            if pid in seen:
                raise ValueError(f"print {pid} is in splits {seen[pid]} and {i}")
            seen[pid] = i
