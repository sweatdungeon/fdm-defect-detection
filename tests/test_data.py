"""Tests for fdm_defect.data: metadata validation, image loading and splitting."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fdm_defect import config
from fdm_defect.data import (
    DataSplit,
    check_no_print_leakage,
    load_image,
    load_images,
    load_metadata,
    print_table,
    split_by_print,
    validate_metadata,
)

# --------------------------------------------------------------------------
# load_metadata / validate_metadata
# --------------------------------------------------------------------------


def test_load_metadata_valid(synthetic_dir):
    df = load_metadata(synthetic_dir / "metadata.csv", synthetic_dir / "images")
    assert len(df) == 60
    assert df["path"].map(lambda p: p.endswith(".jpg")).all()


def test_load_metadata_missing_csv(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata CSV"):
        load_metadata(tmp_path / "nope.csv", tmp_path)


def test_validate_rejects_non_dataframe():
    with pytest.raises(TypeError):
        validate_metadata([1, 2, 3], check_files=False)


def test_validate_missing_column(synthetic_df):
    with pytest.raises(ValueError, match="missing columns"):
        validate_metadata(synthetic_df.drop(columns="print_id"), check_files=False)


def test_validate_empty(synthetic_df):
    with pytest.raises(ValueError, match="no rows"):
        validate_metadata(synthetic_df.iloc[:0], check_files=False)


def test_validate_null_values(synthetic_df):
    synthetic_df.loc[0, "img_path"] = None
    with pytest.raises(ValueError, match="missing values"):
        validate_metadata(synthetic_df, check_files=False)


def test_validate_non_integer_label(synthetic_df):
    synthetic_df["has_under_extrusion"] = synthetic_df["has_under_extrusion"].map(
        {0: "good", 1: "bad"}
    )
    with pytest.raises(TypeError, match="must be integer"):
        validate_metadata(synthetic_df, check_files=False)


def test_validate_label_out_of_range(synthetic_df):
    synthetic_df.loc[0, "has_under_extrusion"] = 2
    with pytest.raises(ValueError, match="must be 0 or 1"):
        validate_metadata(synthetic_df, check_files=False)


def test_validate_malformed_path(synthetic_df):
    synthetic_df.loc[0, "img_path"] = "not/a/valid/path.txt"
    with pytest.raises(ValueError, match="do not match"):
        validate_metadata(synthetic_df, check_files=False)


def test_validate_path_disagrees_with_ids(synthetic_df):
    synthetic_df.loc[0, "printer_id"] = 999
    with pytest.raises(ValueError, match="disagree"):
        validate_metadata(synthetic_df, check_files=False)


def test_validate_duplicate_paths(synthetic_df):
    dup = pd.concat([synthetic_df, synthetic_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        validate_metadata(dup, check_files=False)


def test_validate_mixed_label_print(synthetic_df):
    first_print = synthetic_df["print_id"] == synthetic_df.loc[0, "print_id"]
    idx = synthetic_df.index[first_print][0]
    synthetic_df.loc[idx, "has_under_extrusion"] = 1 - synthetic_df.loc[idx, "has_under_extrusion"]
    with pytest.raises(ValueError, match="more than one label"):
        validate_metadata(synthetic_df, check_files=False)


def test_validate_normalises_backslashes(synthetic_df):
    synthetic_df["img_path"] = synthetic_df["img_path"].str.replace("/", "\\")
    out = validate_metadata(synthetic_df, check_files=False)
    assert not out["img_path"].str.contains("\\\\").any()


def test_validate_requires_image_dir_to_check_files(synthetic_df):
    with pytest.raises(ValueError, match="image_dir is required"):
        validate_metadata(synthetic_df)


def test_validate_missing_image_file(synthetic_dir, synthetic_df):
    synthetic_df.loc[0, "img_path"] = "022/1700000000/1.000000.jpg"
    with pytest.raises(FileNotFoundError, match="1 of 60 images are missing"):
        validate_metadata(synthetic_df, synthetic_dir / "images")


def test_validate_does_not_modify_input(synthetic_df):
    before = synthetic_df.copy()
    validate_metadata(synthetic_df, check_files=False)
    pd.testing.assert_frame_equal(synthetic_df, before)


# --------------------------------------------------------------------------
# load_image / load_images
# --------------------------------------------------------------------------


def test_load_image_shape_dtype_range(synthetic_dir):
    path = next((synthetic_dir / "images").rglob("*.jpg"))
    img = load_image(path, size=32)
    assert img.shape == (32, 32, 3)
    assert img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0


def test_load_image_converts_greyscale_to_rgb(tmp_path):
    path = tmp_path / "grey.png"
    Image.fromarray(np.full((10, 20), 255, dtype=np.uint8), mode="L").save(path)
    img = load_image(path, size=8)
    assert img.shape == (8, 8, 3)
    np.testing.assert_allclose(img, 1.0)


@pytest.mark.parametrize("size, err", [(0, ValueError), (-5, ValueError), (2.5, TypeError), (True, TypeError)])
def test_load_image_bad_size(synthetic_dir, size, err):
    path = next((synthetic_dir / "images").rglob("*.jpg"))
    with pytest.raises(err):
        load_image(path, size=size)


def test_load_image_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.jpg")


def test_load_image_not_an_image(tmp_path):
    path = tmp_path / "fake.jpg"
    path.write_text("this is not a JPEG")
    with pytest.raises(ValueError, match="not a readable image"):
        load_image(path)


def test_load_images_stacks(synthetic_dir):
    df = load_metadata(synthetic_dir / "metadata.csv", synthetic_dir / "images")
    batch = load_images(df["path"].head(4), size=16)
    assert batch.shape == (4, 16, 16, 3)


def test_load_images_empty():
    with pytest.raises(ValueError, match="empty"):
        load_images([])


# --------------------------------------------------------------------------
# print_table / split_by_print / check_no_print_leakage
# --------------------------------------------------------------------------


@pytest.fixture
def valid_df(synthetic_df):
    return validate_metadata(synthetic_df, check_files=False)


def test_print_table(valid_df):
    prints = print_table(valid_df)
    assert len(prints) == 20
    assert (prints["n_frames"] == 3).all()
    assert prints["has_under_extrusion"].sum() == 10


def test_split_no_print_in_two_splits(valid_df):
    split = split_by_print(valid_df, seed=0)
    ids = [set(part["print_id"]) for part in split]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])


def test_split_uses_every_row_once(valid_df):
    split = split_by_print(valid_df, seed=0)
    combined = pd.concat(split)["img_path"]
    assert len(combined) == len(valid_df)
    assert set(combined) == set(valid_df["img_path"])


def test_split_counts_and_stratification(valid_df):
    # 10 prints per class, 15% each -> round(1.5) = 2 val and 2 test prints per class.
    split = split_by_print(valid_df, val_frac=0.15, test_frac=0.15, seed=0)
    for part, n_per_class in zip(split, (6, 2, 2)):
        counts = print_table(part)["has_under_extrusion"].value_counts()
        assert counts[0] == n_per_class and counts[1] == n_per_class


def test_split_keeps_whole_prints(valid_df):
    split = split_by_print(valid_df, seed=0)
    full = print_table(valid_df).set_index("print_id")["n_frames"]
    for part in split:
        per_print = part.groupby("print_id").size()
        assert (per_print == full[per_print.index]).all()


def test_split_deterministic(valid_df):
    a = split_by_print(valid_df, seed=123)
    b = split_by_print(valid_df, seed=123)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)


def test_split_depends_on_seed(valid_df):
    test_sets = {frozenset(split_by_print(valid_df, seed=s).test["print_id"]) for s in range(10)}
    assert len(test_sets) > 1


def test_split_returns_named_tuple(valid_df):
    split = split_by_print(valid_df)
    assert isinstance(split, DataSplit)
    assert split.train is split[0]


@pytest.mark.parametrize("val_frac, test_frac", [(0, 0.2), (0.2, 1.0), (-0.1, 0.2), (0.5, 0.5)])
def test_split_bad_fractions(valid_df, val_frac, test_frac):
    with pytest.raises(ValueError):
        split_by_print(valid_df, val_frac=val_frac, test_frac=test_frac)


def test_split_too_few_prints(valid_df):
    # Keep only two defective prints: not enough for train, val and test.
    keep = print_table(valid_df).query("has_under_extrusion == 0 or print_id < 1700120000")
    small = valid_df[valid_df["print_id"].isin(keep["print_id"])]
    assert print_table(small)["has_under_extrusion"].sum() == 2
    with pytest.raises(ValueError, match="at least 3 prints with label 1"):
        split_by_print(small)


def test_split_no_training_prints_left(valid_df):
    # 10 prints per class, 45% + 45% -> 5 val + 5 test, 0 train.
    with pytest.raises(ValueError, match="left for training"):
        split_by_print(valid_df, val_frac=0.45, test_frac=0.45)


def test_check_no_print_leakage_detects_overlap(valid_df):
    split = split_by_print(valid_df, seed=0)
    leaky = DataSplit(split.train, pd.concat([split.val, split.train.head(1)]), split.test)
    with pytest.raises(ValueError, match="is in splits 0 and 1"):
        check_no_print_leakage(leaky)


# --------------------------------------------------------------------------
# Real data (skipped unless the subset images have been extracted)
# --------------------------------------------------------------------------

needs_real_data = pytest.mark.skipif(
    not config.IMAGE_DIR.is_dir(), reason="real images not extracted to data/raw/images"
)


@needs_real_data
def test_real_subset_loads_and_splits():
    df = load_metadata(config.SUBSET_CSV, config.IMAGE_DIR)
    assert len(df) == 6227
    assert df["print_id"].nunique() == 52
    split = split_by_print(df)
    assert sum(part["print_id"].nunique() for part in split) == 52
    for part in split:
        assert set(part["has_under_extrusion"]) == {0, 1}
