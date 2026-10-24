# Data

## Source

Kaggle competition **Early detection of 3D printing issues**:
https://www.kaggle.com/competitions/early-detection-of-3d-printing-issues/data

You must sign in to Kaggle and accept the competition rules before downloading.
With the Kaggle CLI configured:

```bash
kaggle competitions download -c early-detection-of-3d-printing-issues
```

The images may come as one large archive (`images.zip`, ~12 GB, 106,339 entries:
81,060 train + 25,279 test frames) plus `train.csv` and `test.csv`.

## Licence and terms of use

> **TODO (project owner):** copy the data licence / terms of use from the
> competition's **Rules** tab here (and the date you accepted them). It could
> not be retrieved automatically because the page requires a signed-in Kaggle
> account.

The images are **not** redistributed in this repository: `data/raw/` and
`*.zip` are git-ignored. Only the metadata subset `train_subset.csv` (file
paths and labels) is committed.

## What is in this folder

| Path | In git | Contents |
|---|---|---|
| `train_subset.csv` | yes | metadata for the 6,227-image subset used in this project |
| `raw/images/` | no | the subset images, created by `scripts/extract_subset.py` |
| `raw/*.csv` | no | top-level CSVs copied from the zip, if present |
| `features/` | no | cached feature matrices written by `scripts/run_pipeline.py` |

## Metadata format

One row per image, UTF-8 CSV with a header:

| Column | Type | Domain | Meaning |
|---|---|---|---|
| `img_path` | string | `<printer_id>/<print_id>/<unix_timestamp>.jpg` | image path relative to `raw/images/` (e.g. `022/1672709609/1672709790.713084.jpg`) |
| `printer_id` | int | 22, 101, 102, 103, 104 | printer that made the print |
| `print_id` | int | unix time (s) | start time of the print; identifies the print |
| `has_under_extrusion` | int | 0 or 1 | 1 = under-extrusion, 0 = good |

The timestamp in the file name is the capture time in seconds (with
microseconds). `fdm_defect.data.validate_metadata` checks all of the above,
including that the folder names in `img_path` match `printer_id` and
`print_id`.

## Images

- JPEG, 8-bit RGB, **1280 × 720 px**, close-up of the extrusion region beside the nozzle.
- Consecutive frames of a print are ~0.41 s apart and are near-duplicates.
- In this project images are resized (bilinear, aspect ratio not preserved) to a
  square, 128 × 128 px by default, and scaled to [0, 1].

## Full dataset statistics (verified from the CSVs on 2026-10-24)

| | Value |
|---|---|
| `train.csv` rows | 81,060 (44,342 defective, 36,718 good) |
| `test.csv` rows | 25,279, **no labels** (competition hold-out; not used here) |
| Prints | 52, on 5 printers |
| Frames per print | 107–4,310 |
| Prints with mixed labels | **0 of 52**: every print has a single label for all its frames |

Defective prints per printer:

| Printer | 22 | 101 | 102 | 103 | 104 | Total |
|---|---|---|---|---|---|---|
| Defective / total prints | 7 / 17 | 5 / 9 | 4 / 6 | 4 / 8 | 5 / 12 | 25 / 52 |

Because labels are per print and frames are near-duplicates, **the effective
sample size is 52 prints, not 81,060 images**. All splits and confidence
intervals in this project work at the print level.

## The subset (`train_subset.csv`)

- Up to **120 evenly spaced frames from every print** of `train.csv`; prints
  with fewer frames contribute all of them (minimum 107).
- 6,227 images, all 52 prints, 48% defective (25 defective prints).
- Frames are sampled *within* prints; no print is dropped, because with only
  52 prints every print matters.
- Rationale: 120 frames span each print evenly while cutting the data from
  ~12 GB to ~750 MB, so experiments run on a laptop CPU.

Extract the subset images from the downloaded zip (the zip does not need to be
unpacked):

```bash
python scripts/extract_subset.py --zip path/to/images.zip \
    --subset data/train_subset.csv --out data/raw
```

The script matches each `img_path` with or without an `images/` prefix inside
the zip, reports any missing files, and exits with an error if any are missing.
Afterwards the zip can be deleted.

## Split used in the experiments

`fdm_defect.data.split_by_print` with the defaults (15% validation, 15% test,
seed 42) assigns whole prints, stratified by label:

| Split | Prints | Defective prints | Images | Printers |
|---|---|---|---|---|
| train | 36 | 17 | 4,307 | 22, 101, 102, 103, 104 |
| val | 8 | 4 | 960 | 22, 101, 104 |
| test | 8 | 4 | 960 | 22, 102, 103, 104 |

The exact assignment is saved in `results/splits.csv`.
