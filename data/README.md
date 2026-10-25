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

*Rules tab accessed 25 October 2026.*

- **No formal data licence.** The competition's Rules tab does not state a
  data licence. It asks participants to use common sense, not to cheat, and
  requires prize winners to open-source their code.
- **Kaggle's general [Terms of Use](https://www.kaggle.com/terms) apply** to
  the data, as to all content obtained through Kaggle.
- **Use in this project:** non-commercial academic coursework (MMA3001
  individual project) only.
- **No images are redistributed.** `data/raw/` and `*.zip` are git-ignored;
  the repository contains only `train_subset.csv`, a list of file paths with
  their printer, print and label, from which the images can be extracted
  from the user's own Kaggle download.

## What is in this folder

| Path | In git | Contents |
|---|---|---|
| `train_subset.csv` | yes | metadata for the 6,227-image subset used in this project |
| `raw/images/` | no | the subset images, created by `scripts/extract_subset.py` |
| `raw/*.csv` | no | competition CSVs (e.g. `train.csv`, for `scripts/make_subset.py` and its test) |
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
  ~12 GB to ~750 MB, so experiments run on an ordinary desktop CPU.

**Exact procedure** (`fdm_defect.data.make_subset`): sort `train.csv` by
`img_path` (ordering each print's frames by timestamp), group by `print_id`,
and for a print with `n` frames keep the row positions
`np.unique(np.linspace(0, n - 1, min(120, n)).round().astype(int))`; then
concatenate the prints in `print_id` order and reset the index. `np.round`
rounds halves to even (4.5 → 4); the committed file depends on this.

Rebuild or verify the subset from the competition's `train.csv` (the CSV or
the Kaggle `train.csv.zip`):

```bash
python scripts/make_subset.py --train path/to/train.csv.zip --check   # verify
python scripts/make_subset.py --train path/to/train.csv               # rebuild
```

The rebuilt file is byte-identical to the committed `train_subset.csv`
(verified 25 October 2026). `tests/test_data.py` checks this whenever
`data/raw/train.csv` is present.

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
