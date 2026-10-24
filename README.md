# fdm-defect-detection

Detect **under-extrusion** in FDM 3D printing from close-up nozzle-camera images.

- **Input:** RGB image of the extrusion region beside the nozzle (JPEG/PNG,
  8-bit, resized to a square, default 128 px) plus a metadata CSV.
- **Output:** probability of under-extrusion in [0, 1] and a binary label
  (1 = under-extrusion, 0 = good).

## Setup

Requires Python 3.10+ (developed on 3.12).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Images come from the Kaggle competition
[Early detection of 3D printing issues](https://www.kaggle.com/competitions/early-detection-of-3d-printing-issues/data)
(accept the competition rules first). The full download is ~12 GB; only a
6,227-image subset listed in `data/train_subset.csv` is used. Extract it with:

```bash
python scripts/extract_subset.py --zip path/to/images.zip \
    --subset data/train_subset.csv --out data/raw
```

Images go to `data/raw/images/`, which is git-ignored.

## Usage

```python
from fdm_defect import config
from fdm_defect.data import load_metadata, split_by_print, load_images

df = load_metadata(config.SUBSET_CSV, config.IMAGE_DIR)   # validated metadata
train, val, test = split_by_print(df)                    # whole prints per split
x = load_images(train["path"].head(8), size=128)          # (8, 128, 128, 3) float32
```

The split is by print, not by image: consecutive frames of one print are
near-duplicates and share a label, so an image-level split would leak.

## Tests

```bash
pytest                                     # synthetic data; real-data test runs if images exist
pytest --html=report.html --self-contained-html   # HTML test report
```

## Licence

MIT, see [LICENSE](LICENSE).
