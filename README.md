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

### Baseline

43 hand-crafted NumPy features per image (`fdm_defect.features`: grey
histogram, RGB mean/std, Sobel gradients, uniform LBP, GLCM, FFT periodicity)
feed a logistic regression or random forest (`fdm_defect.baseline`). Train
both and score on the validation split with:

```bash
python scripts/run_baseline.py        # writes results/baseline_val.csv
```

### CNN

`fdm_defect.cnn` fine-tunes an ImageNet-pretrained MobileNetV3-Small (or
ResNet-18 with `--arch resnet18`) with a new single-logit head. By default
only the head is trained and the frozen backbone's BatchNorm layers stay in
eval mode; `--full-finetune` trains everything (use a lower `--lr`, e.g. 1e-4).

```bash
python scripts/train_cnn.py           # writes results/cnn_*.csv and models/cnn_*.pt
```

The first run downloads the ImageNet weights (~10 MB) to the torch cache.

## Experiments

One command runs every experiment on the real data (~12 minutes on a laptop
CPU) and writes CSVs, figures and the comparison table to `results/`:

```bash
python scripts/run_pipeline.py                     # all stages
python scripts/run_pipeline.py --stages sweep report   # or a subset
python scripts/make_report.py                      # rebuild table + figures only
```

| Stage | What it does | Output |
|---|---|---|
| splits | print-level train/val/test split (36/8/8 prints) | `splits.csv` |
| features | 43 hand-crafted features per image at 32/64/128/224 px | `data/features/` (cache) |
| baseline | logistic regression + random forest, 128 px | `predictions.csv` |
| cnn | MobileNetV3-Small, head only, max 10 epochs, 128 px | `cnn_history.csv` |
| cv | 5-fold StratifiedGroupKFold over prints (baselines) | `cv_folds.csv` |
| lopo | leave-one-printer-out (baselines) | `lopo.csv` |
| sweep | resolution 32/64/128/224 px, validation split | `resolution_sweep.csv` |
| robustness | blur, brightness, noise, JPEG on test images | `robustness.csv` |
| profiling | timing, cProfile, FLOPs, arithmetic intensity | `profiling.csv`, `cnn_layer_flops.csv`, `cprofile_features.txt` |
| report | bootstrap CIs over prints, comparison table, figures | `test_metrics.csv`, `comparison.md`, `figures/` |

Settings are chosen on the validation split; the test split is scored once per
final model. See [results/comparison.md](results/comparison.md).

## Tests

```bash
pytest                                     # synthetic data; real-data test runs if images exist
pytest --html=report.html --self-contained-html   # HTML test report
```

## Licence

MIT, see [LICENSE](LICENSE).
