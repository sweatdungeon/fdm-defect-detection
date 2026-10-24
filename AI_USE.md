# AI use log

Record of every material AI contribution to this project, as required by the
assessment. Each entry gives the date, the tool, what was asked, what was kept
or changed, how it was checked, and any AI errors found.

Tool: Claude Code (Anthropic, model Claude Opus 5.5) unless stated otherwise.

---

## 2026-10-24: Repository and data setup

**Task.** Ignore the raw data in git, set up a Python virtual environment, extract
the image subset from the Kaggle zip, and commit.

**What the AI did**
- Added `data/raw/` and `*.zip` to `.gitignore` so images and archives are never
  committed.
- Created a `.venv` virtual environment (Python 3.12.15 via `uv`, since Python
  was not installed system-wide) and installed pandas 3.1.0.
- Identified `Downloads/images.zip` (12.2 GB) as the Kaggle image archive by
  checking its entry count: 106,339 = 81,060 train + 25,279 test images.
- Found and fixed a bug in `scripts/extract_subset.py`: it only matched images
  stored under an `images/` folder inside the zip, but `images.zip` stores them
  at the top level (`<printer_id>/<print_id>/<timestamp>.jpg`), so it would
  have extracted 0 images. The fix strips an `images/` prefix only if one is
  present.
- Ran the script to extract the 6,227 subset images into `data/raw/images/`.

**What was kept / changed.** All of the above was kept. Committed on branch
`chore/ignore-raw-data` and merged to `main`.

**How it was checked**
- Script output reported all 6,227 of 6,227 subset images found, with no
  missing-file warning; `find data/raw -name '*.jpg' | wc -l` gave 6227.
- `git check-ignore -v data/raw/images/022` confirmed the `data/raw/` rule
  applies, and `git ls-files data/raw` listed no tracked files.

**AI errors found.** None in this session's changes. The script bug above was
caught by inspecting the zip's file list before running the extraction.

---

## 2026-10-25: Package structure, data module, synthetic data and tests

**Task.** Set up the package (pyproject.toml, src layout, requirements.txt,
tests) and write the data module: load and validate the subset CSV and images,
and split by print with stratification by label. Add a synthetic data
generator and tests.

**What the AI did**
- `pyproject.toml` (setuptools, src layout, package `fdm_defect`), pinned
  `requirements.txt`, README setup/usage sections.
- `src/fdm_defect/config.py`: default paths, image size (128) and seed (42).
- `src/fdm_defect/data.py`:
  - `load_metadata` / `validate_metadata` check the required columns, empty
    tables, nulls, integer dtypes, 0/1 labels, `img_path` format and agreement
    with `printer_id`/`print_id`, duplicate paths, prints with mixed labels or
    printers, and that image files exist.
  - `load_image` / `load_images` return float32 RGB arrays in [0, 1].
  - `split_by_print` shuffles good and defective prints separately and assigns
    whole prints to train/val/test, then `check_no_print_leakage` confirms no
    print is in two splits.
- `src/fdm_defect/synthetic.py`: bead-stripe image generator (good = wide,
  continuous stripes; defect = thin stripes with gaps) and a writer that
  produces a dataset in the real folder/CSV layout.
- 59 pytest tests in `tests/`, including a test on the real subset that is
  skipped when the images are absent.

**What was kept / changed.** All kept. Real-data split with the defaults
(15% val, 15% test, seed 42): train 36 prints (4,307 images), val 8 prints
(960), test 8 prints (960), each with 4+ defective prints.

**How it was checked**
- `pytest`: 59 passed.
- Known-answer test: a noise-free good synthetic image with 8 beads has its
  strongest FFT frequency at index 8.
- Real data: all 6,227 images load; the 52 prints are split with no overlap and
  both classes in every split.

**AI errors found.** The first version of `split_by_print` used Python's
`round()`, which rounds halves to even (`round(4.5) == 4`), so the split sizes
did not match the intended "nearest whole print". A test caught it
(`test_split_no_training_prints_left`); fixed by rounding halves up.

---

## 2026-10-25: Hand-crafted features and baseline classifiers

**Task.** Write the NumPy feature module and the logistic regression / random
forest baseline from CLAUDE.md, with known-answer tests.

**What the AI did**
- `src/fdm_defect/features.py`: 43 features per image: 16-bin grey
  histogram, RGB mean/std, Sobel gradient stats (mean/std magnitude, mean
  |gx|, mean |gy|, edge density), 10-bin uniform LBP histogram (8 neighbours,
  radius 1), GLCM contrast/homogeneity/energy/correlation (8 levels, offsets
  right and down, symmetric), and 2-D FFT peak fraction and peak frequency.
  All implemented in NumPy only, with input validation.
- `src/fdm_defect/baseline.py`: `StandardScaler` + `LogisticRegression` and
  `RandomForestClassifier` (300 trees, `min_samples_leaf=5`), both
  `class_weight="balanced"`; `fit_baseline`, `predict_proba`,
  `compute_features`.
- `scripts/run_baseline.py`: trains on the train split and reports validation
  metrics (test split not used).
- `tests/test_features.py`, `tests/test_baseline.py`: 54 new tests.
- Added scikit-learn 1.9.1 as a dependency.

**What was kept / changed.** All kept. Design choices to be able to justify:
Sobel uses a "valid" convolution (borders dropped, no padding); LBP uses the
square 3x3 ring rather than interpolated circle points; GLCM energy follows
the scikit-image convention (square root of the angular second moment);
correlation is defined as 1 for a constant image.

**How it was checked**
- Known-answer tests: Sobel on a ramp g = s x gives gx = 8 s; a pure cosine
  puts 0.5 of non-DC power in its peak FFT bin at frequency k/N; a two-level
  checkerboard gives GLCM contrast (L - 1)^2, homogeneity 1/(1 + (L - 1)^2),
  energy sqrt(0.5), correlation -1; plus hand-worked LBP patterns, a step
  edge's edge density (2/6), and histogram endpoints.
- `pytest`: 113 passed.
- Real data (`run_baseline.py`, 128 px, seed 42), validation split, per
  image at threshold 0.5: logistic regression recall 0.710, ROC-AUC 0.715;
  random forest recall 0.715, ROC-AUC 0.798. Feature extraction 5.9 ms/image.
  Validation has only 8 prints, so these are noisy.

**AI errors found.** The first random-forest reproducibility test compared
probabilities for exact equality and failed by 2e-16: with `n_jobs=-1`,
per-tree probabilities are summed in thread order, so the last bit can vary.
The fitted trees are identical; the test now uses a 1e-12 relative tolerance.
