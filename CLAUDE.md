# Project context for Claude Code

## Assessment
MMA3001 Numerical Methods and Machine Learning, individual project (25%).
Due 25/10/2026. Deliverables: GitHub repo (README, pdoc HTML docs, MIT licence,
working main branch with docstrings and inline comments, pytest tests + HTML
test report), a 5-10 page written report, and a 5-minute presentation with
questions. The project brief PDF is the authority; ask me to paste it if needed.

Marking rubric emphasises: clear engineering problem and fully specified
inputs/outputs (format, units, domain, invalid-input handling); reproducible
repo with meaningful commits and branches; consistent docstrings (NumPy style);
rigorous validation; a baseline vs. main-method comparison with quantitative
evidence (accuracy, runtime, memory, FLOPs, arithmetic intensity, profiling);
honest limitations; and a disclosed, critically evaluated AI-use reflection.

## Project
Detect **under-extrusion** in FDM 3D printing from close-up nozzle-camera images.

- **Input:** RGB image of the extrusion region beside the nozzle (JPEG/PNG,
  8-bit; resized to a fixed square size, default 128 px) plus a metadata CSV.
- **Output:** probability of under-extrusion in [0, 1] and a binary label
  (1 = under-extrusion, 0 = good).
- **Intended use:** early-warning monitor to pause failing prints.

## Data (VERIFIED from the real CSVs on 2026-10-24)
Kaggle competition "Early detection of 3D printing issues":
https://www.kaggle.com/competitions/early-detection-of-3d-printing-issues/data
(must accept competition rules; CLI: `kaggle competitions download -c
early-detection-of-3d-printing-issues`). Full zip ~12 GB, 100k+ images.

- **train.csv:** 81,060 rows; columns `img_path`, `printer_id`, `print_id`,
  `has_under_extrusion` (1 = 44,342, 0 = 36,718).
- **test.csv:** 25,279 rows, NO labels (competition hold-out) -> not usable
  for validation. Ignore it, or use only for qualitative inspection.
- `img_path` format: `<printer_id>/<print_id>/<unix_timestamp>.jpg`,
  relative to the `images/` folder.
- **52 prints on 5 printers** (22, 101, 102, 103, 104); 107-4,310 frames per
  print, ~0.41 s apart (consecutive frames are near-duplicates).
- **Every print has a single label for all its frames (0 of 52 mixed).**
  So the label is effectively per print: the true independent sample size is
  52 prints, not 81k images. This must be stated in the report.
- Defective prints per printer: 22: 7/17, 101: 5/9, 102: 4/6, 103: 4/8,
  104: 5/12.
- **Subset used:** `data/train_subset.csv` = up to 120 evenly spaced frames
  from EVERY print -> 6,227 images, all 52 prints, 48% defective.
  `scripts/extract_subset.py` extracts just these images from the zip.
- Never commit images. Record the dataset licence (from the competition
  Rules tab) in data/README.md.

## Prior work found (another student's Kaggle notebook, TensorFlow)
Useful as a "lessons / what not to do" comparison, NOT as code to reuse:
trains on `y_col="printer_id"` (predicts which printer, not the defect);
random image-level `validation_split` (leaks near-duplicate frames);
softmax with binary_crossentropy; casts probabilities to int (all become 0);
training interrupted, no validation results. Cite it if discussed.

## Decisions already made
- Use the 6,227-image subset (sample FRAMES within prints, never drop whole
  prints: with only 52 prints every print matters). Run locally on CPU/MPS;
  Colab or a Kaggle notebook only if a run is too slow.
- **Split by print_id, never by image** (consecutive frames are near-duplicates;
  an image-level split leaks). Stratify prints by label so every split has
  both classes. Assert no print appears in two splits.
- Because there are only 52 independent prints, also report **grouped
  cross-validation over prints** (e.g. 5-fold StratifiedGroupKFold) for the
  baseline, and a **leave-one-printer-out** evaluation as the
  generalisation-to-a-new-printer test. Bootstrap CIs over prints will be
  wide; report them honestly.
- **Baseline:** hand-crafted features implemented in NumPy (grey histogram,
  RGB mean/std, Sobel gradient stats + edge density, uniform LBP histogram,
  GLCM contrast/homogeneity/energy/correlation, 2-D FFT periodicity) ->
  logistic regression and random forest (class_weight="balanced").
- **Main model:** ImageNet-pretrained MobileNetV3-Small (alt: ResNet-18), new
  1-logit head, head-only training by default, optional full fine-tune;
  BCEWithLogitsLoss with pos_weight, AdamW, flip/brightness augmentation,
  early stopping on validation loss restoring the best weights.
- Keep BatchNorm layers of a frozen backbone in eval mode during training.
- MobileNetV3 has BatchNorm momentum 0.01: trained from random weights it
  collapses to constant eval outputs. Always use pretrained weights.
- **Metrics:** recall is primary (a missed defect wastes the print), plus
  precision, F1, specificity, ROC-AUC; 95% CIs via bootstrap over prints
  (skip resamples with one class). Choose thresholds/settings on validation,
  touch test once.
- **Experiments:** resolution sweep (32/64/128/224) on validation; robustness
  on test images (blur, brightness, noise, JPEG) without retraining;
  profiling (perf_counter timing, cProfile, FLOPs via forward hooks for
  Conv2d/Linear with MAC = 2 FLOPs, arithmetic intensity).
- **Tests:** known-answer tests for the numerical routines (Sobel on a linear
  ramp = 8 x slope; pure cosine puts 0.5 of power in one FFT peak; GLCM
  contrast of a two-level checkerboard = (L-1)^2; FLOP counter vs hand
  count), metrics vs hand calculations, input-validation errors, split
  leakage, determinism with fixed seeds. Include a synthetic-data generator
  (bead-stripe images) so tests run without the real dataset.

## Suggested structure
src/<package>/ (config, data, features, baseline, cnn, evaluate, robustness,
profiling, synthetic), scripts/ (one per experiment), tests/, notebooks/,
data/README.md, docs/ (pdoc), results/, report/, AI_USE.md.

## Working rules
- Install the package with `pip install -e .` (pyproject.toml, src layout).
- Run `pytest` after every change; keep tests passing.
- Small, descriptive commits on feature branches, merged to main.
- Append every material AI contribution to AI_USE.md (date, task, what was
  kept/changed, how it was checked), including AI errors found.
- Explain non-obvious code to me; I must be able to defend every decision
  in the interview without AI.
