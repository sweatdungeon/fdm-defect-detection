"""Run every experiment on the real data and save the results in results/.

Stages (run in this order; pick a subset with --stages):

  splits      print-level train/val/test split               -> results/splits.csv
  features    hand-crafted features for every image, cached  -> data/features/
  baseline    logistic regression + random forest at 128 px  -> results/predictions.csv
  cnn         MobileNetV3-Small, head only, 10 epochs, 128 px -> results/cnn_history.csv
  cv          5-fold grouped CV over prints (baseline)       -> results/cv_folds.csv
  lopo        leave-one-printer-out (baseline)               -> results/lopo.csv
  sweep       resolution sweep 32/64/128/224 on validation   -> results/resolution_sweep.csv
  robustness  corrupted test images, no retraining           -> results/robustness.csv
  profiling   timing, cProfile, FLOPs, arithmetic intensity  -> results/profiling.csv, ...
  report      metrics with CIs, comparison table, figures    (scripts/make_report.py)

Protocol: settings are chosen on validation only. The test split is scored
once per final model (baseline and CNN at 128 px) and in the robustness test,
which uses the same fixed models. CV and leave-one-printer-out use all 52
prints to estimate how much results vary between prints and printers; they are
reported, never used to pick settings.

Usage:
  python scripts/run_pipeline.py [--stages splits features ...] [--seed 42]
"""

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fdm_defect import config
from fdm_defect.baseline import MODEL_NAMES, fit_baseline, predict_proba
from fdm_defect.cnn import (
    build_model,
    normalise,
    predict_proba as cnn_predict_proba,
    predict_proba_images,
    seed_everything,
    train_model,
)
from fdm_defect.data import LABEL_COLUMN, load_image, load_images, load_metadata, print_table, split_by_print
from fdm_defect.evaluate import binary_metrics, grouped_cv, leave_one_group_out
from fdm_defect.features import extract_features, extract_features_from_paths
from fdm_defect.profiling import (
    count_flops,
    count_parameters,
    model_size_bytes,
    profile_function,
    summarise_flops,
    time_function,
)
from fdm_defect.robustness import CORRUPTIONS, corrupt

STAGES = ("splits", "features", "baseline", "cnn", "cv", "lopo", "sweep", "robustness", "profiling", "report")
SIZES = (32, 64, 128, 224)
MAIN_SIZE = 128
CNN_EPOCHS = 10
CNN_ARCH = "mobilenet_v3_small"
FEATURE_DIR = config.DATA_DIR / "features"
RESULTS = Path("results")
MODELS = Path("models")
CNN_WEIGHTS = MODELS / f"cnn_{CNN_ARCH}_head_{MAIN_SIZE}.pt"
#: Display names used in every table and figure.
LABELS = {"logreg": "Logistic regression", "random_forest": "Random forest", "cnn": "MobileNetV3-Small"}


class Context:
    """Data shared by all stages: metadata, split and cached features."""

    def __init__(self, seed):
        """Load and validate the metadata, make the split, and index rows by image path."""
        self.seed = seed
        self.df = load_metadata(config.SUBSET_CSV, config.IMAGE_DIR)
        self.split = split_by_print(self.df, seed=seed)
        # Row position of every image in self.df, to index feature matrices.
        self.row = pd.Series(np.arange(len(self.df)), index=self.df["img_path"])
        self._features = {}

    def rows(self, part):
        """Row positions in ``self.df`` of the images in ``part``."""
        return self.row[part["img_path"]].to_numpy()

    def features(self, size):
        """Feature matrix for every image at ``size``, rows in ``self.df`` order."""
        if size not in self._features:
            path = FEATURE_DIR / f"features_{size}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"{path} missing; run the 'features' stage first")
            cached = np.load(path, allow_pickle=False)
            if not np.array_equal(cached["img_path"], self.df["img_path"].to_numpy(dtype=str)):
                raise RuntimeError(f"{path} does not match the metadata; rerun 'features'")
            self._features[size] = cached["x"]
        return self._features[size]

    def xy(self, part, size):
        """Feature matrix and labels for ``part`` at resolution ``size``."""
        return self.features(size)[self.rows(part)], part[LABEL_COLUMN].to_numpy()


def log(msg):
    """Print ``msg`` with a timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage_splits(ctx):
    """Save the print-level split and log a per-split summary."""
    rows = []
    for name, part in zip(ctx.split._fields, ctx.split):
        t = print_table(part)
        t["split"] = name
        rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(RESULTS / "splits.csv", index=False)
    summary = out.groupby("split", sort=False).agg(
        prints=("print_id", "size"),
        defective_prints=(LABEL_COLUMN, "sum"),
        images=("n_frames", "sum"),
        printers=("printer_id", lambda s: " ".join(map(str, sorted(s.unique())))),
    )
    log("splits:\n" + summary.to_string())


def stage_features(ctx):
    """Extract and cache features for every image at every sweep resolution."""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    timing = []
    for size in SIZES:
        t0 = time.perf_counter()
        x = extract_features_from_paths(ctx.df["path"], size=size)
        secs = time.perf_counter() - t0
        # Fixed-width unicode, not object dtype, so the cache loads without pickle.
        np.savez(FEATURE_DIR / f"features_{size}.npz", x=x, img_path=ctx.df["img_path"].to_numpy(dtype=str))
        timing.append({"size": size, "images": len(x), "seconds": secs, "ms_per_image_incl_decode": 1000 * secs / len(x)})
        log(f"features {size}px: {len(x)} images in {secs:.1f} s")
    pd.DataFrame(timing).to_csv(RESULTS / "feature_timing.csv", index=False)


def prediction_frame(part, split_name, model, proba):
    """Table of per-image predictions for one model on one split."""
    return pd.DataFrame({
        "img_path": part["img_path"].to_numpy(),
        "print_id": part["print_id"].to_numpy(),
        "printer_id": part["printer_id"].to_numpy(),
        "label": part[LABEL_COLUMN].to_numpy(),
        "split": split_name,
        "model": model,
        "proba": proba,
    })


def save_predictions(new, models):
    """Replace the rows for ``models`` in results/predictions.csv with ``new``."""
    path = RESULTS / "predictions.csv"
    old = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    if len(old):
        old = old[~old["model"].isin(models)]
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


def stage_baseline(ctx):
    """Fit both baselines at 128 px and save their validation and test predictions."""
    x_train, y_train = ctx.xy(ctx.split.train, MAIN_SIZE)
    preds, fits = [], []
    for name in MODEL_NAMES:
        t0 = time.perf_counter()
        model = fit_baseline(name, x_train, y_train, seed=ctx.seed)
        fits.append({"model": name, "fit_seconds": time.perf_counter() - t0})
        for split_name in ("val", "test"):
            part = getattr(ctx.split, split_name)
            x, y = ctx.xy(part, MAIN_SIZE)
            proba = predict_proba(model, x)
            preds.append(prediction_frame(part, split_name, name, proba))
            m = binary_metrics(y, proba)
            log(f"{name:13s} {split_name:4s} recall {m['recall']:.3f}  AUC {m['roc_auc']:.3f}")
    save_predictions(pd.concat(preds), MODEL_NAMES)
    pd.DataFrame(fits).to_csv(RESULTS / "baseline_fit_times.csv", index=False)


def train_cnn(ctx, size):
    """Seed, build and train the head-only CNN at ``size``; return the result and seconds taken."""
    seed_everything(ctx.seed)
    model = build_model(CNN_ARCH, pretrained=True, freeze_backbone=True)
    t0 = time.perf_counter()
    result = train_model(
        model, ctx.split.train, ctx.split.val, size=size, epochs=CNN_EPOCHS,
        seed=ctx.seed, verbose=True,
    )
    return result, time.perf_counter() - t0


def stage_cnn(ctx):
    """Train the main CNN, save its weights, history and validation/test predictions."""
    result, secs = train_cnn(ctx, MAIN_SIZE)
    history = pd.DataFrame(result.history)
    history["best_epoch"] = result.best_epoch
    history.to_csv(RESULTS / "cnn_history.csv", index=False)
    MODELS.mkdir(exist_ok=True)
    torch.save(result.model.state_dict(), CNN_WEIGHTS)
    preds = []
    for split_name in ("val", "test"):
        part = getattr(ctx.split, split_name)
        proba = cnn_predict_proba(result.model, part, size=MAIN_SIZE)
        preds.append(prediction_frame(part, split_name, "cnn", proba))
        m = binary_metrics(part[LABEL_COLUMN], proba)
        log(f"cnn {split_name:4s} recall {m['recall']:.3f}  AUC {m['roc_auc']:.3f}")
    save_predictions(pd.concat(preds), ["cnn"])
    pd.DataFrame([{"model": "cnn", "fit_seconds": secs, "best_epoch": result.best_epoch,
                   "epochs_run": len(result.history)}]).to_csv(RESULTS / "cnn_fit_time.csv", index=False)


def load_main_cnn():
    """Load the CNN weights saved by the 'cnn' stage, in eval mode."""
    if not CNN_WEIGHTS.is_file():
        raise FileNotFoundError(f"{CNN_WEIGHTS} missing; run the 'cnn' stage first")
    model = build_model(CNN_ARCH, pretrained=False, freeze_backbone=True)
    model.load_state_dict(torch.load(CNN_WEIGHTS, weights_only=True))
    return model.eval()


def stage_cv(ctx):
    """5-fold grouped CV over all prints for both baselines."""
    x, y = ctx.xy(ctx.df, MAIN_SIZE)
    folds = []
    for name in MODEL_NAMES:
        f = grouped_cv(x, y, ctx.df["print_id"].to_numpy(), name, n_splits=5, seed=ctx.seed)
        f.insert(0, "model", name)
        folds.append(f)
        log(f"cv {name:13s} AUC {f['roc_auc'].mean():.3f} +/- {f['roc_auc'].std():.3f}, "
            f"recall {f['recall'].mean():.3f} +/- {f['recall'].std():.3f}")
    pd.concat(folds).to_csv(RESULTS / "cv_folds.csv", index=False)


def stage_lopo(ctx):
    """Leave-one-printer-out evaluation for both baselines."""
    x, y = ctx.xy(ctx.df, MAIN_SIZE)
    printers = ctx.df["printer_id"].to_numpy()
    prints_per_printer = ctx.df.groupby("printer_id")["print_id"].nunique()
    folds = []
    for name in MODEL_NAMES:
        f = leave_one_group_out(x, y, printers, name, seed=ctx.seed)
        f.insert(0, "model", name)
        f["held_out_prints"] = f["held_out"].map(prints_per_printer)
        folds.append(f)
        log(f"lopo {name:13s} AUC per printer: " + ", ".join(
            f"{p}: {a:.2f}" for p, a in zip(f["held_out"], f["roc_auc"])))
    pd.concat(folds).to_csv(RESULTS / "lopo.csv", index=False)


def stage_sweep(ctx):
    """Validation metrics for all three models at each sweep resolution."""
    rows = []
    for size in SIZES:
        x_train, y_train = ctx.xy(ctx.split.train, size)
        x_val, y_val = ctx.xy(ctx.split.val, size)
        for name in MODEL_NAMES:
            t0 = time.perf_counter()
            model = fit_baseline(name, x_train, y_train, seed=ctx.seed)
            secs = time.perf_counter() - t0
            rows.append({"model": name, "size": size, "fit_seconds": secs,
                         **binary_metrics(y_val, predict_proba(model, x_val))})
        log(f"sweep {size}px: training CNN")
        result, secs = train_cnn(ctx, size)
        proba = cnn_predict_proba(result.model, ctx.split.val, size=size)
        rows.append({"model": "cnn", "size": size, "fit_seconds": secs, "best_epoch": result.best_epoch,
                     **binary_metrics(y_val, proba)})
        log("sweep " + ", ".join(f"{r['model']} AUC {r['roc_auc']:.3f}" for r in rows if r["size"] == size))
    pd.DataFrame(rows).to_csv(RESULTS / "resolution_sweep.csv", index=False)


def stage_robustness(ctx):
    """Score the fixed models on corrupted test images, without retraining."""
    test = ctx.split.test
    y = test[LABEL_COLUMN].to_numpy()
    images = load_images(test["path"], size=MAIN_SIZE)
    x_train, y_train = ctx.xy(ctx.split.train, MAIN_SIZE)
    baselines = {name: fit_baseline(name, x_train, y_train, seed=ctx.seed) for name in MODEL_NAMES}
    cnn = load_main_cnn()

    conditions = [("clean", None)] + [(kind, level) for kind, levels in CORRUPTIONS.items() for level in levels]
    rows = []
    for kind, level in conditions:
        rng = np.random.default_rng(ctx.seed)
        corrupted = images if kind == "clean" else np.stack([corrupt(im, kind, level, rng) for im in images])
        x = np.stack([extract_features(im) for im in corrupted])
        probas = {name: predict_proba(model, x) for name, model in baselines.items()}
        probas["cnn"] = predict_proba_images(cnn, corrupted)
        for name, proba in probas.items():
            rows.append({"corruption": kind, "level": level, "model": name, **binary_metrics(y, proba)})
        log(f"robustness {kind} {level}: " + ", ".join(
            f"{r['model']} AUC {r['roc_auc']:.3f}" for r in rows[-3:]))
    pd.DataFrame(rows).to_csv(RESULTS / "robustness.csv", index=False)


def stage_profiling(ctx):
    """Time, profile and count FLOPs for the baselines and the CNN."""
    rows = []

    def add(model, metric, value, unit, note=""):
        """Append one profiling measurement."""
        rows.append({"model": model, "metric": metric, "value": value, "unit": unit, "note": note})

    test = ctx.split.test
    images = load_images(test["path"].head(64), size=MAIN_SIZE)
    one = images[:1]

    # Hand-crafted features: time, peak Python/NumPy memory, cProfile.
    t = time_function(lambda: extract_features(images[0]), repeats=100, warmup=5)
    add("features", "time_per_image", 1000 * t["median"], "ms", f"median of {t['repeats']}, {MAIN_SIZE}px, excludes JPEG decode")
    decode = time_function(lambda: load_image(test["path"].iloc[0], MAIN_SIZE), repeats=30, warmup=3)
    add("all", "jpeg_decode_resize_per_image", 1000 * decode["median"], "ms", "1280x720 JPEG -> 128px, same for every model")
    tracemalloc.start()
    extract_features(images[0])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    add("features", "peak_memory_per_image", peak / 1e6, "MB", "tracemalloc peak during one extraction")
    report = profile_function(lambda: [extract_features(im) for im in images], top=25)
    (RESULTS / "cprofile_features.txt").write_text(report)

    x_train, y_train = ctx.xy(ctx.split.train, MAIN_SIZE)
    x_test, _ = ctx.xy(test, MAIN_SIZE)
    for name in MODEL_NAMES:
        model = fit_baseline(name, x_train, y_train, seed=ctx.seed)
        single = time_function(lambda: predict_proba(model, x_test[:1]), repeats=50, warmup=3)
        batch = time_function(lambda: predict_proba(model, x_test), repeats=10, warmup=2)
        add(name, "predict_time_batch1", 1000 * single["median"], "ms")
        add(name, "predict_time_per_image_batched", 1000 * batch["median"] / len(x_test), "ms", f"batch of {len(x_test)}")
        add(name, "model_size", model_size_bytes(model) / 1e6, "MB", "pickled")
    add("random_forest", "parameters", sum(e.tree_.node_count for e in model.estimators_), "nodes", "total tree nodes")
    add("logreg", "parameters", x_train.shape[1] + 1, "coefficients")

    # CNN: timing, size, FLOPs at every sweep resolution, per-layer table at 128 px.
    cnn = load_main_cnn()
    x1 = normalise(torch.from_numpy(one).permute(0, 3, 1, 2))
    x64 = normalise(torch.from_numpy(images).permute(0, 3, 1, 2))
    with torch.no_grad():
        single = time_function(lambda: cnn(x1), repeats=50, warmup=5)
        batch = time_function(lambda: cnn(x64), repeats=10, warmup=2)
    add("cnn", "predict_time_batch1", 1000 * single["median"], "ms", f"{torch.get_num_threads()} CPU threads")
    add("cnn", "predict_time_per_image_batched", 1000 * batch["median"] / len(images), "ms", f"batch of {len(images)}")
    add("cnn", "parameters", count_parameters(cnn), "params")
    add("cnn", "trainable_parameters", count_parameters(cnn, trainable_only=True), "params", "head only")
    add("cnn", "model_size", model_size_bytes(cnn) / 1e6, "MB", "float32 params + buffers")
    for size in SIZES:
        totals = summarise_flops(count_flops(cnn, (1, 3, size, size)))
        add("cnn", f"gflops_{size}px", totals["flops"] / 1e9, "GFLOP", "conv + linear, 1 MAC = 2 FLOPs")
        add("cnn", f"arithmetic_intensity_{size}px", totals["intensity"], "FLOP/byte", "weights + activations read/written once")
    layers = count_flops(cnn, (1, 3, MAIN_SIZE, MAIN_SIZE))
    layers.to_csv(RESULTS / "cnn_layer_flops.csv", index=False)
    totals = summarise_flops(layers)
    add("cnn", "activation_memory", layers["output_bytes"].sum() / 1e6, "MB", "sum of conv/linear outputs, one image")
    gflops_per_s = totals["flops"] / single["median"] / 1e9
    add("cnn", "achieved_throughput_batch1", gflops_per_s, "GFLOP/s")
    add("cnn", "achieved_throughput_batched", totals["flops"] * len(images) / batch["median"] / 1e9, "GFLOP/s")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "profiling.csv", index=False)
    log("profiling:\n" + out.to_string(index=False))


def stage_report(ctx):
    """Build the tables and figures (scripts/make_report.py)."""
    import make_report  # scripts/ is on sys.path when this file is run as a script

    make_report.main()


def main():
    """Parse arguments and run the selected stages in order."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    sys.path.insert(0, str(Path(__file__).parent))
    ctx = Context(args.seed)
    for stage in STAGES:
        if stage in args.stages:
            log(f"=== {stage} ===")
            t0 = time.perf_counter()
            globals()[f"stage_{stage}"](ctx)
            log(f"{stage} done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
