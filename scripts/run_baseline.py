"""Train the hand-crafted-feature baselines and score them on the validation split.

The test split is deliberately not touched here: thresholds and settings are
chosen on validation, and test is evaluated once at the end of the project.

Metrics are per image at a 0.5 threshold. The validation split holds only 8
prints, so treat these numbers as a sanity check, not a final result.

Usage:
  python scripts/run_baseline.py [--size 128] [--seed 42] [--out results/baseline_val.csv]
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from fdm_defect import config
from fdm_defect.baseline import MODEL_NAMES, compute_features, fit_baseline, predict_proba
from fdm_defect.data import LABEL_COLUMN, load_metadata, split_by_print


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=config.DEFAULT_IMAGE_SIZE, help="image side in px")
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="split and model seed")
    ap.add_argument("--out", default="results/baseline_val.csv", help="output CSV")
    args = ap.parse_args()

    df = load_metadata(config.SUBSET_CSV, config.IMAGE_DIR)
    split = split_by_print(df, seed=args.seed)

    t0 = time.perf_counter()
    x_train = compute_features(split.train, size=args.size)
    x_val = compute_features(split.val, size=args.size)
    n_images = len(x_train) + len(x_val)
    secs = time.perf_counter() - t0
    print(f"features: {n_images} images in {secs:.1f} s ({1000 * secs / n_images:.1f} ms/image)")
    y_train, y_val = split.train[LABEL_COLUMN].to_numpy(), split.val[LABEL_COLUMN].to_numpy()

    rows = []
    for name in MODEL_NAMES:
        t0 = time.perf_counter()
        model = fit_baseline(name, x_train, y_train, seed=args.seed)
        fit_secs = time.perf_counter() - t0
        proba = predict_proba(model, x_val)
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "model": name,
            "size": args.size,
            "seed": args.seed,
            "recall": recall_score(y_val, pred),
            "precision": precision_score(y_val, pred, zero_division=0),
            "f1": f1_score(y_val, pred),
            # Specificity is recall of the good class.
            "specificity": recall_score(y_val, pred, pos_label=0),
            "roc_auc": roc_auc_score(y_val, proba),
            "fit_seconds": fit_secs,
        })

    results = pd.DataFrame(rows)
    print(results.to_string(index=False, float_format="%.3f"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
