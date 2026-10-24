"""Train the pretrained CNN and score it on the validation split.

The test split is deliberately not touched here: settings are chosen on
validation, and test is evaluated once at the end of the project.

Metrics are per image at a 0.5 threshold. The validation split holds only 8
prints, so treat these numbers as a sanity check, not a final result.

Usage:
  python scripts/train_cnn.py [--arch mobilenet_v3_small] [--size 128] [--full-finetune]
      [--epochs 30] [--lr 1e-3] [--batch-size 64] [--patience 5] [--seed 42]
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from fdm_defect import config
from fdm_defect.cnn import ARCHS, build_model, predict_proba, seed_everything, train_model
from fdm_defect.data import LABEL_COLUMN, load_metadata, split_by_print


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=ARCHS, default="mobilenet_v3_small")
    ap.add_argument("--size", type=int, default=config.DEFAULT_IMAGE_SIZE, help="image side in px")
    ap.add_argument("--full-finetune", action="store_true", help="train the backbone too")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="split and training seed")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    df = load_metadata(config.SUBSET_CSV, config.IMAGE_DIR)
    split = split_by_print(df, seed=args.seed)

    seed_everything(args.seed)
    model = build_model(args.arch, pretrained=True, freeze_backbone=not args.full_finetune)
    t0 = time.perf_counter()
    result = train_model(
        model, split.train, split.val, size=args.size, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, patience=args.patience,
        seed=args.seed, verbose=True,
    )
    train_secs = time.perf_counter() - t0

    y_val = split.val[LABEL_COLUMN].to_numpy()
    proba = predict_proba(result.model, split.val, size=args.size)
    pred = (proba >= 0.5).astype(int)
    mode = "full" if args.full_finetune else "head"
    tag = f"{args.arch}_{mode}_{args.size}"
    metrics = pd.DataFrame([{
        "model": args.arch,
        "training": mode,
        "size": args.size,
        "seed": args.seed,
        "best_epoch": result.best_epoch,
        "epochs_run": len(result.history),
        "recall": recall_score(y_val, pred),
        "precision": precision_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred),
        # Specificity is recall of the good class.
        "specificity": recall_score(y_val, pred, pos_label=0),
        "roc_auc": roc_auc_score(y_val, proba),
        "train_seconds": train_secs,
    }])
    print(metrics.to_string(index=False, float_format="%.3f"))

    results_dir, models_dir = Path(args.results_dir), Path(args.models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / f"cnn_val_{tag}.csv", index=False)
    pd.DataFrame(result.history).to_csv(results_dir / f"cnn_history_{tag}.csv", index=False)
    torch.save(result.model.state_dict(), models_dir / f"cnn_{tag}.pt")
    print(f"saved results to {results_dir}/ and weights to {models_dir}/cnn_{tag}.pt")


if __name__ == "__main__":
    main()
