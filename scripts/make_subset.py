"""Rebuild data/train_subset.csv from the competition's train.csv.

Takes up to 120 evenly spaced frames from every print (see
fdm_defect.data.make_subset). With --check, compares the result with the
existing subset file instead of overwriting it, and exits with an error if
they differ.

Usage:
  python scripts/make_subset.py --train path/to/train.csv [--out data/train_subset.csv]
  python scripts/make_subset.py --train path/to/train.csv.zip --check

--train may be the CSV itself or the Kaggle zip containing it.
"""

import argparse
import filecmp
import sys
import tempfile
from pathlib import Path

import pandas as pd

from fdm_defect import config
from fdm_defect.data import make_subset


def main() -> None:
    """Parse arguments, build the subset, then write it or compare it with the existing file."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="train.csv or a zip containing only it")
    ap.add_argument("--out", default=str(config.SUBSET_CSV), help="output CSV")
    ap.add_argument("--frames-per-print", type=int, default=120)
    ap.add_argument("--check", action="store_true",
                    help="compare with --out instead of writing it")
    args = ap.parse_args()

    train = pd.read_csv(args.train)  # pandas reads a single-file zip directly
    subset = make_subset(train, frames_per_print=args.frames_per_print)
    print(f"{len(subset)} frames from {subset['print_id'].nunique()} prints "
          f"(train.csv: {len(train)} frames)")

    if not args.check:
        subset.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "subset.csv"
        subset.to_csv(fresh, index=False)
        if filecmp.cmp(fresh, args.out, shallow=False):
            print(f"OK: identical to {args.out}")
        else:
            print(f"MISMATCH: rebuilt subset differs from {args.out}")
            sys.exit(1)


if __name__ == "__main__":
    main()
