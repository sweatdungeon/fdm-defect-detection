"""Extract only the images listed in a subset CSV from the Kaggle competition zip.

The full download (early-detection-of-3d-printing-issues.zip) is ~12 GB with
100k+ images. This copies just the subset (~6k images) into data/raw/images/,
keeping the original folder structure, so the zip never has to be unpacked.

Usage:
  python scripts/extract_subset.py --zip path/to/early-detection-of-3d-printing-issues.zip \
      --subset data/train_subset.csv --out data/raw

Afterwards the zip can be deleted.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="Kaggle competition zip file")
    ap.add_argument("--subset", required=True, help="CSV with an img_path column")
    ap.add_argument("--out", default="data/raw", help="output folder")
    args = ap.parse_args()

    wanted = set(pd.read_csv(args.subset)["img_path"].astype(str))
    out_images = Path(args.out) / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as zf:
        names = zf.namelist()
        # Images may sit under a prefix inside the zip (e.g. "images/") or at the
        # top level (the separate images.zip download); strip any such prefix and
        # match on the img_path so the exact layout doesn't matter.
        found = {}
        for n in names:
            rel = n.split("images/", 1)[1] if "/images/" in f"/{n}" else n
            if rel in wanted:
                found[rel] = n
        missing = wanted - found.keys()
        print(f"{len(found)} of {len(wanted)} subset images found in the zip")
        if missing:
            print(f"WARNING: {len(missing)} not found, e.g. {sorted(missing)[:3]}")

        for i, (rel, member) in enumerate(sorted(found.items()), 1):
            target = out_images / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if i % 500 == 0:
                print(f"  extracted {i}/{len(found)}")

        # Also extract the CSVs at the top level of the zip, if present.
        for n in names:
            if n.endswith(".csv") and "/" not in n.strip("/"):
                zf.extract(n, args.out)
                print(f"extracted {n}")
    print(f"Done: images in {out_images}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
