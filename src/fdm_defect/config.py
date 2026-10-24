"""Project-wide defaults: data locations, image size and random seed.

Paths are resolved relative to the repository root, which assumes the package
is installed in editable mode (``pip install -e .``) from the src layout.
"""

from pathlib import Path

#: Repository root (``src/fdm_defect/config.py`` -> two levels up from ``src``).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Folder holding the metadata CSVs and the (git-ignored) raw images.
DATA_DIR = PROJECT_ROOT / "data"

#: Subset metadata: up to 120 evenly spaced frames from each of the 52 prints.
SUBSET_CSV = DATA_DIR / "train_subset.csv"

#: Images extracted by ``scripts/extract_subset.py``; ``img_path`` is relative to this.
IMAGE_DIR = DATA_DIR / "raw" / "images"

#: Side length in pixels that images are resized to (square).
DEFAULT_IMAGE_SIZE = 128

#: Seed used for splits and any other randomness unless one is given explicitly.
DEFAULT_SEED = 42
