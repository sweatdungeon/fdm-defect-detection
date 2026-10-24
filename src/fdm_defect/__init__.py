"""Under-extrusion detection for FDM 3D printing from nozzle-camera images.

Submodules
----------
config
    Default paths, image size and random seed.
data
    Loading and validating the metadata CSV and images, and the print-level
    train/validation/test split.
synthetic
    Synthetic bead-stripe images so tests can run without the real dataset.
"""

__version__ = "0.1.0"
