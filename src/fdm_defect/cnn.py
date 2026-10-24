"""Pretrained CNN classifier (MobileNetV3-Small, or ResNet-18) for under-extrusion.

The ImageNet classifier's last layer is replaced by a single-logit layer, and
``sigmoid(logit)`` is the probability of under-extrusion. By default only the
head is trained and the backbone is frozen:

- MobileNetV3-Small: the head is ``model.classifier`` (pretrained
  ``Linear(576, 1024)`` + Hardswish + Dropout + new ``Linear(1024, 1)``);
  the backbone is ``model.features``.
- ResNet-18: the head is the new ``model.fc = Linear(512, 1)``.

Two details matter for correctness:

1. **Frozen BatchNorm must stay in eval mode.** ``model.train()`` puts every
   BatchNorm layer in training mode, where it normalises with the current
   batch's statistics and overwrites the ImageNet running statistics, even if
   its weights are frozen. :func:`set_train_mode` puts frozen BatchNorm layers
   back into eval mode.
2. **Always start from pretrained weights.** MobileNetV3's BatchNorm momentum
   is 0.01, so running statistics move only 1% per batch. From random weights
   they lag far behind the changing activations, and the eval-mode network
   collapses to near-constant outputs.

Training uses ``BCEWithLogitsLoss`` with ``pos_weight = n_good / n_defective``,
AdamW, random horizontal flips and brightness changes, and early stopping on
validation loss that restores the best weights.
"""

from __future__ import annotations

import copy
import random
import time
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

from . import config
from .data import LABEL_COLUMN, load_image

#: Architectures accepted by :func:`build_model`.
ARCHS = ("mobilenet_v3_small", "resnet18")

#: ImageNet channel means and standard deviations the pretrained weights expect.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

#: Brightness is multiplied by a factor drawn uniformly from this range.
BRIGHTNESS_RANGE = (0.8, 1.2)


class TrainResult(NamedTuple):
    """Output of :func:`train_model`."""

    model: nn.Module
    history: list  # one dict per epoch: epoch, train_loss, val_loss, seconds
    best_epoch: int


def seed_everything(seed=config.DEFAULT_SEED):
    """Seed Python, NumPy and PyTorch so a run can be repeated exactly.

    Call it *before* :func:`build_model`, since the new head's initial
    weights are random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_device():
    """Return ``"cuda"`` or ``"mps"`` if available, else ``"cpu"``."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(arch="mobilenet_v3_small", pretrained=True, freeze_backbone=True):
    """Build a CNN with a single-logit head.

    Parameters
    ----------
    arch : {"mobilenet_v3_small", "resnet18"}, default "mobilenet_v3_small"
        Backbone architecture.
    pretrained : bool, default True
        Load ImageNet weights (downloaded once, then cached by torchvision).
        Only use False for tests of the mechanics; see the module docstring.
    freeze_backbone : bool, default True
        If True, only the head's parameters have ``requires_grad=True``.
        If False, the whole network is trainable (full fine-tune).

    Returns
    -------
    torch.nn.Module
        Model mapping ``(B, 3, H, W)`` normalised images to ``(B, 1)`` logits.

    Raises
    ------
    ValueError
        If ``arch`` is not in :data:`ARCHS`.
    """
    if arch == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        head = model.classifier
    elif arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        head = model.fc
    else:
        raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")

    if freeze_backbone:
        head_params = {id(p) for p in head.parameters()}
        for p in model.parameters():
            p.requires_grad = id(p) in head_params
    return model


def set_train_mode(model):
    """Put ``model`` in training mode, but keep frozen BatchNorm layers in eval mode.

    A BatchNorm layer counts as frozen when none of its parameters require
    gradients. In eval mode it keeps normalising with its pretrained running
    statistics instead of batch statistics, and those statistics are not
    updated.

    Parameters
    ----------
    model : torch.nn.Module

    Returns
    -------
    torch.nn.Module
        The same model, for chaining.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            params = list(module.parameters())
            if params and not any(p.requires_grad for p in params):
                module.eval()
    return model


def normalise(x):
    """Apply ImageNet normalisation to a ``(3, H, W)`` or ``(B, 3, H, W)`` tensor in [0, 1]."""
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def augment_image(x, generator=None):
    """Random horizontal flip and brightness change.

    Parameters
    ----------
    x : torch.Tensor
        Image of shape ``(3, H, W)`` with values in [0, 1].
    generator : torch.Generator, optional
        Source of randomness, for reproducibility.

    Returns
    -------
    torch.Tensor
        Augmented image, same shape, values clipped to [0, 1].
    """
    if torch.rand(1, generator=generator).item() < 0.5:
        x = x.flip(-1)
    lo, hi = BRIGHTNESS_RANGE
    factor = lo + (hi - lo) * torch.rand(1, generator=generator).item()
    return (x * factor).clamp(0.0, 1.0)


class FrameDataset(Dataset):
    """Images and labels from a metadata table, as normalised tensors.

    Resized images are cached in memory as uint8 the first time they are read
    (about 48 KB each at 128 px), so JPEG decoding happens once, not every
    epoch. The cache and the augmentation generator live in this object, so
    use it with ``num_workers=0`` (the default in :func:`train_model`).

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata with ``path`` and ``has_under_extrusion`` columns, e.g. from
        :func:`fdm_defect.data.load_metadata` with ``image_dir`` set.
    size : int, default ``config.DEFAULT_IMAGE_SIZE``
        Image side length.
    augment : bool, default False
        Apply :func:`augment_image` to each item.
    seed : int, default ``config.DEFAULT_SEED``
        Seed for the augmentation generator.
    """

    def __init__(self, df, size=config.DEFAULT_IMAGE_SIZE, augment=False, seed=config.DEFAULT_SEED):
        missing = [c for c in ("path", LABEL_COLUMN) if c not in df.columns]
        if missing:
            raise ValueError(f"df is missing columns {missing}; load it with image_dir set")
        if len(df) == 0:
            raise ValueError("df has no rows")
        self.paths = df["path"].tolist()
        self.labels = torch.tensor(df[LABEL_COLUMN].to_numpy(), dtype=torch.float32)
        self.size = size
        self.augment = augment
        self.generator = torch.Generator().manual_seed(seed)
        self._cache = {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if i not in self._cache:
            # load_image returns float32 = uint8 / 255, so this round trip is exact.
            img = np.round(load_image(self.paths[i], self.size) * 255).astype(np.uint8)
            self._cache[i] = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        x = self._cache[i].float() / 255.0
        if self.augment:
            x = augment_image(x, self.generator)
        return normalise(x), self.labels[i]


def compute_pos_weight(labels):
    """``pos_weight`` for ``BCEWithLogitsLoss``: number of negatives / positives.

    This makes the defective class carry the same total weight in the loss as
    the good class.

    Parameters
    ----------
    labels : array_like
        0/1 labels.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If either class is absent.
    """
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"need both classes, got {n_neg} negatives and {n_pos} positives")
    return n_neg / n_pos


@torch.no_grad()
def _mean_loss(model, loader, criterion, device):
    """Mean loss over a loader with the model in eval mode."""
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += criterion(model(x).squeeze(1), y).item() * len(y)
        n += len(y)
    return total / n


def train_model(
    model,
    train_df,
    val_df,
    size=config.DEFAULT_IMAGE_SIZE,
    epochs=30,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    patience=5,
    augment=True,
    seed=config.DEFAULT_SEED,
    device=None,
    verbose=False,
):
    """Train ``model`` with early stopping on validation loss.

    After each epoch the validation loss is computed; training stops once it
    has not improved for ``patience`` epochs, and the weights from the best
    epoch are loaded back into ``model``.

    Parameters
    ----------
    model : torch.nn.Module
        Model from :func:`build_model`. Modified in place.
    train_df, val_df : pandas.DataFrame
        Metadata for training and validation (see :class:`FrameDataset`).
        ``train_df`` must contain both classes.
    size : int, default ``config.DEFAULT_IMAGE_SIZE``
        Image side length.
    epochs : int, default 30
        Maximum number of epochs.
    batch_size : int, default 64
    lr : float, default 1e-3
        AdamW learning rate. Use a smaller value (e.g. 1e-4) for a full
        fine-tune.
    weight_decay : float, default 1e-4
        AdamW weight decay.
    patience : int, default 5
        Epochs without validation-loss improvement before stopping.
    augment : bool, default True
        Random flips and brightness changes on the training images.
    seed : int, default ``config.DEFAULT_SEED``
        Seed for batch shuffling and augmentation.
    device : str, optional
        Torch device; defaults to :func:`default_device`.
    verbose : bool, default False
        Print one line per epoch.

    Returns
    -------
    TrainResult
        The model (best weights, eval mode), the per-epoch history and the
        best epoch (1-based).

    Raises
    ------
    ValueError
        If a hyperparameter is out of range, ``train_df`` lacks a class, or
        the model has no trainable parameters.
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be at least 1")
    if lr < 0 or weight_decay < 0 or patience < 0:
        raise ValueError("lr, weight_decay and patience must be non-negative")
    device = device or default_device()
    pos_weight = compute_pos_weight(train_df[LABEL_COLUMN])
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")

    train_loader = DataLoader(
        FrameDataset(train_df, size, augment=augment, seed=seed),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(FrameDataset(val_df, size), batch_size=batch_size)

    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    history = []
    best_loss, best_epoch, epochs_without_improvement = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        set_train_mode(model)
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(y)
            n += len(y)
        val_loss = _mean_loss(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": total / n,
            "val_loss": val_loss,
            "seconds": time.perf_counter() - t0,
        })
        if verbose:
            h = history[-1]
            print(f"epoch {epoch:3d}  train {h['train_loss']:.4f}  val {val_loss:.4f}  ({h['seconds']:.1f} s)")

        if val_loss < best_loss:
            best_loss, best_epoch, epochs_without_improvement = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model, history, best_epoch)


@torch.no_grad()
def predict_proba(model, df, size=config.DEFAULT_IMAGE_SIZE, batch_size=64, device=None):
    """Probability of under-extrusion for each image in ``df``.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    df : pandas.DataFrame
        Metadata (see :class:`FrameDataset`).
    size : int, default ``config.DEFAULT_IMAGE_SIZE``
        Must match the size used in training.
    batch_size : int, default 64
    device : str, optional
        Torch device; defaults to :func:`default_device`.

    Returns
    -------
    numpy.ndarray
        Probabilities in [0, 1], shape ``(len(df),)``, in ``df`` order.
    """
    device = device or default_device()
    model.to(device).eval()
    loader = DataLoader(FrameDataset(df, size), batch_size=batch_size)
    probs = [torch.sigmoid(model(x.to(device)).squeeze(1)).cpu() for x, _ in loader]
    return torch.cat(probs).numpy()
