"""Tests for fdm_defect.cnn.

Most tests use random weights (``pretrained=False``) and 32 px images so they
run fast and offline; they check the training mechanics, not accuracy. The
one accuracy test needs the cached ImageNet weights and is skipped otherwise.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from fdm_defect.cnn import (
    IMAGENET_MEAN,
    FrameDataset,
    augment_image,
    build_model,
    compute_pos_weight,
    normalise,
    predict_proba,
    predict_proba_images,
    seed_everything,
    set_train_mode,
    train_model,
)
from fdm_defect.data import load_images, load_metadata, split_by_print

SIZE = 32


@pytest.fixture(scope="module")
def split(synthetic_dir):
    df = load_metadata(synthetic_dir / "metadata.csv", synthetic_dir / "images")
    return split_by_print(df, seed=0)


def batchnorms(module):
    return [m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]


# --------------------------------------------------------------------------
# Model construction and freezing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["mobilenet_v3_small", "resnet18"])
def test_model_outputs_one_logit(arch):
    model = build_model(arch, pretrained=False).eval()
    assert model(torch.zeros(2, 3, SIZE, SIZE)).shape == (2, 1)


def test_unknown_arch():
    with pytest.raises(ValueError, match="unknown arch"):
        build_model("vgg16", pretrained=False)


def test_frozen_backbone_trains_head_only():
    model = build_model(pretrained=False, freeze_backbone=True)
    assert not any(p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_resnet_frozen_backbone_trains_fc_only():
    model = build_model("resnet18", pretrained=False)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["fc.weight", "fc.bias"]


def test_full_finetune_trains_everything():
    model = build_model(pretrained=False, freeze_backbone=False)
    assert all(p.requires_grad for p in model.parameters())


def test_set_train_mode_keeps_frozen_batchnorm_in_eval():
    model = set_train_mode(build_model(pretrained=False, freeze_backbone=True))
    assert all(not bn.training for bn in batchnorms(model.features))
    assert model.classifier.training  # dropout in the head is active


def test_set_train_mode_full_finetune_batchnorm_trains():
    model = set_train_mode(build_model(pretrained=False, freeze_backbone=False))
    assert all(bn.training for bn in batchnorms(model))


def test_frozen_batchnorm_statistics_unchanged_by_training_step():
    # The bug this guards against: model.train() alone lets frozen BatchNorm
    # layers overwrite their running statistics with batch statistics.
    model = build_model(pretrained=False, freeze_backbone=True)
    before = [bn.running_mean.clone() for bn in batchnorms(model)]
    set_train_mode(model)
    model(torch.rand(4, 3, SIZE, SIZE))
    after = [bn.running_mean for bn in batchnorms(model)]
    assert all(torch.equal(a, b) for a, b in zip(before, after))


# --------------------------------------------------------------------------
# Preprocessing, dataset and loss weighting
# --------------------------------------------------------------------------


def test_normalise_maps_imagenet_mean_to_zero():
    x = IMAGENET_MEAN.expand(3, 4, 4)
    torch.testing.assert_close(normalise(x), torch.zeros(3, 4, 4))


def test_augment_keeps_shape_and_range():
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(3, 8, 8)
    for _ in range(20):
        y = augment_image(x, gen)
        assert y.shape == x.shape
        assert y.min() >= 0 and y.max() <= 1


def test_augment_flips_about_half_the_time():
    gen = torch.Generator().manual_seed(0)
    # Left half 0, right half 1: a flip swaps them, brightness keeps 0 at 0.
    x = torch.zeros(3, 4, 4)
    x[..., 2:] = 1.0
    flipped = sum(augment_image(x, gen)[0, 0, 0].item() > 0 for _ in range(400))
    assert 150 < flipped < 250


def test_dataset_item(split):
    ds = FrameDataset(split.train, size=SIZE)
    x, y = ds[0]
    assert x.shape == (3, SIZE, SIZE) and x.dtype == torch.float32
    assert y.item() in (0.0, 1.0)
    assert len(ds) == len(split.train)


def test_dataset_cache_matches_fresh_load(split):
    ds = FrameDataset(split.train, size=SIZE)
    first = ds[0][0]
    torch.testing.assert_close(ds[0][0], first)  # second read comes from the cache
    torch.testing.assert_close(FrameDataset(split.train, size=SIZE)[0][0], first)


def test_dataset_requires_path_column(synthetic_df):
    with pytest.raises(ValueError, match="path"):
        FrameDataset(synthetic_df)


@pytest.mark.parametrize("labels, expected", [([0, 0, 0, 1], 3.0), ([0, 1, 1, 1], 1 / 3), ([0, 1], 1.0)])
def test_compute_pos_weight(labels, expected):
    assert compute_pos_weight(labels) == pytest.approx(expected)


def test_compute_pos_weight_one_class():
    with pytest.raises(ValueError, match="both classes"):
        compute_pos_weight([1, 1, 1])


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------


def quick_train(split, seed=0, **kwargs):
    seed_everything(seed)
    model = build_model(pretrained=False)
    options = dict(size=SIZE, epochs=2, batch_size=16, seed=seed, device="cpu")
    options.update(kwargs)
    return train_model(model, split.train, split.val, **options)


def test_train_records_history(split):
    result = quick_train(split, epochs=3, patience=10)
    assert [h["epoch"] for h in result.history] == [1, 2, 3]
    assert all(np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"]) for h in result.history)
    assert 1 <= result.best_epoch <= 3
    assert not result.model.training


def test_early_stopping_when_val_loss_does_not_improve(split):
    # lr = 0 leaves the weights unchanged, so the validation loss is identical
    # every epoch: epoch 1 is best and training stops after `patience` more.
    result = quick_train(split, epochs=20, lr=0.0, weight_decay=0.0, patience=2)
    assert len(result.history) == 3
    assert result.best_epoch == 1


def test_best_weights_are_restored(split):
    result = quick_train(split, epochs=4, lr=1e-2, patience=10)
    best = min(h["val_loss"] for h in result.history)
    assert result.history[result.best_epoch - 1]["val_loss"] == best
    # Recompute the validation loss with the restored weights.
    pos_weight = compute_pos_weight(split.train["has_under_extrusion"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    x, y = zip(*FrameDataset(split.val, size=SIZE))
    with torch.no_grad():
        loss = criterion(result.model(torch.stack(x)).squeeze(1), torch.stack(y)).item()
    assert loss == pytest.approx(best, rel=1e-5)


def test_training_is_deterministic(split):
    a = quick_train(split, seed=3).model.state_dict()
    b = quick_train(split, seed=3).model.state_dict()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_training_updates_head_not_backbone(split):
    seed_everything(0)
    model = build_model(pretrained=False)
    backbone = {k: v.clone() for k, v in model.features.state_dict().items()}
    head = {k: v.clone() for k, v in model.classifier.state_dict().items()}
    train_model(model, split.train, split.val, size=SIZE, epochs=1, batch_size=16, device="cpu")
    assert all(torch.equal(v, model.features.state_dict()[k]) for k, v in backbone.items())
    assert any(not torch.equal(v, model.classifier.state_dict()[k]) for k, v in head.items())


@pytest.mark.parametrize(
    "kwargs", [{"epochs": 0}, {"batch_size": 0}, {"lr": -1.0}, {"patience": -1}, {"weight_decay": -0.1}]
)
def test_train_invalid_arguments(split, kwargs):
    with pytest.raises(ValueError):
        train_model(build_model(pretrained=False), split.train, split.val, size=SIZE, **kwargs)


def test_train_single_class(split):
    good_only = split.train[split.train["has_under_extrusion"] == 0]
    with pytest.raises(ValueError, match="both classes"):
        train_model(build_model(pretrained=False), good_only, split.val, size=SIZE)


def test_train_no_trainable_parameters(split):
    model = build_model(pretrained=False)
    for p in model.parameters():
        p.requires_grad = False
    with pytest.raises(ValueError, match="no trainable"):
        train_model(model, split.train, split.val, size=SIZE)


def test_predict_proba(split):
    model = build_model(pretrained=False)
    proba = predict_proba(model, split.test, size=SIZE, device="cpu")
    assert proba.shape == (len(split.test),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predict_proba_images_matches_dataset_path(split):
    # In-memory prediction must give the same answer as loading from disk.
    model = build_model(pretrained=False)
    images = load_images(split.test["path"], size=SIZE)
    from_disk = predict_proba(model, split.test, size=SIZE, device="cpu")
    in_memory = predict_proba_images(model, images, device="cpu")
    np.testing.assert_allclose(in_memory, from_disk, rtol=1e-5, atol=1e-6)


def test_predict_proba_images_invalid():
    with pytest.raises(ValueError):
        predict_proba_images(build_model(pretrained=False), np.zeros((4, 4, 3)))


# --------------------------------------------------------------------------
# Accuracy with pretrained weights (needs the weights cached locally)
# --------------------------------------------------------------------------

WEIGHTS = Path(torch.hub.get_dir()) / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="MobileNetV3-Small ImageNet weights not cached")
def test_pretrained_head_learns_synthetic_task(split):
    seed_everything(0)
    model = build_model(pretrained=True)
    train_model(model, split.train, split.val, size=64, epochs=15, batch_size=16, lr=3e-3, patience=15, device="cpu")
    proba = predict_proba(model, split.test, size=64, device="cpu")
    accuracy = ((proba >= 0.5) == split.test["has_under_extrusion"].to_numpy()).mean()
    assert accuracy >= 0.8
