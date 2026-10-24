"""Tests for fdm_defect.profiling: FLOP counter against hand counts, timing."""

import pytest
from torch import nn

from fdm_defect.cnn import build_model
from fdm_defect.profiling import (
    count_flops,
    count_parameters,
    model_size_bytes,
    profile_function,
    summarise_flops,
    time_function,
)


def tiny_net():
    # Conv2d(3 -> 8, 3x3, pad 1) on 10x10, then a depthwise 3x3 conv, then Linear(800 -> 5).
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.Conv2d(8, 8, 3, padding=1, groups=8),
        nn.Flatten(),
        nn.Linear(800, 5),
    )


def test_flop_counter_matches_hand_count():
    layers = count_flops(tiny_net(), (1, 3, 10, 10)).set_index("layer")
    # Conv: 8 x 10 x 10 outputs x (3 x 3 x 3) MACs each = 21,600 MACs.
    assert layers.loc["0", "macs"] == 21_600
    # Depthwise conv: 800 outputs x (8 / 8) x 3 x 3 = 7,200 MACs.
    assert layers.loc["1", "macs"] == 7_200
    # Linear: 5 outputs x 800 inputs = 4,000 MACs.
    assert layers.loc["3", "macs"] == 4_000
    total = summarise_flops(layers)
    assert total["flops"] == 2 * (21_600 + 7_200 + 4_000)


def test_flop_counter_is_per_image():
    single = summarise_flops(count_flops(tiny_net(), (1, 3, 10, 10)))["flops"]
    batch = summarise_flops(count_flops(tiny_net(), (4, 3, 10, 10)))["flops"]
    assert single == batch


def test_linear_arithmetic_intensity_hand_count():
    # Linear(800 -> 5) without bias: 4,000 MACs = 8,000 FLOPs; bytes =
    # 4 x (4,000 weights + 800 inputs + 5 outputs) = 19,220 -> 0.416 FLOP/byte.
    net = nn.Sequential(nn.Linear(800, 5, bias=False))
    row = count_flops(net, (1, 800)).iloc[0]
    assert row["intensity"] == pytest.approx(8_000 / 19_220)


def test_mobilenet_v3_small_matches_published_macs():
    # torchvision reports ~0.06 GMACs for MobileNetV3-Small at 224 px; conv
    # and linear layers account for nearly all of it.
    macs = summarise_flops(count_flops(build_model(pretrained=False), (1, 3, 224, 224)))["macs"]
    assert 50e6 < macs < 65e6


def test_count_flops_restores_training_mode():
    net = tiny_net().train()
    count_flops(net, (1, 3, 10, 10))
    assert net.training


def test_time_function_counts_calls():
    calls = []
    stats = time_function(lambda: calls.append(1), repeats=5, warmup=2)
    assert len(calls) == 7
    assert stats["repeats"] == 5 and stats["min"] <= stats["median"]


def test_time_function_invalid():
    with pytest.raises(ValueError):
        time_function(lambda: None, repeats=0)


def test_profile_function_reports_called_function():
    def work():
        return sum(i * i for i in range(1000))

    assert "work" in profile_function(work)


def test_model_size_and_parameters():
    net = nn.Linear(10, 2)  # 20 weights + 2 biases
    assert count_parameters(net) == 22
    assert model_size_bytes(net) == 22 * 4
    assert model_size_bytes({"a": 1}) > 0  # pickle path for non-torch objects
