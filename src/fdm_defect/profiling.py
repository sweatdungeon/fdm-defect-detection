"""Profiling: wall-clock timing, cProfile, FLOP counting and arithmetic intensity.

FLOPs are counted with forward hooks on ``Conv2d`` and ``Linear`` layers, which
hold almost all of a CNN's arithmetic. One multiply-accumulate (MAC) counts as
2 FLOPs. Bias additions, activations, BatchNorm (foldable into the preceding
convolution at inference) and pooling are not counted, so totals are a slight
underestimate.

Arithmetic intensity is FLOPs per byte of memory traffic. Traffic is estimated
as each layer reading its weights and input once and writing its output once,
all float32. Real hardware may move more (cache misses) or less (fusion), so
treat it as an order-of-magnitude indicator of whether a layer is compute- or
memory-bound.
"""

from __future__ import annotations

import cProfile
import io
import pickle
import pstats
import statistics
import time

import pandas as pd
import torch
from torch import nn

#: Bytes per float32 value.
BYTES_PER_VALUE = 4


def count_flops(model, input_shape=(1, 3, 128, 128)):
    """Per-layer FLOPs and memory traffic for one forward pass.

    For a ``Conv2d`` with output ``(N, C_out, H_out, W_out)``, kernel
    ``k_h x k_w`` and ``groups`` groups, each output value takes
    ``(C_in / groups) k_h k_w`` MACs. For a ``Linear`` layer each output value
    takes ``in_features`` MACs.

    Parameters
    ----------
    model : torch.nn.Module
        Model to profile; it is run once in eval mode with zeros as input.
    input_shape : tuple of int, default (1, 3, 128, 128)
        Input tensor shape. Results are divided by the batch size, so they are
        per image.

    Returns
    -------
    pandas.DataFrame
        One row per counted layer: ``layer``, ``type``, ``macs``, ``flops``,
        ``params``, ``weight_bytes``, ``input_bytes``, ``output_bytes``,
        ``intensity`` (flops / total bytes).
    """
    batch = input_shape[0]
    rows = []

    def hook(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            macs = output.numel() * (module.in_channels // module.groups) * kh * kw
        else:
            macs = output.numel() * module.in_features
        weight_values = module.weight.numel()
        rows.append({
            "layer": names[module],
            "type": type(module).__name__,
            "macs": macs / batch,
            "params": sum(p.numel() for p in module.parameters()),
            "weight_bytes": weight_values * BYTES_PER_VALUE,
            "input_bytes": inputs[0].numel() / batch * BYTES_PER_VALUE,
            "output_bytes": output.numel() / batch * BYTES_PER_VALUE,
        })

    names = {m: n for n, m in model.named_modules()}
    handles = [
        m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(input_shape))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    df = pd.DataFrame(rows)
    df["flops"] = 2 * df["macs"]
    total_bytes = df["weight_bytes"] + df["input_bytes"] + df["output_bytes"]
    df["intensity"] = df["flops"] / total_bytes
    return df[["layer", "type", "macs", "flops", "params", "weight_bytes", "input_bytes", "output_bytes", "intensity"]]


def summarise_flops(layers):
    """Totals from :func:`count_flops`.

    Returns
    -------
    dict
        ``flops``, ``macs``, ``bytes`` (estimated traffic) and
        ``intensity`` (total flops / total bytes).
    """
    total_bytes = float((layers["weight_bytes"] + layers["input_bytes"] + layers["output_bytes"]).sum())
    flops = float(layers["flops"].sum())
    return {"flops": flops, "macs": float(layers["macs"].sum()), "bytes": total_bytes, "intensity": flops / total_bytes}


def time_function(fn, repeats=20, warmup=3):
    """Wall-clock timing of ``fn()`` with ``time.perf_counter``.

    Parameters
    ----------
    fn : callable
        Function with no arguments.
    repeats : int, default 20
        Timed calls.
    warmup : int, default 3
        Untimed calls first (caches, lazy initialisation, thread pools).

    Returns
    -------
    dict
        ``median``, ``mean``, ``std``, ``min`` in seconds and ``repeats``.
    """
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup >= 0")
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "median": statistics.median(times),
        "mean": statistics.fmean(times),
        "std": statistics.stdev(times) if repeats > 1 else 0.0,
        "min": min(times),
        "repeats": repeats,
    }


def profile_function(fn, top=20, sort="cumulative"):
    """Run ``fn()`` under cProfile and return the top entries as text.

    Parameters
    ----------
    fn : callable
        Function with no arguments.
    top : int, default 20
        Number of rows to keep.
    sort : str, default "cumulative"
        ``pstats`` sort key.

    Returns
    -------
    str
        Formatted ``pstats`` report.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        fn()
    finally:
        profiler.disable()
    out = io.StringIO()
    pstats.Stats(profiler, stream=out).strip_dirs().sort_stats(sort).print_stats(top)
    return out.getvalue()


def model_size_bytes(model):
    """Storage size of a model.

    For a ``torch.nn.Module``: bytes of all parameters and buffers. Otherwise
    (e.g. a scikit-learn estimator): length of its pickle.
    """
    if isinstance(model, nn.Module):
        tensors = list(model.parameters()) + list(model.buffers())
        return sum(t.numel() * t.element_size() for t in tensors)
    return len(pickle.dumps(model))


def count_parameters(model, trainable_only=False):
    """Number of parameters of a torch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
