"""Post-hoc embedding compression schemes applied in NumPy/torch to stored embeddings.

All functions take X (n, d) float32 and return the *dequantized* float32 array,
so scoring is always a plain dot product with the (float32 or quantized) query.
"""
from __future__ import annotations
import numpy as np
import torch


def fp16(X):
    return X.astype(np.float16).astype(np.float32)


def fp8(X, fmt="e4m3"):
    dt = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
    return torch.from_numpy(X).to(dt).float().numpy()


def calib_ranges(C, pct=0.0):
    """Per-dimension (min, max) from calibration embeddings C (m, d).
    pct>0 clips to the [pct, 100-pct] percentiles (robust range, fewer wasted levels)."""
    if pct <= 0:
        return C.min(0), C.max(0)
    return np.percentile(C, pct, axis=0), np.percentile(C, 100 - pct, axis=0)


def uniform(X, ranges, bits):
    """Asymmetric per-dimension uniform quantization with 2**bits levels
    (sentence-transformers' int8 recipe generalised to any bit width)."""
    lo, hi = ranges
    levels = 2 ** bits - 1
    step = (hi - lo) / levels
    step = np.where(step == 0, 1e-12, step)
    q = np.clip(np.round((X - lo) / step), 0, levels)  # clipping = out-of-range calibration damage
    return (q * step + lo).astype(np.float32)


def clip_fraction(X, ranges):
    lo, hi = ranges
    return float(((X < lo) | (X > hi)).mean())


def binary(X, thresh=None):
    """Sign binarisation; thresh (d,) optional per-dimension threshold (e.g. calibration mean)."""
    t = 0.0 if thresh is None else thresh
    return np.where(X > t, 1.0, -1.0).astype(np.float32)


# ---- scheme registry -------------------------------------------------------
# name -> (function(X, calib) , needs_calibration)
SCHEMES = {
    "fp32": (lambda X, C: X, False),
    "fp16": (lambda X, C: fp16(X), False),
    "fp8_e4m3": (lambda X, C: fp8(X, "e4m3"), False),
    "int8": (lambda X, C: uniform(X, calib_ranges(C), 8), True),
    "int4": (lambda X, C: uniform(X, calib_ranges(C), 4), True),
    "int2": (lambda X, C: uniform(X, calib_ranges(C), 2), True),
    "int4_p1": (lambda X, C: uniform(X, calib_ranges(C, 1.0), 4), True),
    "int2_p1": (lambda X, C: uniform(X, calib_ranges(C, 1.0), 2), True),
    "int2_p5": (lambda X, C: uniform(X, calib_ranges(C, 5.0), 2), True),
    "binary": (lambda X, C: binary(X), False),
    "binary_cal": (lambda X, C: binary(X, C.mean(0)), True),
}
BITS = {"fp32": 32, "fp16": 16, "fp8_e4m3": 8, "int8": 8, "int4": 4, "int2": 2, "int4_p1": 4, "int2_p1": 2, "int2_p5": 2, "binary": 1, "binary_cal": 1}
