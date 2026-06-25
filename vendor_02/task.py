# PORTED from 02-memristive-crossbar/task.py @95868e7 — do not edit to diverge; re-vendor on upstream change.
# VENDORED from F:\science-projects\self-assembly\stage1_network.py (read-only upstream), 2026-06-18.
# Kept self-contained for parallel 02-05 execution. Do not edit to diverge silently;
# re-vendor if upstream changes. See docs/superpowers/specs/2026-06-18-memristive-crossbar-design.md.
"""Parity-3 task data and the two-sided separation / SNR verdict metrics."""
import numpy as np


def parity3_data():
    """8 parity-3 patterns in +/-1 encoding; label y = x1*x2*x3."""
    X = np.array([[1 - 2 * ((i >> b) & 1) for b in range(3)] for i in range(8)], float)
    y = np.prod(X, axis=1)
    return X, y


def margin(out, y):
    """Signed one-polarity class separation: min over y=+1 of out, minus max over y=-1 of out."""
    pos = out[np.asarray(y) > 0]
    neg = out[np.asarray(y) < 0]
    return float(np.min(pos) - np.max(neg))


def separation(out, y):
    """TWO-SIDED separability: > 0 iff a threshold separates the classes in EITHER polarity,
    <= 0 iff the classes OVERLAP. = max( min(+)-max(-) , min(-)-max(+) )."""
    out = np.asarray(out, float)
    yy = np.asarray(y)
    pos, neg = out[yy > 0], out[yy < 0]
    return float(max(np.min(pos) - np.max(neg), np.min(neg) - np.max(pos)))


def verdict(separation, floor, k=3.0):
    """Budget verdict from the two-sided separation vs the readout-noise floor.
    separation <= 0 (overlap) => FORECLOSE. Else SNR = separation/floor:
    PASS (>= k), AMBIGUOUS (1 <= SNR < k), FORECLOSE (< 1)."""
    if separation <= 0.0:
        return {"status": "FORECLOSE", "snr": 0.0, "separation": float(separation), "floor": float(floor)}
    snr = separation / floor if floor > 0 else float("inf")
    status = "PASS" if snr >= k else ("FORECLOSE" if snr < 1.0 else "AMBIGUOUS")
    return {"status": status, "snr": float(snr), "separation": float(separation), "floor": float(floor)}
