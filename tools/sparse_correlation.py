"""Cross-correlation of binary probe signals using their phase-change runs."""

from __future__ import annotations

import numpy as np
from scipy import signal


def run_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Match ``scipy.signal.correlate(second, first, mode='full')`` after AWCC normalization.

    The overlap of two constant-valued runs is piecewise linear in lag. Each
    pair contributes four slope changes, followed by two cumulative sums to
    recover the full correlation vector. The number of run pairs depends on
    signal transitions, not the probe sampling frequency.

    Inputs must be equal-length, nonconstant, binary sample vectors.
    """
    n = len(first)
    if n != len(second) or n == 0:
        raise ValueError("correlation requires equal-length, nonempty signals")
    a_edges = np.r_[0, np.flatnonzero(np.diff(second)) + 1, n]
    b_edges = np.r_[0, np.flatnonzero(np.diff(first)) + 1, n]
    # Event pairing becomes expensive for nearly alternating signals. Keep
    # the established FFT path for those unusually transition-dense windows.
    if (len(a_edges) - 1) * (len(b_edges) - 1) > n // 16:
        normalized_first = (first - np.mean(first)) / (np.std(first) * n)
        normalized_second = (second - np.mean(second)) / np.std(second)
        return signal.correlate(normalized_second, normalized_first, mode="full")
    ma = np.mean(second)
    mb = np.mean(first)
    va = (second[a_edges[:-1]] - ma) / np.sqrt(ma * (1 - ma))
    vb = (first[b_edges[:-1]] - mb) / (np.sqrt(mb * (1 - mb)) * n)
    weight = (va[:, None] * vb[None, :]).ravel()
    a0 = a_edges[:-1, None]
    a1 = a_edges[1:, None]
    b0 = b_edges[None, :-1]
    b1 = b_edges[None, 1:]
    changes = np.zeros(2 * n + 1, dtype=np.float64)
    np.add.at(changes, (a0 - b1 + n).ravel(), weight)
    np.add.at(changes, (a0 - b0 + n).ravel(), -weight)
    np.add.at(changes, (a1 - b1 + n).ravel(), -weight)
    np.add.at(changes, (a1 - b0 + n).ravel(), weight)
    slopes = np.cumsum(changes)
    return (np.cumsum(slopes) - slopes)[1:-1]
