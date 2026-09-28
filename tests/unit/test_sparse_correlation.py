"""Differential tests for the binary AWCC correlation kernel."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from mssp import velocity
from tools.sparse_correlation import run_correlation


@pytest.mark.parametrize("length", [3, 10, 100, 1000])
def test_run_correlation_matches_fft(length: int) -> None:
    rng = np.random.default_rng(length)
    for _ in range(20):
        first = (rng.random(length) < 0.25).astype(np.int8)
        second = (rng.random(length) < 0.25).astype(np.int8)
        if np.min(first) == np.max(first) or np.min(second) == np.max(second):
            continue
        normalized_first = ((first - first.mean()) / (first.std() * length))
        normalized_second = (second - second.mean()) / second.std()
        expected = signal.correlate(normalized_second, normalized_first, mode="full")
        np.testing.assert_allclose(run_correlation(first, second), expected, rtol=1e-10, atol=1e-10)


def test_constant_binary_window_is_undefined() -> None:
    first = np.zeros(100, dtype=np.int8)
    second = np.r_[np.zeros(50, dtype=np.int8), np.ones(50, dtype=np.int8)]
    assert all(np.isnan(item) for item in velocity(100, 0.005, 500000, first, second))


def test_sparse_runs_match_fft() -> None:
    first = np.zeros(1000, dtype=np.int8)
    second = np.zeros(1000, dtype=np.int8)
    first[100:140] = first[600:650] = 1
    second[120:160] = second[620:670] = 1
    expected = signal.correlate(
        (second - second.mean()) / second.std(),
        (first - first.mean()) / (first.std() * len(first)),
        mode="full",
    )
    np.testing.assert_allclose(run_correlation(first, second), expected, rtol=1e-10, atol=1e-10)
    filtered, r12, spr, unfiltered, _, _ = velocity(
        len(first), 0.005, 500000, first, second
    )
    assert filtered == unfiltered == pytest.approx(125.0)
    assert r12 > 0.4 * (spr**2 + 1)


def test_nonbinary_velocity_keeps_general_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = signal.correlate

    def wrapped(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(signal, "correlate", wrapped)
    first = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    second = np.roll(first, 1)
    velocity(len(first), 0.005, 500000, first, second)
    assert calls
