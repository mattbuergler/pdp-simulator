"""Equivalence checks for sparse probe-signal generation and AWCC input."""

from __future__ import annotations

import numpy as np
import pytest

from mssp import chord
from stsg_ssg_functions import get_chunk_indices
from tools.globals import compress_signal
from tools.signal_transitions import (
    TransitionSignal,
    contact_intervals,
    encode_intervals,
    regular_sample_chunk_indices,
)


@pytest.mark.parametrize("length", [2, 3, 10, 1_001])
def test_transition_windows_and_chords_match_dense_signal(length: int) -> None:
    generator = np.random.default_rng(42 + length)
    dense = generator.integers(0, 2, size=(length, 2), dtype=np.int8)
    compressed = [compress_signal(dense[:, column]) for column in range(2)]
    sparse = TransitionSignal(compressed, length)
    for start, stop in [(0, length), (0, 1), (length - 1, length), (length // 3, 2 * length // 3)]:
        np.testing.assert_array_equal(sparse.window(start, stop), dense[start:stop])
    assert sparse.first_tip_chords(length / 500_000) == chord(dense[:, 0], length / 500_000)
    assert sparse.first_tip_mean() == np.mean(dense[:, 0])


def test_contact_intervals_encode_existing_format() -> None:
    contact = np.array([0, 1, 4, 5, 5, 6, 11])
    intervals = contact_intervals(np.unique(contact))
    compressed = encode_intervals(intervals, 12)
    dense = np.zeros(12, dtype=np.int8)
    dense[contact] = 1
    np.testing.assert_array_equal(compressed, compress_signal(dense))


@pytest.mark.parametrize("duration,frequency", [(10.0, 500_000), (0.013, 10_000), (3.7, 12_345)])
def test_regular_chunk_indices_match_dense_search(duration: float, frequency: int) -> None:
    dt = 1 / frequency
    length = int(np.ceil(duration / dt)) + 1
    times = np.linspace(0, (length - 1) * dt, length)
    starts = np.arange(0, duration, min(1.0, duration / 10))
    chunks = np.column_stack(
        (
            np.maximum(0, starts - 0.01),
            np.minimum(duration, starts + min(1.0, duration / 10) + 0.01),
        )
    )
    actual = regular_sample_chunk_indices(chunks, length, dt)
    expected = get_chunk_indices(len(chunks), chunks, times)
    np.testing.assert_array_equal(actual, expected)
