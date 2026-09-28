"""Regression tests for the AWCC windowing conventions."""

from __future__ import annotations

import numpy as np
import pytest

from mssp import chord, windows


@pytest.mark.parametrize(
    ("samples", "water", "air"),
    [
        ([], [], []),
        ([0], [], []),
        ([1], [], []),
        ([0, 0, 0], [3], []),
        ([1, 1, 1], [], [3]),
        ([0, 0, 1, 1, 0, 0], [2, 2], [2]),
        ([0, 0, 1], [2], []),  # final one-sample chord is excluded
        ([1, 0, 1, 0], [1], [1, 1]),
    ],
)
def test_chord_preserves_end_of_signal_convention(
    samples: list[int], water: list[int], air: list[int]
) -> None:
    actual_water, actual_air, frequency = chord(np.asarray(samples, dtype=np.int8), 2.0)
    assert actual_water == water
    assert actual_air == air
    assert frequency == len(air) / 2.0


def test_windows_preserve_boundaries_and_unequal_chord_counts() -> None:
    count, start, stop, centers, chord_times = windows([2, 3, 5, 7], [11, 13, 17, 19, 23], 2, 10)
    assert count == 2
    np.testing.assert_array_equal(start, [0, 29])
    np.testing.assert_array_equal(stop, [29, 99])
    np.testing.assert_array_equal(centers, [1.4, 6.4])
    np.testing.assert_allclose(chord_times, [[0.2, 0.3], [0.5, 0.7]])


def test_windows_with_no_complete_window() -> None:
    count, start, stop, centers, chord_times = windows([2], [3], 2, 10)
    assert count == 0
    assert len(start) == len(stop) == len(centers) == len(chord_times) == 0
