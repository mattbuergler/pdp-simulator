"""AWCC evaluation must average the same fluid samples as before."""

from __future__ import annotations

import numpy as np

from evaluate import window_average_fluid_velocity


def test_window_average_uses_left_closed_right_open_intervals() -> None:
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    velocity = np.column_stack((times, 2 * times, 3 * times))
    windows = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 2.1], [0.1, 0.2]])
    actual = window_average_fluid_velocity(times, velocity, windows)
    np.testing.assert_allclose(actual[:3], [[0.25, 0.5, 0.75], [1.25, 2.5, 3.75], [2, 4, 6]])
    assert np.isnan(actual[3]).all()


def test_window_average_matches_scanning_with_nans_and_overlap() -> None:
    generator = np.random.default_rng(42)
    times = np.sort(generator.uniform(0.0, 10.0, size=500))
    velocity = generator.normal(size=(len(times), 3))
    velocity[20:30, 1] = np.nan
    velocity[400:410, 2] = np.nan
    starts = np.arange(0.0, 10.0, 0.2)
    windows = np.column_stack((starts, starts + 0.35))
    expected = np.empty((len(windows), 3))
    for index, (start, stop) in enumerate(windows):
        selected = velocity[(times >= start) & (times < stop)]
        expected[index] = np.nanmean(selected, axis=0) if len(selected) else np.nan
    np.testing.assert_allclose(
        window_average_fluid_velocity(times, velocity, windows), expected, equal_nan=True
    )
