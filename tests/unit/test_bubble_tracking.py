"""Boundary cases for indexed bubble arrival lookup."""

from __future__ import annotations

import numpy as np
import pytest

from stsg_ssg_functions import (
    SBG_simulate_bubble_piercing,
    get_bubble_properties,
    get_virtual_distance,
)


def test_virtual_distance_matches_numpy_extrema() -> None:
    generator = np.random.default_rng(73)
    volume = np.array([0.011, 0.007, 0.007])
    for first, second in generator.uniform(-0.01, 0.01, size=(100, 2, 3)):
        y_min, y_max = np.min([first[1], second[1]]), np.max([first[1], second[1]])
        z_min, z_max = np.min([first[2], second[2]]), np.max([first[2], second[2]])
        dx = abs(second[0] - first[0])
        dy = (volume[1] / 2 - y_max) + (y_min + volume[1] / 2)
        dz = (volume[2] / 2 - z_max) + (z_min + volume[2] / 2)
        expected = np.sqrt(dx**2 + dy**2 + dz**2)
        assert get_virtual_distance(first, second, volume) == expected


@pytest.mark.parametrize(
    ("arrival_time", "first_fluid_time"),
    [(0.0, 0.0), (2.0, 2.0), (2.25, 2.0), (5.0, 5.0)],
)
def test_bubble_starts_at_latest_fluid_sample_not_after_arrival(
    arrival_time: float, first_fluid_time: float
) -> None:
    fluid_time = np.arange(6, dtype=float)
    trajectory = np.column_stack((fluid_time, np.zeros((6, 2))))
    velocity = np.column_stack((np.ones(6), np.zeros((6, 2))))
    properties = get_bubble_properties(
        np.array([arrival_time]),
        np.zeros(3),
        fluid_time,
        trajectory,
        velocity,
        np.ones(3),
    )
    assert properties["time"][0] == first_fluid_time


def test_bubble_before_first_fluid_sample_is_rejected() -> None:
    fluid_time = np.arange(1, 4, dtype=float)
    trajectory = np.column_stack((fluid_time, np.zeros((3, 2))))
    velocity = np.column_stack((np.ones(3), np.zeros((3, 2))))
    with pytest.raises(ValueError, match="before the first fluid sample"):
        get_bubble_properties(
            np.array([0.5]), np.zeros(3), fluid_time, trajectory, velocity, np.ones(3)
        )


def test_indexed_exit_lookup_matches_full_scan_for_monotonic_trajectory() -> None:
    generator = np.random.default_rng(42)
    fluid_time = np.linspace(0.0, 5.0, 5_001)
    streamwise = np.cumsum(generator.uniform(0.0, 0.002, size=len(fluid_time)))
    trajectory = np.column_stack((streamwise, np.zeros((len(fluid_time), 2))))
    velocity = np.column_stack((np.ones(len(fluid_time)), np.zeros((len(fluid_time), 2))))
    volume = np.array([0.03, 0.01, 0.01])
    for arrival_time in np.linspace(0.0, 4.9, 50):
        arguments = (
            np.array([arrival_time]),
            np.zeros(3),
            fluid_time,
            trajectory,
            velocity,
            volume,
        )
        scanned = get_bubble_properties(*arguments)
        indexed = get_bubble_properties(*arguments, streamwise_monotonic=True)
        for name in ("time", "trajectory", "velocity", "exit_time", "exit_location"):
            np.testing.assert_array_equal(indexed[name], scanned[name])


@pytest.mark.parametrize("vibrating", [False, True])
def test_bubble_piercing_prefilter_matches_original_path(vibrating: bool) -> None:
    generator = np.random.default_rng(42)
    fluid_time = np.arange(101, dtype=float) / 10_000
    trajectory = np.column_stack((10.0 * fluid_time, np.zeros((len(fluid_time), 2))))
    velocity = np.column_stack((np.full(len(fluid_time), 10.0), np.zeros((len(fluid_time), 2))))
    sensors = {0: np.array([-0.0025, 0.0005, 0.0]), 1: np.array([0.0025, -0.0005, 0.0])}
    control_volume = np.array([0.011, 0.007, 0.007])
    bubble_size = np.full(3, 0.003)
    sample_step = 1 / 500_000
    probe_motion = np.zeros((5_001, 3)) if vibrating else np.array([])
    if vibrating:
        probe_motion[:, 1] = 0.00025

    for location in generator.uniform(-0.0035, 0.0035, size=(30, 2)):
        arrival = np.array([0.004])
        bubble_location = np.array([0.0, *location])
        arguments = (
            0, fluid_time, trajectory, velocity, control_volume,
            arrival, bubble_location, bubble_size, sample_step, sensors, 0,
            probe_motion,
        )
        actual = SBG_simulate_bubble_piercing(
            *arguments, streamwise_monotonic=True,
        )
        reference = SBG_simulate_bubble_piercing(
            *arguments, streamwise_monotonic=True, prefilter_static_probe=False,
        )
        assert actual[0] == reference[0]
        for sensor_id in sensors:
            np.testing.assert_array_equal(actual[1][sensor_id][0], reference[1][sensor_id][0])
        for property_name in reference[2]:
            np.testing.assert_array_equal(actual[2][property_name], reference[2][property_name])
