"""Repeatable bubble-piercing benchmark at the historical 500 kHz probe rate."""

from __future__ import annotations

from statistics import median
from time import perf_counter

import numpy as np

from stsg_ssg_functions import SBG_simulate_bubble_piercing


def main() -> None:
    generator = np.random.default_rng(42)
    fluid_time = np.arange(10_001, dtype=float) / 10_000
    trajectory = np.column_stack((10.0 * fluid_time, np.zeros((len(fluid_time), 2))))
    velocity = np.column_stack((np.full(len(fluid_time), 10.0), np.zeros((len(fluid_time), 2))))
    control_volume = np.array([0.011, 0.007, 0.007])
    sensors = {0: np.array([-0.0025, 0.0005, 0.0]), 1: np.array([0.0025, -0.0005, 0.0])}
    arrivals = generator.uniform(0.0, 0.99, size=5_000)
    locations = generator.uniform(-0.0035, 0.0035, size=(len(arrivals), 2))
    bubble_size = np.full(3, 0.003)

    def timed(prefilter: bool) -> tuple[float, int, int]:
        start = perf_counter()
        pierced = 0
        signal_samples = 0
        for bubble_id, (arrival, location) in enumerate(zip(arrivals, locations)):
            result = SBG_simulate_bubble_piercing(
                bubble_id,
                fluid_time,
                trajectory,
                velocity,
                control_volume,
                np.array([arrival]),
                np.array([0.0, *location]),
                bubble_size,
                1 / 500_000,
                sensors,
                0,
                streamwise_monotonic=True,
                prefilter_static_probe=prefilter,
            )
            count = sum(len(indices[0]) for indices in result[1].values())
            pierced += count > 0
            signal_samples += count
        return perf_counter() - start, pierced, signal_samples

    runs = [(mode, timed(mode)) for mode in (False, True, True, False)]
    baseline = median(result[0] for mode, result in runs if not mode)
    optimized = median(result[0] for mode, result in runs if mode)
    assert len({result[1:] for _, result in runs}) == 1
    print(f"bubbles: {len(arrivals)}")
    print(f"pierced: {runs[0][1][1]}")
    print(f"signal samples: {runs[0][1][2]}")
    print(f"baseline: {baseline:.3f} s")
    print(f"optimized: {optimized:.3f} s")
    print(f"speedup: {baseline / optimized:.2f}x")


if __name__ == "__main__":
    main()
