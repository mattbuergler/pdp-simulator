"""Repeatable microbenchmark for the velocity recurrence optimization."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from stsg_ssg_functions import SBG_dt_corr, generate_velocity_chunk


def iterative_velocity(
    previous: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    time_scale: np.ndarray,
    dt: float,
    random_values: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(random_values)
    prior = previous
    for index, random_value in enumerate(random_values):
        result[index] = SBG_dt_corr(prior, mean, sigma, time_scale, dt, random_value)
        prior = result[index]
    return result


def timed(function: object, *arguments: object) -> float:
    start = perf_counter()
    function(*arguments)  # type: ignore[operator]
    return perf_counter() - start


def main() -> None:
    generator = np.random.default_rng(42)
    random_values = generator.normal(size=(1_000_000, 3))
    arguments = (
        np.array([2.0, 0.1, -0.2]),
        np.array([2.1, 0.0, 0.0]),
        np.array([0.2, 0.1, 0.05]),
        np.array([0.5, 0.3, 0.2]),
        0.001,
        random_values,
    )

    optimized = timed(generate_velocity_chunk, *arguments)
    iterative = timed(iterative_velocity, *arguments)
    print(f"optimized: {optimized:.3f} s")
    print(f"iterative: {iterative:.3f} s")
    print(f"speedup:   {iterative / optimized:.1f}x")


if __name__ == "__main__":
    main()
