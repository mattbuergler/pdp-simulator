from __future__ import annotations

import numpy as np

from stsg_ssg_functions import SBG_dt_corr, generate_velocity_chunk


def test_vectorized_velocity_chunk_matches_langevin_recurrence() -> None:
    generator = np.random.default_rng(42)
    random_values = generator.multivariate_normal(
        mean=np.zeros(3),
        cov=np.array([[1.0, -0.3, 0.0], [-0.3, 1.0, 0.1], [0.0, 0.1, 1.0]]),
        size=100,
    )
    previous = np.array([2.0, 0.1, -0.2])
    mean = np.array([2.1, 0.0, 0.0])
    sigma = np.array([0.2, 0.1, 0.05])
    time_scale = np.array([0.5, 0.3, 0.2])
    dt = 0.001

    expected = np.empty_like(random_values)
    prior = previous
    for index, random_value in enumerate(random_values):
        expected[index] = SBG_dt_corr(prior, mean, sigma, time_scale, dt, random_value)
        prior = expected[index]

    result = generate_velocity_chunk(previous, mean, sigma, time_scale, dt, random_values)

    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)
