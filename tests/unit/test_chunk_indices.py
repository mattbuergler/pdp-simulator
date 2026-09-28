"""Chunk lookup must retain the original inclusive-end convention."""

from __future__ import annotations

import numpy as np

from stsg_ssg_functions import get_chunk_indices


def test_chunk_boundaries_include_only_final_right_endpoint() -> None:
    times = np.arange(0.0, 2.5, 0.5)
    chunks = np.array([[0.0, 1.0], [1.0, 2.0]])
    np.testing.assert_array_equal(get_chunk_indices(2, chunks, times), [[0, 1], [2, 4]])


def test_chunk_lookup_supports_overlap_empty_chunks_and_column_times() -> None:
    times = np.arange(0.0, 2.5, 0.5).reshape(-1, 1)
    chunks = np.array([[0.1, 0.2], [0.0, 1.2], [0.8, 2.0]])
    np.testing.assert_array_equal(get_chunk_indices(3, chunks, times), [[-1, -1], [0, 2], [2, 4]])


def test_chunk_lookup_matches_scanning_reference() -> None:
    generator = np.random.default_rng(42)
    times = np.sort(generator.uniform(0.0, 10.0, size=100))
    starts = np.arange(0.0, 10.0, 0.25)
    chunks = np.column_stack((starts, np.minimum(starts + 0.4, 10.0)))
    expected = np.full((len(chunks), 2), -1, dtype=int)
    for index, (start, end) in enumerate(chunks):
        indices = np.flatnonzero(
            (start <= times) & (times <= end if index == len(chunks) - 1 else times < end)
        )
        if len(indices):
            expected[index] = [indices[0], indices[-1]]
    np.testing.assert_array_equal(get_chunk_indices(len(chunks), chunks, times), expected)
