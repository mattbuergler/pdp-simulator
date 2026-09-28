"""Measure chunk-index scaling without generating bubbles or writing HDF5 files."""

from __future__ import annotations

from statistics import median
from time import perf_counter

import numpy as np

from stsg_ssg_functions import get_chunk_indices, get_chunk_times


def scanning_reference(chunks: np.ndarray, times: np.ndarray) -> np.ndarray:
    indices = np.full((len(chunks), 2), -1, dtype=int)
    for index, (start, end) in enumerate(chunks):
        selected = np.flatnonzero(
            (start <= times) & (times <= end if index == len(chunks) - 1 else times < end)
        )
        if len(selected):
            indices[index] = [selected[0], selected[-1]]
    return indices


def measure(function: object, *arguments: object, repeats: int = 3) -> float:
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        function(*arguments)  # type: ignore[operator]
        durations.append(perf_counter() - start)
    return median(durations)


def main() -> None:
    frequency = 50_000
    for duration in (10, 300):
        times = np.linspace(0.0, duration, duration * frequency + 1)
        chunks = get_chunk_times(duration, 1.0, duration)
        indexed = measure(get_chunk_indices, len(chunks), chunks, times)
        print(f"{duration:3d} s, {len(times):,} samples: indexed {indexed * 1e6:.0f} us")
        if duration == 10:
            scanned = measure(scanning_reference, chunks, times)
            assert np.array_equal(
                get_chunk_indices(len(chunks), chunks, times), scanning_reference(chunks, times)
            )
            print(f"  previous scan {scanned:.4f} s; speedup {scanned / indexed:.1f}x")


if __name__ == "__main__":
    main()
