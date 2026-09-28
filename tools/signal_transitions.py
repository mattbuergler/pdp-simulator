"""Sparse binary probe signals, indexed by their sample transitions."""

from __future__ import annotations

import math

import numpy as np


def contact_intervals(indices: np.ndarray) -> list[tuple[int, int]]:
    """Convert sorted contact sample indices into half-open intervals."""
    samples = np.asarray(indices, dtype=np.int64)
    if not samples.size:
        return []
    breaks = np.flatnonzero(np.diff(samples) != 1) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, samples.size]
    return [(int(samples[a]), int(samples[b - 1]) + 1) for a, b in zip(starts, stops)]


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the union of half-open intervals, including touching intervals."""
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(intervals):
        if stop <= start:
            raise ValueError("contact interval must have positive length")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def encode_intervals(intervals: list[tuple[int, int]], length: int) -> np.ndarray:
    """Encode a union of contact intervals in the existing transition format."""
    if length <= 0:
        raise ValueError("signal length must be positive")
    merged = merge_intervals(intervals)
    if merged and (merged[0][0] < 0 or merged[-1][1] > length):
        raise ValueError("contact interval outside signal")
    initial = int(bool(merged and merged[0][0] == 0))
    changes = [boundary for start, stop in merged for boundary in (start, stop)
               if 0 < boundary < length]
    return np.asarray([initial, *changes], dtype=np.int64)


def regular_sample_chunk_indices(chunk_times: np.ndarray, length: int, dt: float) -> np.ndarray:
    """Locate inclusive chunk bounds without allocating the full regular time vector."""
    if length <= 0:
        raise ValueError("signal length must be positive")
    end_time = (length - 1) * dt
    step = end_time / (length - 1) if length > 1 else 0.0

    def sample_time(index: int) -> float:
        return end_time if index == length - 1 else index * step

    def first_at_or_after(bound: float) -> int:
        index = max(0, min(length, math.floor(bound / step) if step else 0))
        while index > 0 and sample_time(index - 1) >= bound:
            index -= 1
        while index < length and sample_time(index) < bound:
            index += 1
        return index

    def first_after(bound: float) -> int:
        index = first_at_or_after(bound)
        while index < length and sample_time(index) <= bound:
            index += 1
        return index

    result = np.empty((len(chunk_times), 2), dtype=int)
    if not len(chunk_times):
        return result
    result[:, 0] = [first_at_or_after(float(start)) for start in chunk_times[:, 0]]
    result[:, 1] = [first_at_or_after(float(stop)) - 1 for stop in chunk_times[:, 1]]
    result[-1, 1] = first_after(float(chunk_times[-1, 1])) - 1
    result[result[:, 0] > result[:, 1]] = -1
    return result


class TransitionSignal:
    """Read AWCC windows from transition indices without full decompression."""

    def __init__(self, transitions: list[np.ndarray], length: int):
        if length <= 0 or not transitions:
            raise ValueError("a nonempty signal is required")
        self.length = int(length)
        self.transitions = [np.asarray(item, dtype=np.int64) for item in transitions]
        for item in self.transitions:
            if item.size == 0 or item[0] not in (0, 1):
                raise ValueError("invalid initial signal state")
            if (np.any(np.diff(item[1:]) <= 0) or np.any(item[1:] <= 0)
                    or np.any(item[1:] >= length)):
                raise ValueError("invalid signal transitions")

    def window(self, start: int, stop: int) -> np.ndarray:
        if not (0 <= start <= stop <= self.length):
            raise IndexError("signal window outside measurement")
        result = np.zeros((stop - start, len(self.transitions)), dtype=np.int8)
        for column, compressed in enumerate(self.transitions):
            changes = compressed[1:]
            left = int(np.searchsorted(changes, start, side="right"))
            right = int(np.searchsorted(changes, stop, side="left"))
            state = int(compressed[0]) ^ (left & 1)
            cursor = start
            for boundary in changes[left:right]:
                boundary = int(boundary)
                if state:
                    result[cursor - start:boundary - start, column] = 1
                state ^= 1
                cursor = boundary
            if state:
                result[cursor - start:, column] = 1
        return result

    def first_tip_chords(self, duration: float) -> tuple[list[int], list[int], float]:
        if self.length < 2:
            return [], [], 0.0
        compressed = self.transitions[0]
        boundaries = compressed[1:]
        ends = boundaries
        if not boundaries.size or boundaries[-1] != self.length - 1:
            ends = np.r_[ends, self.length]
        starts = np.r_[0, boundaries][:len(ends)]
        lengths = ends - starts
        phases = int(compressed[0]) ^ (np.arange(len(ends)) & 1)
        water = lengths[phases == 0].tolist()
        air = lengths[phases == 1].tolist()
        return water, air, len(air) / duration

    def first_tip_mean(self) -> float:
        compressed = self.transitions[0]
        boundaries = np.r_[0, compressed[1:], self.length]
        phases = int(compressed[0]) ^ (np.arange(len(boundaries) - 1) & 1)
        return int(np.sum(np.diff(boundaries)[phases == 1])) / self.length
