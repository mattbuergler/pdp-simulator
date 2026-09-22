from __future__ import annotations

import numpy as np
import pytest

from tools.globals import compress_signal, decompress_signal, get_signal_frequency


@pytest.mark.parametrize(
    "signal",
    [
        np.array([0, 0, 1, 1, 0, 1], dtype=np.int8),
        np.array([1, 1, 0, 0, 1, 0], dtype=np.int8),
        np.zeros(8, dtype=np.int8),
        np.ones(8, dtype=np.int8),
    ],
)
def test_signal_compression_round_trip(signal: np.ndarray) -> None:
    result = decompress_signal(compress_signal(signal), len(signal))
    np.testing.assert_array_equal(result, signal)


def test_signal_frequency() -> None:
    time = np.linspace(0.0, 0.9, 10)
    assert get_signal_frequency(time) == pytest.approx(10.0)


@pytest.mark.parametrize("compressed", [[], [2], [0, 0], [1, 6]])
def test_signal_decompression_rejects_invalid_encoding(compressed: list[int]) -> None:
    with pytest.raises(ValueError):
        decompress_signal(compressed, 6)
