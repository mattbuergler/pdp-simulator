from __future__ import annotations

from pathlib import Path

import numpy as np

from dataio.H5Reader import H5Reader
from dataio.H5Writer import H5Writer


def test_hdf5_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sample.h5"
    values = np.arange(12, dtype=np.float64).reshape(4, 3)

    writer = H5Writer(path, "w")
    writer.writeDataSet("fluid/velocity", values, "float64")
    writer.close()

    reader = H5Reader(path)
    result = reader.getDataSet("fluid/velocity")[:]
    reader.close()

    np.testing.assert_array_equal(result, values)


def test_row_reader_does_not_load_dataset_to_inspect_shape(tmp_path: Path) -> None:
    path = tmp_path / "sample.h5"
    writer = H5Writer(path, "w")
    writer.writeDataSet("fluid/velocity", np.zeros((1, 3)), "float64")
    writer.close()

    class SliceOnlyDataset:
        shape = (1_000_000, 3)

        def __getitem__(self, key: object) -> np.ndarray:
            assert key == (slice(2, 5), slice(None))
            return np.ones((3, 3))

    reader = H5Reader(path)
    reader.existDataset = lambda _name: True  # type: ignore[method-assign]
    reader.getF5 = lambda: {"fluid/velocity": SliceOnlyDataset()}  # type: ignore[method-assign]
    try:
        np.testing.assert_array_equal(
            reader.getRowsFromDataSet("fluid/velocity", 2, 4), np.ones((3, 3))
        )
    finally:
        reader.close()
