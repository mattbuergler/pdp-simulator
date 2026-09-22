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
