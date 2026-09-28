"""Direct transitions must match the established dense generator."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from stsg_ssg_functions import SBG_signal
from tools.globals import compress_signal, get_signal_frequency


def test_stationary_probe_direct_transitions_match_dense(tmp_path: Path) -> None:
    fixture = Path("tests/2_SBG_Signal/A_Sphere_ConstD/input")
    config = json.loads((fixture / "config.json").read_text(encoding="utf-8"))
    config["FLOW_PROPERTIES"]["duration"] = 0.2
    dense = tmp_path / "dense"
    sparse = tmp_path / "sparse"
    for directory in (dense, sparse):
        directory.mkdir()
        shutil.copyfile(fixture / "flow_data.h5", directory / "flow_data.h5")

    arguments = (
        config["FLOW_PROPERTIES"], config["PROBE"], config["REPRODUCIBILITY"],
        {}, False, 1,
    )
    SBG_signal(dense, *arguments, "False")
    SBG_signal(sparse, *arguments, "True")

    assert not (sparse / "binary_signal.h5").exists()
    with (
        h5py.File(dense / "binary_signal.h5") as old,
        h5py.File(sparse / "compressed_signal.h5") as new,
    ):
        signal = old["signal"][:]
        assert new["length"][0] == len(signal)
        assert new["f_s"][0] == get_signal_frequency(old["time"][:])
        for column, sensor_id in enumerate(new["sensor_ids"][:]):
            np.testing.assert_array_equal(
                new[f"signal{int(sensor_id)}"][:], compress_signal(signal[:, column])
            )

    with h5py.File(dense / "flow_data.h5") as old, h5py.File(sparse / "flow_data.h5") as new:
        dataset_names: list[str] = []
        old.visititems(lambda name, item: dataset_names.append(name)
                       if isinstance(item, h5py.Dataset) else None)
        for name in dataset_names:
            if name == "bubbles/pierced_overlapped_bubble_void_fraction":
                np.testing.assert_allclose(old[name][:], new[name][:], rtol=0, atol=1e-14)
            else:
                np.testing.assert_array_equal(old[name][:], new[name][:])
