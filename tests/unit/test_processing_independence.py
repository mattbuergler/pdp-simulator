"""AWCC must run using a recorded signal without simulator truth data."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


def test_dual_tip_awcc_needs_only_signal_and_probe_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path("tests/3_RA/C_2tip_AWCC")
    full_config = json.loads((fixture / "input/config.json").read_text(encoding="utf-8"))
    measurement_config = {
        "PROBE": full_config["PROBE"],
        "RECONSTRUCTION": full_config["RECONSTRUCTION"],
    }
    (tmp_path / "config.json").write_text(json.dumps(measurement_config), encoding="utf-8")
    shutil.copyfile(fixture / "input/binary_signal.h5", tmp_path / "binary_signal.h5")
    assert not (tmp_path / "flow_data.h5").exists()

    from mssp import main

    monkeypatch.setattr(sys, "argv", ["pdp-sim-process", str(tmp_path)])
    main()

    with h5py.File(tmp_path / "reconstructed.h5") as reconstructed:
        actual_velocity = reconstructed["bubbles/velocity"][:, 0]
        actual_times = reconstructed["bubbles/interaction_times"][:]
    expected_velocity = np.loadtxt(fixture / "ref/reconstructed_bubbles_velocity_c0.dat")
    expected_times = np.loadtxt(fixture / "ref/reconstructed_bubbles_interaction_times.dat")
    np.testing.assert_allclose(actual_velocity, expected_velocity, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_times, expected_times, rtol=1e-6, atol=1e-6)
    assert not (tmp_path / "flow_data.h5").exists()


def test_dual_tip_awcc_reads_compressed_windows_without_dense_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib import pyplot as plt

    plt.switch_backend("Agg")
    fixture = Path("tests/3_RA/C_2tip_AWCC")
    full_config = json.loads((fixture / "input/config.json").read_text(encoding="utf-8"))
    measurement_config = {
        "PROBE": full_config["PROBE"],
        "RECONSTRUCTION": full_config["RECONSTRUCTION"],
    }
    (tmp_path / "config.json").write_text(json.dumps(measurement_config), encoding="utf-8")

    from mssp import main, write_compressed_signal

    with h5py.File(fixture / "input/binary_signal.h5") as source:
        time = source["time"][:]
        signal = source["signal"][:]
        sensor_ids = source["signal"].attrs["sensor_id"]
    write_compressed_signal(tmp_path, time, signal, sensor_ids)
    assert not (tmp_path / "binary_signal.h5").exists()
    assert not (tmp_path / "flow_data.h5").exists()

    monkeypatch.setattr(sys, "argv", ["pdp-sim-process", str(tmp_path), "-c", "True"])
    main()

    with h5py.File(tmp_path / "reconstructed.h5") as reconstructed:
        actual_velocity = reconstructed["bubbles/velocity"][:, 0]
        actual_times = reconstructed["bubbles/interaction_times"][:]
    expected_velocity = np.loadtxt(fixture / "ref/reconstructed_bubbles_velocity_c0.dat")
    expected_times = np.loadtxt(fixture / "ref/reconstructed_bubbles_interaction_times.dat")
    np.testing.assert_allclose(actual_velocity, expected_velocity, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_times, expected_times, rtol=1e-6, atol=1e-6)
    assert not (tmp_path / "flow_data.h5").exists()
