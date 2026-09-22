# Processing a measured dual-tip signal

The AWCC processor can run on a real probe recording without simulation inputs. Prepare a directory
containing `binary_signal.h5` and `config.json`.

The HDF5 file must contain a one-dimensional `time` dataset in seconds and a two-column binary
`signal` dataset with one column per tip. The current AWCC implementation expects the leading tip
(ID `0`) in column 0 and the trailing tip (ID `1`) in column 1, matching the order in `PROBE.sensors`.
Set the `sensor_id` attribute on `signal` to `[0, 1]`. The signal values are `0` and `1` for the two
phases.

For example, a minimal `config.json` is:

```json
{
  "PROBE": {
    "sampling_frequency": 50000,
    "sensors": [
      {"id": 0, "relative_location": [-0.00253, -0.0005, 0.0]},
      {"id": 1, "relative_location": [0.00253, 0.0005, 0.0]}
    ]
  },
  "RECONSTRUCTION": {
    "type": "dual_tip_AWCC",
    "n_particles": 8
  }
}
```

The positions are in metres and the sampling frequency is in hertz. `n_particles` specifies the
number of detected air chords used to form each averaging window. Run:

```bash
uv run pdp-sim process path/to/measurement
```

The command writes `reconstructed.h5` and AWCC diagnostic files in that directory. It reads the
recorded signals, their time axis, probe geometry, and reconstruction settings. `FLOW_PROPERTIES`,
`REPRODUCIBILITY`, `flow_data.h5`, simulated bubble locations, and true velocities are not inputs to
the reconstruction. The `bubbles/` HDF5 group name is historical; in AWCC mode, its per-row velocity
and interaction-time values refer to averaging windows.
