# Testing and validation

PDP-SIM uses two complementary test layers.

## Fast unit tests

```bash
uv run pytest
```

These tests cover configuration validation, HDF5 round trips, signal compression, and the installed
command-line interface. CI enforces an 80% coverage floor for the maintained package and configuration
loader.

## Numerical regression tests

The scenario directories under `tests/1_SBG_Timeseries`, `tests/2_SBG_Signal`, `tests/3_RA`, and
`tests/4_UQ` contain immutable inputs, per-dataset tolerances, and golden reference output. Run one
scenario with:

```bash
uv run python tests/runtests.py -r tests/1_SBG_Timeseries/A_Velocity
```

The runner copies inputs to an ignored `run/` directory, executes the workflow with the same Python
interpreter, and compares generated datasets with the tracked reference data. GitHub CI runs the small
velocity scenario as an end-to-end smoke regression. GitLab CI preserves the complete sequential and
two-worker regression jobs for exhaustive scientific validation.

To execute a group, pass its directory instead of one scenario:

```bash
uv run python tests/runtests.py -r tests/1_SBG_Timeseries
```

Reference updates are scientific changes. Inspect the numerical differences and document their cause
before using the runner's update utility.

## Quality and packaging gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
```

These gates check the maintained interface, strict type annotations, and both source and wheel builds.
The dependency lock is verified with `uv sync --locked --all-groups` in CI on Python 3.11, 3.12, and
3.13.

## Performance benchmark

The velocity generator benchmark compares the published iterative implementation with the optimized
compiled recurrence using the same one-million-sample input:

```bash
uv run python -m benchmarks.benchmark_velocity
```

Report the hardware, Python version, and both timings when using benchmark results in a review.
