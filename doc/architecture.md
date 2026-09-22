# Architecture

PDP-SIM models the complete path from turbulent carrier-flow statistics to the quantities recovered
from an intrusive phase-detection probe.

```text
config.json
    |
    v
stochastic 3-D velocity time series
    |
    v
bubble population and probe interaction
    |
    v
synthetic binary sensor signals
    |
    v
multi-sensor signal processing
    |
    v
reconstructed velocity, void fraction, diameter and IAC
```

## Components

| Component | Responsibility | Main interface |
| --- | --- | --- |
| `configuration.py` and `schemadef/` | Load and validate scenario inputs | `load_config()` |
| `stsg_ssg.py` and `stsg_ssg_functions.py` | Generate turbulent time series and synthetic signals | `pdp-sim generate` |
| `mssp.py` | Reconstruct flow properties from sensor signals | `pdp-sim process` |
| `evaluate.py` | Extract results from completed simulations | `pdp-sim evaluate` |
| `dataio/` | Read and write the HDF5 interchange format | `H5Reader`, `H5Writer` |
| `tests/unit/` | Fast behavioral checks for maintained interfaces | `pytest` |
| `tests/1_*` to `tests/4_*` | End-to-end numerical regression fixtures | `tests/runtests.py` |

The command package is deliberately a thin façade around the published numerical implementation.
This keeps installation and automation conventional without obscuring the research kernels or
silently changing established results.

The processor has a separate input boundary: it loads only `PROBE` and `RECONSTRUCTION` from the
configuration and reads `binary_signal.h5`. It does not receive the generator's flow or bubble data.
See [the measurement guide](measurements.md) for the real-recording format.

## Reproducibility boundary

`pyproject.toml` declares supported dependency ranges; `uv.lock` captures the exact cross-platform
resolution used by CI. Scenario JSON is validated against the bundled schema before execution. The
regression suite compares generated HDF5 datasets against committed golden references with tolerances
defined per scenario in `testdef`.

The large scientific fixtures are intentionally retained in Git for now. Moving them to DVC, Git LFS,
or an object store should only happen together with a durable public remote and a preservation plan,
so that a fresh clone remains reproducible.
