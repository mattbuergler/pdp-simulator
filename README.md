# PDP-SIM

[![GitHub CI](https://github.com/mattbuergler/pdp-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/mattbuergler/pdp-simulator/actions/workflows/ci.yml)
[![GitLab pipeline](https://gitlab.ethz.ch/vaw/public/pdp-sim/badges/master/pipeline.svg)](https://gitlab.ethz.ch/vaw/public/pdp-sim/-/pipelines)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11–3.13-blue.svg)](https://www.python.org/)
[![GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ijmultiphaseflow.2024.104978-blue)](https://doi.org/10.1016/j.ijmultiphaseflow.2024.104978)

PDP-SIM is an open-source phase-detection probe simulator for turbulent bubbly flows. It creates
three-dimensional stochastic carrier-flow time series, tracks a synthetic bubble population past a
virtual probe, generates binary sensor signals, and reconstructs flow quantities with multi-sensor
signal-processing algorithms.

The simulator supports reproducible uncertainty analysis of intrusive probe measurements. Its methods
and scientific application are described in the peer-reviewed article
[Bürgler et al. (2024)](https://doi.org/10.1016/j.ijmultiphaseflow.2024.104978).

## Scientific workflow

```text
flow statistics -> stochastic velocity -> bubble/probe interaction
                -> binary sensor signals -> reconstructed flow properties
```

Reconstructed quantities include velocity, void fraction, bubble diameter, interaction times, and
interfacial area concentration (IAC), depending on the selected probe and processing algorithm. See
[the architecture guide](doc/architecture.md) for the component and reproducibility boundaries.

## Installation

PDP-SIM supports Python 3.11–3.13. The recommended setup uses
[uv](https://docs.astral.sh/uv/) and the committed cross-platform lockfile:

```bash
git clone https://gitlab.ethz.ch/vaw/public/pdp-sim.git
cd pdp-sim
uv sync --locked
```

For development and testing, install the development group as well:

```bash
uv sync --locked --all-groups
```

## Usage

Each simulation directory contains a `config.json` describing the flow, bubble population, probe, and
reconstruction settings. The complete JSON Schema is bundled under `schemadef/`; working examples are
available throughout `tests/`.

Generate a stochastic velocity time series:

```bash
uv run pdp-sim generate -r timeseries path/to/simulation
```

Generate the synthetic probe signal:

```bash
uv run pdp-sim generate -r signal path/to/simulation
```

Process the signal:

```bash
uv run pdp-sim process path/to/simulation
```

The dedicated commands `pdp-sim-generate`, `pdp-sim-process`, and `pdp-sim-evaluate` are also
installed. The original script interfaces remain available for compatibility.

## Reproducibility and tests

The project combines conventional software checks with domain-specific numerical regression:

- `uv.lock` fixes the dependency resolution used locally and in CI.
- JSON configurations are validated before a workflow runs.
- Unit tests cover the maintained interface, HDF5 I/O, and signal encoding.
- Golden reference scenarios verify numerical datasets with explicit tolerances.
- GitHub Actions and GitLab CI test Python 3.11, 3.12, and 3.13 and build the package.

Run the fast suite and one representative end-to-end regression with:

```bash
uv run pytest
uv run python tests/runtests.py -r tests/1_SBG_Timeseries/A_Velocity
```

Performance-sensitive changes can be measured with
`uv run python -m benchmarks.benchmark_velocity`; see the
[performance notes](doc/performance.md) for the measured baseline and optimization method.

The repository currently keeps its large scientific reference fixtures in Git so a clone is
self-contained. See [the testing guide](doc/testing.md) before modifying any golden data.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/pdp_sim/` | Installable command-line interface |
| `stsg_ssg.py` | Stochastic time-series and synthetic-signal generation |
| `mssp.py` | Multi-sensor signal processing |
| `configuration.py`, `schemadef/` | Validated scenario configuration |
| `dataio/` | HDF5 readers and writers |
| `tests/unit/` | Fast automated unit tests |
| `tests/1_*`–`tests/4_*` | Scientific end-to-end regression cases |
| `doc/` | Architecture, testing, user, and developer documentation |

## Citation

If PDP-SIM contributes to your research, cite both the software and the associated paper. GitHub and
compatible reference managers can read the complete metadata from [`CITATION.cff`](CITATION.cff).

> Bürgler, M., Valero, D., Hohermuth, B., Boes, R. M., & Vetsch, D. F. (2024). Uncertainties in
> measurements of bubbly flows using phase-detection probes. *International Journal of Multiphase
> Flow, 181*, 104978. https://doi.org/10.1016/j.ijmultiphaseflow.2024.104978

## Contributing and support

Contributions are welcome; start with [CONTRIBUTING.md](CONTRIBUTING.md). For bug reports and feature
requests, use the [GitLab issue tracker](https://gitlab.ethz.ch/vaw/public/pdp-sim/-/issues).

PDP-SIM was developed by Matthias Bürgler with Daniel Valero, Benjamin Hohermuth, David F. Vetsch, and
Robert M. Boes at the Laboratory of Hydraulics, Hydrology and Glaciology (VAW), ETH Zurich. Matthias
Bürgler and Benjamin Hohermuth were supported by the Swiss National Science Foundation (grant 197208).

## License

Copyright © 2024 ETH Zurich and the PDP-SIM authors. PDP-SIM is distributed under the
[GNU General Public License v3.0](LICENSE).
