# Performance

Performance work in PDP-SIM is profile-driven and must preserve the numerical regression fixtures.
The first optimization pass targeted stochastic velocity generation because profiling identified its
Langevin recurrence and per-sample Reynolds-stress loop as the dominant compute work.

## Implemented optimization

The published algorithm is unchanged, but its execution is reorganized:

- The three autoregressive velocity components are evaluated by SciPy's compiled IIR filter instead
  of calling a Python function for every sample.
- Probe trajectory integration uses a vectorized cumulative sum.
- Reynolds stress is calculated directly as a matrix product instead of constructing an `(n, 3, 3)`
  intermediate tensor sample by sample.
- Unused pandas and Matplotlib imports were removed from the generation path.

The unit test compares the optimized recurrence with the iterative equation at a tolerance of
`1e-13`. The end-to-end golden-data regression validates velocity, trajectory, Reynolds stress, mean
velocity, and turbulent intensity at the scenario's published tolerances.

## Baseline result

Measured on the same local Windows workstation with Python 3.11.9 and a one-million-sample synthetic
input:

| Implementation | Time | Relative speed |
| --- | ---: | ---: |
| Iterative Python recurrence | 12.078 s | 1.0x |
| Compiled/vectorized recurrence | 0.061 s | 198.8x |

For the tracked 100,001-sample `A_Velocity` scenario, the simulator's timed generation section fell
from 2.26 s to 0.07 s (about 32x). Full process startup remains dominated by scientific-library
imports, so end-to-end gains vary with workload size and operating-system cache state.

Reproduce the focused benchmark with:

```bash
uv run python -m benchmarks.benchmark_velocity
```

Treat wall-clock comparisons as local measurements: report the hardware, interpreter, dependency
lock, sample count, and both raw timings when evaluating later optimizations.
