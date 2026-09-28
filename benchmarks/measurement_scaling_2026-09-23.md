# Historical dual-tip measurement scaling

Measured on Windows with Python 3.11.9, one worker, and the transition-based
`-c True` generation and dual-tip AWCC processing paths. The flow and probe
settings match `run/config.json`; the legacy `REPRODUCIBLE` field was translated
to the current schema with seed 42. The resulting 1,000-second configuration is
saved as `benchmarks/historical_measurement_1000s.json`. Each duration was run
on a local temporary disk directory rather than the OneDrive-synced workspace.

| Measurement duration | Fluid generation | Signal generation | AWCC | AWCC windows | Stage total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 s | 0.40 s | 11.38 s | 23.74 s | 513 | 35.52 s |
| 300 s | 1.63 s | 40.45 s | 59.04 s | 1,538 | 101.12 s |
| 600 s | 3.19 s | 115.92 s | 144.42 s | 3,067 | 263.53 s |
| 1,000 s | 4.52 s | 158.67 s | 229.95 s | 5,096 | 393.14 s |

The historical `run/run.output` records 131.50 s fluid generation, 12,402.39 s
signal generation, and 3,274.14 s AWCC for the 1,000-second case: 15,808.03 s
total. The measured current stage total is approximately 40.2 times lower.
This is a comparison across different code versions and possibly different
hardware, not an isolated speedup attributable to one change.

Validation at 1,000 s:

- The new unfiltered `U.csv` and historical `run/U.csv` contain the same 5,096
  numerical values, including NaN positions.
- The compressed signal has 500,000,001 samples per tip, represented by 101,925
  and 98,377 transition entries. Its file is 1.534 MiB; the flow-data file is
  606.133 MiB and the reconstructed output is 1.762 MiB.
- AWCC wrote 14 reconstructed datasets. Of 5,096 windows, 624 have finite
  filtered velocities; the unfiltered `U.csv` has 5,095 finite values.
- AWCC emitted the existing invalid-value/FFT warning for a constant-signal
  window but completed successfully, as in the historical log.

The preceding 100-second dense-versus-transition comparison confirmed identical
transition indices at both tips, identical 14 reconstructed datasets and AWCC
CSV values, and 21 of 22 identical flow-data datasets. The remaining overlap
summary differed by approximately 1e-17 because of floating-point summation
order. The full unit suite passed (60 tests).

## AWCC correlation optimization (2026-09-24)

The 1,000-second compressed signal above was reprocessed without regenerating
the fluid or probe signal. Binary AWCC windows now accumulate cross-correlation
from signal phase-change runs, retaining the FFT path for transition-dense or
nonbinary input. The Python-level sum over every correlation sample was also
removed. The same Windows machine and Python 3.11 runtime measured **34.16 s**
for AWCC, versus **229.95 s** in the preceding full workflow: approximately
**6.7× faster** for this stage. This is a stage-only rerun, not a new end-to-end
measurement.

All 5,096 unfiltered velocities and all 14 reconstructed datasets matched the
previous run exactly, including NaN positions. The same 624 windows passed the
unchanged AWCC quality filter. The maximum absolute numerical differences in
the diagnostic files were 1.8e-12 for `R12max.csv` and 2.1e-11 for `SPR.csv`;
neither changed a filtering decision. The 67 unit tests passed without coverage.

## Signal-generation storage pass (2026-09-24)

Profiling the 100-second case showed 34,660 per-bubble tracking calls taking
8.7 s of 13.7 s under the profiler. Repeated opening of the same flow-data
HDF5 file also took about 0.9 s. Signal generation now keeps one read/write
handle open across tracking chunks; the bubble-placement virtual-distance
calculation uses scalar extrema rather than constructing small NumPy arrays.

On paired 100-second inputs, the reference implementation took 9.25 s and the
accepted implementation took 7.92 s (**1.17×**). A separate 10,000-bubble
placement microbenchmark measured 0.51–0.53 s before and 0.18 s after the
scalar-extrema change. The 1,000-second signal-generation stage took 85.70 s
in the new run, compared with 158.67 s in the prior-day run; those are not
paired timings, so their ratio should not be attributed solely to these edits.

At both 100 and 1,000 seconds, all five compressed-signal datasets and all 22
flow-data datasets matched the saved reference exactly, including NaN
positions. The full unit suite passed (68 tests, without coverage). A compiled
per-bubble prototype had excessive first-run compilation overhead, and larger
parallel chunks were slower and changed exit estimates; neither was retained.
