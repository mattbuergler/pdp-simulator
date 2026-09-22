# Contributing

Thanks for improving PDP-SIM. Changes to numerical research software should be reviewable,
reproducible, and accompanied by evidence that the scientific results remain valid.

## Development setup

Install [uv](https://docs.astral.sh/uv/) and create the locked development environment:

```bash
uv sync --locked --all-groups
```

Use a focused branch and keep generated simulation output out of commits. Before opening a merge
request or pull request, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python tests/runtests.py -r tests/1_SBG_Timeseries/A_Velocity
uv build
```

See [the testing guide](doc/testing.md) for the distinction between fast unit tests and numerical
regression tests.

## Scientific changes

For a change that can affect numerical output:

1. State the physical or numerical rationale in the merge request.
2. Add or update a focused test and report the before/after result.
3. Do not overwrite golden reference data merely to make a test pass. Explain and review every
   intentional reference update independently.
4. Record any change to assumptions, units, schema fields, random seeds, or tolerances.

## Scope of automated style checks

The modern typed interface under `src/pdp_sim`, configuration loader, and unit tests are enforced by
Ruff and mypy. The validated research kernels remain covered primarily by numerical regression tests;
formatting-only rewrites of those modules should not be mixed with scientific changes.
