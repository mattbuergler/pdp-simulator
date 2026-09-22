"""Installed command-line entry points for the legacy scientific modules."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence


def _delegate(entry_point: Callable[[], None], arguments: Sequence[str]) -> None:
    original = sys.argv
    try:
        sys.argv = [original[0], *arguments]
        entry_point()
    finally:
        sys.argv = original


def _run_legacy(module_name: str) -> None:
    """Import a legacy workflow lazily and run its public entry point."""
    module = importlib.import_module(module_name)
    entry_point = getattr(module, "main", None)
    if not callable(entry_point):
        raise RuntimeError(f"{module_name} does not expose a callable main()")
    entry_point()


def generate() -> None:
    """Run stochastic time-series and synthetic-signal generation."""
    _run_legacy("stsg_ssg")


def process() -> None:
    """Run multi-sensor signal processing."""
    _run_legacy("mssp")


def evaluate() -> None:
    """Extract a completed simulation into a result table."""
    _run_legacy("evaluate")


def main(arguments: Sequence[str] | None = None) -> None:
    """Dispatch the unified ``pdp-sim`` command."""
    parser = argparse.ArgumentParser(
        prog="pdp-sim",
        description="Simulate and process phase-detection probe measurements.",
    )
    parser.add_argument("command", choices=("generate", "process", "evaluate"))
    namespace, remaining = parser.parse_known_args(arguments)
    entry_points = {
        "generate": generate,
        "process": process,
        "evaluate": evaluate,
    }
    original = sys.argv
    try:
        sys.argv = [f"pdp-sim {namespace.command}", *remaining]
        entry_points[namespace.command]()
    finally:
        sys.argv = original
