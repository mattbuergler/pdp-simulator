from __future__ import annotations

import sys
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from pdp_sim import cli


def test_unified_cli_requires_known_command() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["unknown"])


def test_delegate_restores_arguments() -> None:
    original = sys.argv.copy()
    observed: list[str] = []

    def entry_point() -> None:
        observed.extend(sys.argv)

    cli._delegate(entry_point, ["--example"])
    assert observed[1:] == ["--example"]
    assert sys.argv == original


@pytest.mark.parametrize(
    ("command", "module_name", "function"),
    [
        ("generate", "stsg_ssg", cli.generate),
        ("process", "mssp", cli.process),
        ("evaluate", "evaluate", cli.evaluate),
    ],
)
def test_legacy_entry_points_delegate(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    module_name: str,
    function: Callable[[], None],
) -> None:
    called: list[str] = []
    monkeypatch.setitem(
        sys.modules, module_name, SimpleNamespace(main=lambda: called.append(command))
    )
    function()
    assert called == [command]


def test_unified_cli_forwards_remaining_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[str] = []

    def generate() -> None:
        observed.extend(sys.argv)

    monkeypatch.setattr(cli, "generate", generate)
    cli.main(["generate", "--progress", "scenario"])
    assert observed[1:] == ["--progress", "scenario"]
