"""Phase-detection probe simulator for turbulent bubbly flows."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pdp-sim")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = ["__version__"]
