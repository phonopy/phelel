"""Import phelel.Phelel and phelel.load."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

try:
    __version__ = _package_version("phelel")
except PackageNotFoundError:
    try:
        from phelel._version import __version__
    except ImportError:
        __version__ = "0.0.0"

from phelel.api_phelel import Phelel  # noqa F401
from phelel.cui.load import load  # noqa F401

__all__ = ["Phelel", "load", "__version__"]
