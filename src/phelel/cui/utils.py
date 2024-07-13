"""CUI utility functions."""

import os
import sys

from phelel.version import __version__


# AA is created at http://www.network-science.de/ascii/.
def print_phelel():
    """Show phelel logo."""
    phelel_logo = r"""       _          _      _
 _ __ | |__   ___| | ___| |
| '_ \| '_ \ / _ \ |/ _ \ |
| |_) | | | |  __/ |  __/ |
| .__/|_| |_|\___|_|\___|_|
|_|"""
    print(phelel_logo + " " * 19 + "%s" % __version__)


def print_version(version):
    """Show phelel version."""
    print(" " * 42 + "%s" % version)
    print("")


def print_end():
    """Show end AA."""
    print(
        r"""                 _
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
"""
    )


def print_error():
    """Show error AA."""
    print(
        r"""  ___ _ __ _ __ ___  _ __
 / _ \ '__| '__/ _ \| '__|
|  __/ |  | | | (_) | |
 \___|_|  |_|  \___/|_|
"""
    )


def print_error_message(message):
    """Show error message."""
    print("")
    print(message)


def file_exists(filename, log_level):
    """Check existance of a file."""
    if os.path.exists(filename):
        return True
    else:
        error_text = "%s not found." % filename
        print_error_message(error_text)
        if log_level > 0:
            print_error()
        sys.exit(1)
