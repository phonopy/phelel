"""velph command line tool module."""

import click


@click.group()
@click.help_option("-h", "--help")
def cmd_root():
    """Command-line utility to help VASP el-ph calculation."""
    pass


from phelel.velph.cli.el_bands import cmd_el_bands  # noqa F401
from phelel.velph.cli.generate import cmd_generate  # noqa F401
from phelel.velph.cli.hints import cmd_hints  # noqa F401
from phelel.velph.cli.init import cmd_init  # noqa F401
from phelel.velph.cli.nac import cmd_nac  # noqa F401
from phelel.velph.cli.ph_bands import cmd_ph_bands  # noqa F401
from phelel.velph.cli.phelel import cmd_phelel  # noqa F401
from phelel.velph.cli.phono3py import cmd_phono3py  # noqa F401
from phelel.velph.cli.phonopy import cmd_phonopy  # noqa F401
from phelel.velph.cli.relax import cmd_relax  # noqa F401
from phelel.velph.cli.selfenergy import cmd_selfenergy  # noqa F401
from phelel.velph.cli.transport import cmd_transport  # noqa F401

__all__ = [
    "cmd_el_bands",
    "cmd_generate",
    "cmd_hints",
    "cmd_init",
    "cmd_nac",
    "cmd_phelel",
    "cmd_phonopy",
    "cmd_phono3py",
    "cmd_ph_bands",
    "cmd_relax",
    "cmd_selfenergy",
    "cmd_transport",
]
