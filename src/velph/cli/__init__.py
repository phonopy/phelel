"""velph command line tool module."""

import click


@click.group()
@click.help_option("-h", "--help")
def cmd_root():
    """Command-line utility to help VASP el-ph calculation."""
    pass


from velph.cli.el_bands import cmd_el_bands  # noqa F401
from velph.cli.generate import cmd_generate  # noqa F401
from velph.cli.hints import cmd_hints  # noqa F401
from velph.cli.init import cmd_init  # noqa F401
from velph.cli.nac import cmd_nac  # noqa F401
from velph.cli.ph_bands import cmd_ph_bands  # noqa F401
from velph.cli.relax import cmd_relax  # noqa F401

# The followings are written here to avoid ciclic import but needed.
from velph.cli.selfenergy import cmd_selfenergy  # noqa F401
from velph.cli.supercell import cmd_supercell  # noqa F401
from velph.cli.transport import cmd_transport  # noqa F401
