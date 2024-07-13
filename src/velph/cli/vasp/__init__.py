"""velph command line tool / velph-files-relaxa."""

import click

from velph.cli import cmd_root


@cmd_root.group("vasp")
@click.help_option("-h", "--help")
def cmd_vasp():
    """Choose vasp options."""
    pass


from velph.cli.vasp.band import cmd_band  # noqa F401
from velph.cli.vasp.dos import cmd_dos  # noqa F401
from velph.cli.vasp.nonscf import cmd_nonscf  # noqa F401
