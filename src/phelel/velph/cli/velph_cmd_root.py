"""velph command line tool module."""

import click

from phelel.velph.errors import VelphError


class _VelphGroup(click.Group):
    """Group that shows VelphError as click.ClickException.

    Group.invoke of click parses the arguments of the subcommand and runs it
    inside itself, and a subgroup does the same for its subcommands.  So all
    the commands of velph run inside invoke of the root group, and VelphError
    raised in any of them is converted here.  This relies on the code of click
    rather than on its documentation; test_velph_cmd_root.py fails when click
    runs the subcommands outside invoke of the root group.

    """

    def invoke(self, ctx: click.Context):
        """Invoke the command, with VelphError converted to ClickException."""
        try:
            return super().invoke(ctx)
        except VelphError as e:
            raise click.ClickException(str(e)) from e


@click.group(cls=_VelphGroup)
@click.help_option("-h", "--help")
def cmd_root():
    """Command-line utility to help VASP el-ph calculation."""
    pass


from phelel.velph.cli.convert.cmd_convert import cmd_convert  # noqa: E402, F401
from phelel.velph.cli.el_bands.cmd_el_bands import cmd_el_bands  # noqa: E402, F401
from phelel.velph.cli.el_dos.cmd_el_dos import cmd_el_dos  # noqa: E402, F401
from phelel.velph.cli.generate.cmd_generate import cmd_generate  # noqa: E402, F401
from phelel.velph.cli.init.cmd_init import cmd_init  # noqa: E402, F401
from phelel.velph.cli.nac.cmd_nac import cmd_nac  # noqa: E402, F401
from phelel.velph.cli.ph_bands.cmd_ph_bands import cmd_ph_bands  # noqa: E402, F401
from phelel.velph.cli.ph_selfenergy.cmd_ph_selfenergy import (  # noqa: E402, F401
    cmd_ph_selfenergy,
)
from phelel.velph.cli.phelel.cmd_phelel import cmd_phelel  # noqa: E402, F401
from phelel.velph.cli.phono3py.cmd_phono3py import cmd_phono3py  # noqa: E402, F401
from phelel.velph.cli.phonopy.cmd_phonopy import cmd_phonopy  # noqa: E402, F401
from phelel.velph.cli.relax.cmd_relax import cmd_relax  # noqa: E402, F401
from phelel.velph.cli.scheduler.cmd_scheduler import (  # noqa: E402, F401
    cmd_scheduler,
)
from phelel.velph.cli.selfenergy.cmd_selfenergy import (  # noqa: E402, F401
    cmd_selfenergy,
)
from phelel.velph.cli.transport.cmd_transport import cmd_transport  # noqa: E402, F401
