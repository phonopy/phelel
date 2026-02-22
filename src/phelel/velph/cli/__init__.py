"""velph command line tool module."""

from phelel.velph.cli.cmd_root import cmd_root
from phelel.velph.cli.el_bands.cmd_el_bands import cmd_el_bands
from phelel.velph.cli.generate.cmd_generate import cmd_generate
from phelel.velph.cli.init.cmd_init import cmd_init
from phelel.velph.cli.nac.cmd_nac import cmd_nac
from phelel.velph.cli.ph_bands.cmd_ph_bands import cmd_ph_bands
from phelel.velph.cli.ph_selfenergy.cmd_ph_selfenergy import cmd_ph_selfenergy
from phelel.velph.cli.phelel.cmd_phelel import cmd_phelel
from phelel.velph.cli.phono3py.cmd_phono3py import cmd_phono3py
from phelel.velph.cli.phonopy.cmd_phonopy import cmd_phonopy
from phelel.velph.cli.relax.cmd_relax import cmd_relax
from phelel.velph.cli.selfenergy.cmd_selfenergy import cmd_selfenergy
from phelel.velph.cli.transport.cmd_transport import cmd_transport

__all__ = [
    "cmd_el_bands",  # velph el-bands <subcommand>
    "cmd_generate",  # velph generate
    "cmd_init",  # velph init
    "cmd_nac",  # velph nac <subcommand>
    "cmd_phelel",  # velph phelel <subcommand>
    "cmd_phonopy",  # velph phonopy <subcommand>
    "cmd_phono3py",  # velph phono3py <subcommand>
    "cmd_ph_bands",  # velph ph-bands <subcommand>
    "cmd_ph_selfenergy",  # velph ph-selfenergy <subcommand>
    "cmd_relax",  # velph relax <subcommand>
    "cmd_root",  # velph <subcommand>
    "cmd_selfenergy",  # velph selfenergy <subcommand>
    "cmd_transport",  # velph transport <subcommand>
]
