"""velph command line tool / velph-nac."""

from __future__ import annotations

import click

from phelel.velph.cli.utils import write_input_set_files
from phelel.velph.cli.velph_cmd_root import cmd_root


@cmd_root.group("nac")
@click.help_option("-h", "--help")
def cmd_nac():
    """Born effective charges and dielectric constant.

    Write VASP input files of the calculation of Born effective charges and
    dielectric constant for NAC.

    """
    pass


@cmd_nac.command("generate")
@click.option(
    "--toml-filename",
    "toml_filename",
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate nac input files."""
    write_input_set_files(toml_filename, "nac")
