"""velph command line tool / velph-selfenergy."""

import pathlib

import click

from phelel.velph.cli.selfenergy.generate import write_input_files
from phelel.velph.cli.utils import toml_filename_option
from phelel.velph.cli.velph_cmd_root import cmd_root


@cmd_root.group("selfenergy")
@click.help_option("-h", "--help")
def cmd_selfenergy():
    """Electron self-energy.

    Write VASP input files of the electron self-energy calculation.

    """
    pass


@cmd_selfenergy.command("generate")
@toml_filename_option
@click.option(
    "--dry-run/--no-dry-run",
    "-d",
    "dry_run",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str, dry_run: bool):
    """Generate elph input files."""
    write_input_files(pathlib.Path(toml_filename), "selfenergy", dry_run)
