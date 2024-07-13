"""velph command line tool / velph-generate."""

import pathlib

import click
import tomli
from phonopy.interface.calculator import write_crystal_structure
from phonopy.structure.atoms import parse_cell_dict

from phelel.velph.cli import cmd_root


#
# velph generate
#
@cmd_root.command("generate")
@click.option(
    "-f",
    "toml_filename",
    nargs=1,
    type=click.Path(exists=True),
    default="velph.toml",
    show_default=True,
    help="Specify velph.toml",
)
@click.option(
    "--prefix",
    nargs=1,
    type=click.Path(),
    default="POSCAR",
    show_default=True,
    help="{prefix}-unitcell, {prefix}-primitive",
)
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str, prefix: str):
    """Write POSCAR-unitcell and POSCAR-primitive.

    Filename prefix "POSCAR" can be replaced using the "prefix" option.

    """
    _run_generate(toml_filename, prefix)


def _run_generate(toml_filename: str, prefix: str) -> None:
    """Generate {prefix}-unitcell and {prefix}-primitive."""
    with open(toml_filename, "rb") as f:
        toml_dict = tomli.load(f)

    filename = f"{prefix}-unitcell"
    _write_cell(filename, toml_dict["unitcell"])
    if "primitive_cell" in toml_dict:
        filename = f"{prefix}-primitive"
        _write_cell(filename, toml_dict["primitive_cell"])


def _write_cell(filename, toml_cell_dict):
    if pathlib.Path(filename).exists():
        click.echo(f'"{filename}" was not overwritten because it exists.', err=True)
    else:
        write_crystal_structure(filename, parse_cell_dict(toml_cell_dict))
        click.echo(f'"{filename}" was generated.', err=True)
