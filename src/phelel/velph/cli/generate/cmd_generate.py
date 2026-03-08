"""velph command line tool / velph-generate."""

from __future__ import annotations

import pathlib

import click
import tomli
from phonopy.interface.calculator import write_crystal_structure
from phonopy.structure.atoms import parse_cell_dict

from phelel.velph.cli.cmd_root import cmd_root


#
# velph generate
#
@cmd_root.command("generate")
@click.option(
    "-f",
    "toml_filename",
    type=click.Path(exists=True),
    default="velph.toml",
    show_default=True,
    help="Specify velph.toml",
)
@click.option(
    "--prefix",
    "prefix",
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


def _write_cell(filename: str, toml_cell_dict: dict):
    if pathlib.Path(filename).exists():
        click.echo(f'"{filename}" was not overwritten because it exists.', err=True)
    else:
        cell = parse_cell_dict(toml_cell_dict)
        if cell is None:
            click.echo(
                f'"{filename}" was not generated because of invalid cell data.',
                err=True,
            )
            return
        write_crystal_structure(filename, cell)
        click.echo(f'"{filename}" was generated.')
