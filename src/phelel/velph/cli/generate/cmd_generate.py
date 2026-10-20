"""velph command line tool / velph-generate."""

from __future__ import annotations

import pathlib

import click
from phonopy.interface.vasp import write_vasp
from phonopy.structure.atoms import PhonopyAtoms

from phelel.velph.cli.utils import echo_vasp_vca_hint, toml_filename_option
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.jobs.project import read_velph_toml


#
# velph generate
#
@cmd_root.command("generate")
@toml_filename_option
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
    project = read_velph_toml(toml_filename)
    _write_cell(f"{prefix}-unitcell", project.unitcell)
    _write_cell(f"{prefix}-primitive", project.primitive)


def _write_cell(filename: str, cell: PhonopyAtoms):
    if pathlib.Path(filename).exists():
        click.echo(f'"{filename}" was not overwritten because it exists.', err=True)
        return
    write_vasp(filename, cell, for_vca=cell.has_weighted_species)
    click.echo(f'"{filename}" was written.')
    echo_vasp_vca_hint(cell)
