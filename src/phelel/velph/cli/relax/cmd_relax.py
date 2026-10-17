"""velph command line tool / velph-relax."""

from __future__ import annotations

import itertools
import pathlib

import click

from phelel.velph.cli.relax.generate import write_input_files
from phelel.velph.cli.velph_cmd_root import cmd_root


@cmd_root.group("relax")
@click.help_option("-h", "--help")
def cmd_relax():
    """Choose relax options."""
    pass


@cmd_relax.command("generate")
@click.option(
    "--toml-filename",
    "toml_filename",
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate relax input files."""
    if not pathlib.Path("POTCAR").exists():
        click.echo('"POTCAR" not found in current directory.')

    prev_directory = None
    for i in itertools.count(1):
        directory = pathlib.Path(f"relax/iter{i}")
        if not directory.exists():
            break
        click.echo(f'"{directory}" exists.')
        prev_directory = directory

    write_input_files(
        pathlib.Path(toml_filename), directory, prev_directory=prev_directory
    )
