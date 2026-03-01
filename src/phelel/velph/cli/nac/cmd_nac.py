"""velph command line tool / velph-nac."""

from __future__ import annotations

import pathlib

import click

from phelel.velph.cli.cmd_root import cmd_root
from phelel.velph.cli.nac.generate import write_input_files


@cmd_root.group("nac")
@click.help_option("-h", "--help")
def cmd_nac():
    """Choose nac options."""
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
    """Generate relax input files."""
    if not pathlib.Path("POTCAR").exists():
        click.echo('"POTCAR" not found in current directory.')

    write_input_files(pathlib.Path(toml_filename))
