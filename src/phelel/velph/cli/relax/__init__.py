"""velph command line tool / velph-relax."""

import pathlib

import click

from phelel.velph.cli import cmd_root
from phelel.velph.cli.relax.generate import write_input_files


@cmd_root.group("relax")
@click.help_option("-h", "--help")
def cmd_relax():
    """Choose relax options."""
    pass


@cmd_relax.command("generate")
@click.argument(
    "toml_filename",
    nargs=1,
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate relax input files."""
    if not pathlib.Path("POTCAR").exists():
        click.echo('"POTCAR" not found in current directory.')

    if not pathlib.Path(toml_filename).exists():
        click.echo(f'"{toml_filename}" not found.', err=True)
        return None

    prev_directory = None
    for i in range(1, 101):
        directory = pathlib.Path(f"relax/iter{i}")
        if directory.exists():
            click.echo(f'"{directory}" exists.')
        else:
            break
        prev_directory = directory

    write_input_files(
        pathlib.Path(toml_filename), directory, prev_directory=prev_directory
    )
