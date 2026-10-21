"""velph command line tool / velph-relax."""

from __future__ import annotations

import itertools
import pathlib

import click

from phelel.velph.cli.relax.generate import write_input_files
from phelel.velph.cli.utils import toml_filename_option
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.jobs.project_directory import ProjectDirectory


@cmd_root.group("relax")
@click.help_option("-h", "--help")
def cmd_relax():
    """Relaxation of the crystal structure.

    Write VASP input files of the relaxation of the crystal structure.

    """
    pass


@cmd_relax.command("generate")
@toml_filename_option
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate relax input files."""
    project_dir = ProjectDirectory()
    prev_directory = None
    for i in itertools.count(1):
        directory = project_dir.relax_directory(i)
        if not directory.exists():
            break
        click.echo(f'"{directory}" exists.')
        prev_directory = directory

    write_input_files(
        pathlib.Path(toml_filename), directory, prev_directory=prev_directory
    )
