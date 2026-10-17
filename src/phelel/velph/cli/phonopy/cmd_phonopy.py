"""velph command line tool / velph-phonopy."""

from __future__ import annotations

import pathlib

import click

from phelel.velph.cli.phonopy.generate import write_supercell_input_files
from phelel.velph.cli.phonopy.init import run_init
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.jobs.vasp_job import read_velph_toml


@cmd_root.group("phonopy")
@click.help_option("-h", "--help")
def cmd_phonopy():
    """Supercells of phonopy.

    Make the supercells of phonopy and write their VASP input files.

    """
    pass


#
# velph phonopy init
#
@cmd_phonopy.command("init")
@click.option(
    "--toml-filename",
    "toml_filename",
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_init(toml_filename: str):
    """Generate displacements and write phonopy_disp.yaml."""
    config = read_velph_toml(toml_filename)
    ph = run_init(config, toml_filename=toml_filename)

    phonopy_yaml_filename = pathlib.Path("phonopy/phonopy_disp.yaml")
    phonopy_yaml_filename.parent.mkdir(parents=True, exist_ok=True)
    ph.save(phonopy_yaml_filename)

    click.echo(f'"{phonopy_yaml_filename}" was written.')
    click.echo('Run "velph phonopy generate" to write the VASP input files.')


#
# velph phonopy generate
#
@cmd_phonopy.command("generate")
@click.option(
    "--toml-filename",
    "toml_filename",
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate phonopy input files."""
    yaml_filename = "phonopy/phonopy_disp.yaml"
    if not pathlib.Path("POTCAR").exists():
        click.echo('"POTCAR" not found in current directory.')

    write_supercell_input_files(
        pathlib.Path(toml_filename), pathlib.Path(yaml_filename)
    )
