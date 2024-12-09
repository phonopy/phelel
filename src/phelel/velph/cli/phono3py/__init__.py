"""velph command line tool / velph-supercell."""

from __future__ import annotations

import pathlib
from typing import Optional

import click
import tomli

from phelel.velph.cli import cmd_root
from phelel.velph.cli.phono3py.generate import write_supercell_input_files
from phelel.velph.cli.phono3py.init import run_init


@cmd_root.group("phono3py")
@click.help_option("-h", "--help")
def cmd_phono3py():
    """Choose phono3py options."""
    pass


#
# velph phono3py init
#
@cmd_phono3py.command("init")
@click.argument(
    "toml_filename",
    nargs=1,
    type=click.Path(),
    default="velph.toml",
)
@click.option(
    "--rd",
    "random_displacements",
    nargs=1,
    default=None,
    type=int,
    help="Number of snapshots of supercells with random directional displacement.",
)
@click.option(
    "--rd-fc2",
    "random_displacements_fc2",
    nargs=1,
    default=None,
    type=int,
    help=(
        "Number of snapshots of phonon supercells "
        "with random directional displacement."
    ),
)
@click.help_option("-h", "--help")
def cmd_init(
    toml_filename: str,
    random_displacements: Optional[int],
    random_displacements_fc2: Optional[int],
):
    """Generate displacements and write phelel_disp.yaml."""
    with open(toml_filename, "rb") as f:
        toml_dict = tomli.load(f)

    ph3py = run_init(
        toml_dict,
        number_of_snapshots=random_displacements,
        number_of_snapshots_fc2=random_displacements_fc2,
    )

    phono3py_yaml_filename = pathlib.Path("phono3py/phono3py_disp.yaml")
    phono3py_yaml_filename.parent.mkdir(parents=True, exist_ok=True)
    ph3py.save(phono3py_yaml_filename)

    click.echo(f'"{phono3py_yaml_filename}" was generated. ')
    click.echo('VASP input files will be generated by "velph phono3py generate".')


#
# velph phono3py generate
#
@cmd_phono3py.command("generate")
@click.argument(
    "toml_filename",
    nargs=1,
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate phono3py input files."""
    yaml_filename = "phono3py/phono3py_disp.yaml"
    if not pathlib.Path("POTCAR").exists():
        click.echo('"POTCAR" not found in current directory.')

    write_supercell_input_files(
        pathlib.Path(toml_filename), pathlib.Path(yaml_filename)
    )
