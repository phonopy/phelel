"""velph command line tool / velph-phono3py."""

from __future__ import annotations

import click

from phelel.velph.cli.phono3py.init import run_init
from phelel.velph.cli.supercells import write_supercell_input_files
from phelel.velph.cli.utils import toml_filename_option
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.jobs.project import read_velph_toml
from phelel.velph.jobs.project_directory import ProjectDirectory


@cmd_root.group("phono3py")
@click.help_option("-h", "--help")
def cmd_phono3py():
    """Supercells of phono3py.

    Make the supercells of phono3py and write their VASP input files.

    """
    pass


#
# velph phono3py init
#
@cmd_phono3py.command("init")
@toml_filename_option
@click.option(
    "--rd",
    "random_displacements",
    default=None,
    type=int,
    help="Number of snapshots of supercells with random directional displacement.",
)
@click.option(
    "--phonon-from-phonopy",
    "phonon_from_phonopy",
    is_flag=True,
    default=False,
    help=(
        "Take the phonon supercells from "
        f'{ProjectDirectory().disp_yaml("phonopy")} written by "velph phonopy init".'
    ),
)
@click.help_option("-h", "--help")
def cmd_init(
    toml_filename: str,
    random_displacements: int | None,
    phonon_from_phonopy: bool,
):
    """Generate displacements and write phono3py_disp.yaml."""
    ph3 = run_init(
        read_velph_toml(toml_filename),
        number_of_snapshots=random_displacements,
        phonon_from_phonopy=phonon_from_phonopy,
    )

    phono3py_yaml_filename = ProjectDirectory().disp_yaml("phono3py")
    phono3py_yaml_filename.parent.mkdir(parents=True, exist_ok=True)
    ph3.save(phono3py_yaml_filename)

    click.echo(f'"{phono3py_yaml_filename}" was written.')
    click.echo('Run "velph phono3py generate" to write the VASP input files.')


#
# velph phono3py generate
#
@cmd_phono3py.command("generate")
@toml_filename_option
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate phono3py input files."""
    write_supercell_input_files(toml_filename, "phono3py")
