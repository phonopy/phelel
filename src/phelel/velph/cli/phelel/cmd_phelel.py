"""velph command line tool / velph-phelel."""

from __future__ import annotations

import pathlib

import click

import phelel
from phelel.cui.phelel_script import finalize_phelel
from phelel.velph.cli.phelel.differentiate import get_fft_mesh, run_derivatives
from phelel.velph.cli.phelel.init import run_init
from phelel.velph.cli.phelel.phonopy import create_phonopy_yaml
from phelel.velph.cli.supercells import write_supercell_input_files
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.jobs.project import read_velph_toml
from phelel.velph.jobs.project_directory import ProjectDirectory


@cmd_root.group("phelel")
@click.help_option("-h", "--help")
def cmd_phelel():
    """Supercells and derivatives of phelel.

    Make the supercells of phelel, write their VASP input files and compute the
    derivatives.

    """
    pass


#
# velph phelel init
#
@cmd_phelel.command("init")
@click.option(
    "--toml-filename",
    "toml_filename",
    type=click.Path(),
    default="velph.toml",
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
def cmd_init(toml_filename: str, phonon_from_phonopy: bool):
    """Generate displacements and write phelel_disp.yaml."""
    project = read_velph_toml(toml_filename)
    phe = run_init(project, phonon_from_phonopy=phonon_from_phonopy)

    phelel_yaml_filename = ProjectDirectory().disp_yaml("phelel")
    phelel_yaml_filename.parent.mkdir(parents=True, exist_ok=True)
    finalize_phelel(
        phe,
        displacements_mode=True,
        filename=phelel_yaml_filename,
        sys_exit_after_finalize=False,
    )

    click.echo(f'"{phelel_yaml_filename}" was written.')
    click.echo('Run "velph phelel generate" to write the VASP input files.')


#
# velph phelel generate
#
@cmd_phelel.command("generate")
@click.option(
    "--toml-filename",
    "toml_filename",
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate phelel input files."""
    write_supercell_input_files(toml_filename, "phelel")


#
# velph phelel differentiate
#
@cmd_phelel.command("differentiate")
@click.option(
    "--toml-filename",
    "toml_filename",
    type=click.Path(),
    default="velph.toml",
)
@click.option(
    "--encut",
    type=float,
    default=None,
    help=(
        "ENCUT in eV for the FFT mesh of the local potentials, in place of encut "
        "of the electron-phonon input sets of velph.toml."
    ),
)
@click.option(
    "-v",
    "verbose",
    is_flag=True,
    default=False,
    help=("More information will be printed."),
)
@click.help_option("-h", "--help")
def cmd_differentiate(
    toml_filename: str,
    encut: float | None,
    verbose: bool,
) -> None:
    """Calculate derivatives and write phelel_params.hdf5.

    The FFT mesh of the local potentials is that of VASP for ENCUT and PREC of
    the electron-phonon input sets of velph.toml, or for --encut.

    """
    project_dir = ProjectDirectory()
    hdf5_filename = project_dir.phelel_params
    yaml_filename = project_dir.disp_yaml("phelel")

    project = read_velph_toml(toml_filename)
    is_symmetry = not project.code_section("phelel").nosym
    if not is_symmetry:
        click.echo(
            'Found "nosym = true" in [phelel] section. Symmetrization is turned off.'
        )

    phe = phelel.load(yaml_filename, is_symmetry=is_symmetry)
    phe.fft_mesh = get_fft_mesh(project, phe.primitive.cell, encut=encut)
    if encut is None:
        click.echo(f"FFT mesh: {phe.fft_mesh}.")
    else:
        click.echo(f"FFT mesh: {phe.fft_mesh} (encut={encut}).")

    run_derivatives(phe, project_dir=project_dir, verbose=verbose)
    hdf5_filename.parent.mkdir(parents=True, exist_ok=True)
    phe.save_hdf5(filename=hdf5_filename)
    click.echo(f'"{hdf5_filename}" was written.')


#
# velph phelel phonopy
#
@cmd_phelel.command("phonopy")
@click.option(
    "--toml-filename",
    "toml_filename",
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_phonopy(toml_filename: str):
    """Create phonopy_params.yaml."""
    create_phonopy_yaml(pathlib.Path(toml_filename))
