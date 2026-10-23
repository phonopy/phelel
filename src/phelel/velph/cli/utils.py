"""Utilities for cli."""

from __future__ import annotations

import os
import pathlib
import xml.parsers.expat
from collections.abc import Sequence

import click
from phonopy.interface.vasp import VasprunxmlExpat, get_vasp_vca_hint_lines
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon

from phelel.velph.config.schema import InputSetName
from phelel.velph.jobs.project import read_velph_toml
from phelel.velph.jobs.vasp_job import VaspJob, plan_input_set, write_vasp_job

# Option --toml-filename of the commands that read velph.toml.  velph init,
# which writes velph.toml, has an option of the same name of its own.
toml_filename_option = click.option(
    "--toml-filename",
    "toml_filename",
    type=click.Path(),
    default="velph.toml",
    show_default=True,
    help="velph.toml that the command reads.",
)


def write_jobs(jobs: Sequence[VaspJob]) -> None:
    """Write the VASP input files of jobs and show what was written.

    A message is shown first when POTCAR is not copied, because it is not
    found in the current directory.  The hint of VCA is shown once, for the
    cell of the first job, because the species rows of POSCAR are the same
    in all the jobs of a command.

    """
    if "POTCAR" not in jobs[0].copied_files:
        click.echo(
            '"POTCAR" was not found in the current directory, so the VASP input '
            "files are written without POTCAR."
        )
    for job in jobs:
        write_vasp_job(job)
        if job.poscar_source is not None:
            click.echo(
                f'"{job.poscar_source}" was copied to "{job.directory / "POSCAR"}".'
            )
        click.echo(f'VASP input files were written in "{job.directory}".')
    echo_vasp_vca_hint(jobs[0].cell)


def write_input_set_files(
    toml_filename: str | pathlib.Path, name: InputSetName
) -> None:
    """Write the VASP input files of an input set in its directory.

    This is for the input sets whose job needs nothing but velph.toml, such as
    nac, el_bands and el_dos (plan_input_set).

    """
    write_jobs([plan_input_set(read_velph_toml(toml_filename, name), name)])


def echo_vasp_vca_hint(cell: PhonopyAtoms) -> None:
    """Show POSCAR species rows, POTCAR order and VCA tag of site mixture."""
    if cell.has_weighted_species:
        for line in get_vasp_vca_hint_lines(cell):
            click.echo(line)


def get_nac_params(
    nac_cell: PhonopyAtoms,
    vasprun_path: pathlib.Path,
    primitive: PhonopyAtoms,
    is_symmetry: bool,
    symprec: float = 1e-5,
) -> dict:
    """Collect NAC parameters from vasprun.xml and return them.

    nac_cell is the cell of the NAC calculation, given by "cell" of [vasp.nac].
    click.ClickException is raised when vasprun.xml cannot be parsed, for
    example when VASP has not finished writing it.

    """
    with open(vasprun_path, "rb") as f:
        try:
            vasprun = VasprunxmlExpat(f)
            vasprun.parse()
        except xml.parsers.expat.ExpatError as e:
            raise click.ClickException(
                f'"{vasprun_path}" cannot be parsed: {e}.\n'
                f'Check that VASP in "{vasprun_path.parent}" has finished.'
            ) from e

    borns_, epsilon_ = symmetrize_borns_and_epsilon(
        vasprun.born,
        vasprun.epsilon,
        nac_cell,
        primitive=primitive,
        symprec=symprec,
        is_symmetry=is_symmetry,
    )

    nac_params = {
        "born": borns_,
        "factor": get_physical_units().Hartree * get_physical_units().Bohr,
        "dielectric": epsilon_,
    }
    return nac_params


def existing_vaspout(vaspout_filename: str | os.PathLike) -> pathlib.Path:
    """Return the path of vaspout.h5 given by --vaspout-filename.

    click.ClickException is raised when the file does not exist.

    """
    path = pathlib.Path(vaspout_filename)
    if not path.exists():
        raise click.ClickException(
            f'"{path}" not found.  Give the path of vaspout.h5 with --vaspout-filename.'
        )
    return path


def save_or_show_plot(
    plot_filename: str | os.PathLike, save_plot: bool, name: str
) -> None:
    """Save the current figure in plot_filename, or show it, and close it.

    The fonts are embedded in the pdf as TrueType fonts (Type 42).  name is the
    name of the plot in the message, such as "Phonon band structure".

    """
    import matplotlib.pyplot as plt

    plt.tight_layout()
    if save_plot:
        plt.rcParams["pdf.fonttype"] = 42
        plt.savefig(plot_filename)
        click.echo(f'{name} plot was written to "{plot_filename}".')
    else:
        plt.show()
    plt.close()
