"""Implementation of velph-ph_bands-generate."""

from __future__ import annotations

import pathlib

import click

from phelel.velph.cli.selfenergy.generate import PHELEL_PARAMS
from phelel.velph.cli.utils import echo_vasp_vca_hint
from phelel.velph.jobs.project import Project, read_velph_toml
from phelel.velph.jobs.vasp_job import (
    VaspJob,
    add_cell_tags,
    find_potcar,
    job_script,
    write_vasp_job,
)


def plan_ph_bands(project: Project) -> VaspJob:
    """Return the VASP job of ph_bands in the directory "ph_bands".

    The cell is [primitive_cell].  QPOINTS is the band path of
    [vasp.ph_bands.qpoints], or the band path that seekpath gives when it does
    not give "path".  phelel/phelel_params.hdf5 is copied to the directory, and
    click.ClickException is raised when it does not exist.

    """
    if not PHELEL_PARAMS.exists():
        raise click.ClickException(
            f'"{PHELEL_PARAMS}" not found.  It is copied to "ph_bands".\n'
            'Run "velph phelel differentiate" to make it.'
        )
    cell = project.cell_of("ph_bands")
    return VaspJob(
        directory=pathlib.Path("ph_bands"),
        cell=cell,
        incar=add_cell_tags(project.input_set("ph_bands").incar or {}, cell),
        kpoints=project.kpoints("ph_bands"),
        job_script=job_script(project, "ph_bands"),
        copied_files={**find_potcar(), PHELEL_PARAMS.name: PHELEL_PARAMS},
    )


def write_input_files(toml_filename: pathlib.Path) -> None:
    """Generate VASP inputs of the phonon band structure."""
    job = plan_ph_bands(read_velph_toml(toml_filename, "ph_bands"))
    write_vasp_job(job)
    click.echo(f'VASP input files were written in "{job.directory}".')
    echo_vasp_vca_hint(job.cell)
