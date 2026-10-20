"""Implementation of velph-ph_bands-generate."""

from __future__ import annotations

import pathlib

import click

from phelel.velph.cli.utils import write_jobs
from phelel.velph.jobs.project import Project, read_velph_toml
from phelel.velph.jobs.project_directory import ProjectDirectory
from phelel.velph.jobs.vasp_job import VaspJob, plan_input_set


def plan_ph_bands(
    project: Project, project_dir: ProjectDirectory | None = None
) -> VaspJob:
    """Return the VASP job of ph_bands in the directory "ph_bands".

    The cell is [primitive_cell].  QPOINTS is the band path of
    [vasp.ph_bands.qpoints], or the band path that seekpath gives when it does
    not give "path".  phelel/phelel_params.hdf5 is copied to the directory, and
    click.ClickException is raised when it does not exist.

    """
    if project_dir is None:
        project_dir = ProjectDirectory()
    params = project_dir.phelel_params
    if not params.exists():
        raise click.ClickException(
            f'"{params}" not found.  It is copied to "ph_bands".\n'
            'Run "velph phelel differentiate" to make it.'
        )
    return plan_input_set(
        project, "ph_bands", project_dir=project_dir, extra_files={params.name: params}
    )


def write_input_files(toml_filename: pathlib.Path) -> None:
    """Generate VASP inputs of the phonon band structure."""
    write_jobs([plan_ph_bands(read_velph_toml(toml_filename, "ph_bands"))])
