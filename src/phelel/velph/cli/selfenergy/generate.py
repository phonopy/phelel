"""Generate the VASP inputs of the electron-phonon input sets.

selfenergy, transport and ph_selfenergy are the electron-phonon input sets.
Their VASP calculations are run on the primitive cell with
phelel/phelel_params.hdf5, which velph phelel differentiate makes.

"""

from __future__ import annotations

import pathlib

import click

from phelel.velph.cli.utils import write_jobs
from phelel.velph.config.input_sets import ElphInputSetName
from phelel.velph.jobs.project import Project, read_velph_toml
from phelel.velph.jobs.project_directory import ProjectDirectory
from phelel.velph.jobs.vasp_job import VaspJob, plan_input_set


def plan_elph(
    project: Project,
    set_name: ElphInputSetName,
    dry_run: bool = False,
    project_dir: ProjectDirectory | None = None,
) -> VaspJob:
    """Return the VASP job of an electron-phonon input set.

    The directory is named after the input set, and the cell is
    [primitive_cell].  phelel/phelel_params.hdf5 is copied to the directory,
    and click.ClickException is raised when it does not exist.  With dry_run,
    NELM = 0 and ELPH_RUN = False are added to the INCAR, so that VASP stops
    before the electron-phonon calculation, and phelel_params.hdf5 is neither
    needed nor copied.

    """
    if project_dir is None:
        project_dir = ProjectDirectory()
    if dry_run:
        return plan_input_set(
            project,
            set_name,
            project_dir=project_dir,
            incar_update={"nelm": 0, "elph_run": False},
        )
    params = project_dir.phelel_params
    if not params.exists():
        raise click.ClickException(
            f'"{params}" not found.  It is copied to "{set_name}".\n'
            'Run "velph phelel differentiate" to make it, or run this command '
            "with --dry-run."
        )
    return plan_input_set(
        project, set_name, project_dir=project_dir, extra_files={params.name: params}
    )


def write_input_files(
    toml_filename: pathlib.Path, set_name: ElphInputSetName, dry_run: bool
) -> None:
    """Generate VASP inputs of an electron-phonon input set."""
    project = read_velph_toml(toml_filename, set_name)
    write_jobs([plan_elph(project, set_name, dry_run=dry_run)])
