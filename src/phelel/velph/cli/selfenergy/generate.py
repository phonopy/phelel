"""Generate the VASP inputs of the electron-phonon input sets.

selfenergy, transport and ph_selfenergy are the electron-phonon input sets.
Their VASP calculations are run on the primitive cell with
phelel/phelel_params.hdf5, which velph phelel differentiate makes.

"""

from __future__ import annotations

import pathlib
from typing import Literal

import click

from phelel.velph.cli.utils import echo_vasp_vca_hint
from phelel.velph.jobs.project import Project, read_velph_toml
from phelel.velph.jobs.vasp_job import (
    VaspJob,
    add_cell_tags,
    find_potcar,
    job_script,
    write_vasp_job,
)

ElphInputSetName = Literal["selfenergy", "transport", "ph_selfenergy"]

PHELEL_PARAMS = pathlib.Path("phelel/phelel_params.hdf5")


def plan_elph(
    project: Project, set_name: ElphInputSetName, dry_run: bool = False
) -> VaspJob:
    """Return the VASP job of an electron-phonon input set.

    The directory is named after the input set, and the cell is
    [primitive_cell].  phelel/phelel_params.hdf5 is copied to the directory,
    and click.ClickException is raised when it does not exist.  With dry_run,
    NELM = 0 and ELPH_RUN = False are added to the INCAR, so that VASP stops
    before the electron-phonon calculation, and phelel_params.hdf5 is neither
    needed nor copied.

    """
    cell = project.cell_of(set_name)
    incar = dict(project.input_set(set_name).incar or {})
    copied_files = find_potcar()
    if dry_run:
        incar.update({"nelm": 0, "elph_run": False})
    else:
        if not PHELEL_PARAMS.exists():
            raise click.ClickException(
                f'"{PHELEL_PARAMS}" not found.  It is copied to "{set_name}".\n'
                'Run "velph phelel differentiate" to make it, or run this command '
                "with --dry-run."
            )
        copied_files[PHELEL_PARAMS.name] = PHELEL_PARAMS
    return VaspJob(
        directory=pathlib.Path(set_name),
        cell=cell,
        incar=add_cell_tags(incar, cell),
        kpoints=project.kpoints(set_name),
        job_script=job_script(project, set_name),
        copied_files=copied_files,
    )


def write_input_files(
    toml_filename: pathlib.Path, set_name: ElphInputSetName, dry_run: bool
) -> None:
    """Generate VASP inputs of an electron-phonon input set."""
    job = plan_elph(read_velph_toml(toml_filename, set_name), set_name, dry_run=dry_run)
    write_vasp_job(job)
    click.echo(f'VASP input files were written in "{job.directory}".')
    echo_vasp_vca_hint(job.cell)
