"""Implementation of velph-relax-generate."""

from __future__ import annotations

import pathlib

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


def plan_relax(
    project: Project,
    directory: pathlib.Path,
    prev_directory: pathlib.Path | None = None,
) -> VaspJob:
    """Return the VASP job of relax in directory.

    The cell is the one given by "cell" of [vasp.relax].  With prev_directory,
    CONTCAR of the previous relax is used as POSCAR, and click.ClickException
    is raised when it does not exist.  The directory name, such as "iter2", is
    appended to job_name of the scheduler settings.

    """
    cell = project.cell_of("relax")
    poscar_source = None
    if prev_directory is not None:
        poscar_source = prev_directory / "CONTCAR"
        if not poscar_source.exists():
            raise click.ClickException(
                f'"{poscar_source}" not found.  It is used as POSCAR of "{directory}".'
            )
    return VaspJob(
        directory=directory,
        cell=cell,
        incar=add_cell_tags(project.input_set("relax").incar or {}, cell),
        kpoints=project.kpoints("relax"),
        job_script=job_script(project, "relax", job_id=directory.name),
        copied_files=find_potcar(),
        poscar_source=poscar_source,
    )


def write_input_files(
    toml_filename: pathlib.Path,
    directory: pathlib.Path,
    prev_directory: pathlib.Path | None = None,
) -> None:
    """Generate VASP relax inputs."""
    project = read_velph_toml(toml_filename, "relax")
    job = plan_relax(project, directory, prev_directory=prev_directory)
    write_vasp_job(job)
    if job.poscar_source is not None:
        click.echo(f'"{job.poscar_source}" was copied to "{directory / "POSCAR"}".')
    click.echo(f'VASP input files were written in "{directory}".')
    echo_vasp_vca_hint(job.cell)
