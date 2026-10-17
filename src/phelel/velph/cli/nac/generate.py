"""Implementation of velph-nac-generate."""

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


def plan_nac(project: Project) -> VaspJob:
    """Return the VASP job of nac in the directory "nac".

    The cell is the one given by "cell" of [vasp.nac].

    """
    cell = project.cell_of("nac")
    return VaspJob(
        directory=pathlib.Path("nac"),
        cell=cell,
        incar=add_cell_tags(project.input_set("nac").incar or {}, cell),
        kpoints=project.kpoints("nac"),
        job_script=job_script(project, "nac"),
        copied_files=find_potcar(),
    )


def write_input_files(toml_filename: pathlib.Path) -> None:
    """Generate VASP nac inputs."""
    job = plan_nac(read_velph_toml(toml_filename, "nac"))
    write_vasp_job(job)
    click.echo(f'VASP input files were written in "{job.directory}".')
    echo_vasp_vca_hint(job.cell)
