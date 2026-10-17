"""Implementation of velph-el_bands-generate."""

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


def plan_el_bands(project: Project) -> VaspJob:
    """Return the VASP job of el_bands in the directory "el_bands".

    The cell is [primitive_cell].  KPOINTS_OPT is the band path of
    [vasp.el_bands.kpoints_opt], or the band path that seekpath gives when it
    does not give "path".

    """
    cell = project.cell_of("el_bands")
    return VaspJob(
        directory=pathlib.Path("el_bands"),
        cell=cell,
        incar=add_cell_tags(project.input_set("el_bands").incar or {}, cell),
        kpoints=project.kpoints("el_bands"),
        job_script=job_script(project, "el_bands"),
        copied_files=find_potcar(),
    )


def write_input_files(toml_filename: pathlib.Path) -> None:
    """Generate VASP inputs of the electronic band structure."""
    job = plan_el_bands(read_velph_toml(toml_filename, "el_bands"))
    write_vasp_job(job)
    click.echo(f'VASP input files were written in "{job.directory}".')
    echo_vasp_vca_hint(job.cell)
