"""Implementation of velph-el_dos-generate."""

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


def plan_el_dos(project: Project) -> VaspJob:
    """Return the VASP job of el_dos in the directory "el_dos".

    The cell is [primitive_cell].  KPOINTS_OPT is the dense mesh of
    [vasp.el_dos.kpoints_dense], on which VASP computes the DOS and the Fermi
    level.

    """
    cell = project.cell_of("el_dos")
    return VaspJob(
        directory=pathlib.Path("el_dos"),
        cell=cell,
        incar=add_cell_tags(project.input_set("el_dos").incar or {}, cell),
        kpoints=project.kpoints("el_dos"),
        job_script=job_script(project, "el_dos"),
        copied_files=find_potcar(),
    )


def write_input_files(toml_filename: pathlib.Path) -> None:
    """Generate VASP inputs of the electronic DOS."""
    job = plan_el_dos(read_velph_toml(toml_filename, "el_dos"))
    write_vasp_job(job)
    click.echo(f'VASP input files were written in "{job.directory}".')
    echo_vasp_vca_hint(job.cell)
