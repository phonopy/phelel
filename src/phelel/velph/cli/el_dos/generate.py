"""Implementation of velph-el_dos-generate."""

from __future__ import annotations

import pathlib

import click

from phelel.velph.cli.utils import echo_vasp_vca_hint
from phelel.velph.config.schema import VelphConfig
from phelel.velph.jobs.vasp_job import (
    VaspJob,
    add_cell_tags,
    effective_scheduler,
    find_potcar,
    get_cell,
    get_input_set,
    get_kpoints,
    read_velph_toml,
    write_vasp_job,
)


def plan_el_dos(config: VelphConfig) -> VaspJob:
    """Return the VASP job of el_dos in the directory "el_dos".

    The cell is [primitive_cell].  KPOINTS_OPT is the dense mesh of
    [vasp.el_dos.kpoints_dense], on which VASP computes the DOS and the Fermi
    level.

    """
    cell = get_cell(config, "primitive")
    return VaspJob(
        directory=pathlib.Path("el_dos"),
        poscar=cell,
        incar=add_cell_tags(get_input_set(config, "el_dos").incar or {}, cell),
        kpoints=get_kpoints(config, "el_dos"),
        scheduler=effective_scheduler(config, "el_dos"),
        copied_files=find_potcar(),
    )


def write_input_files(toml_filename: pathlib.Path) -> None:
    """Generate VASP inputs of the electronic DOS."""
    config = read_velph_toml(toml_filename, "el_dos")
    job = plan_el_dos(config)
    write_vasp_job(job)
    click.echo(f'VASP input files were generated in "{job.directory}".')
    echo_vasp_vca_hint(get_cell(config, "primitive"))
