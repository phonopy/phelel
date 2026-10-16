"""Implementation of velph-nac-generate."""

from __future__ import annotations

import pathlib

import click
from phonopy.structure.atoms import PhonopyAtoms

from phelel.velph.cli.utils import echo_vasp_vca_hint
from phelel.velph.config.schema import VelphConfig
from phelel.velph.jobs.vasp_job import (
    VaspJob,
    add_cell_tags,
    find_potcar,
    get_cell,
    get_input_set,
    get_kpoints,
    job_script,
    read_velph_toml,
    write_vasp_job,
)


def plan_nac(config: VelphConfig) -> VaspJob:
    """Return the VASP job of nac in the directory "nac".

    The cell is the one given by "cell" of [vasp.nac].

    """
    input_set = get_input_set(config, "nac")
    assert input_set.cell is not None
    cell = get_cell(config, input_set.cell)
    return VaspJob(
        directory=pathlib.Path("nac"),
        poscar=cell,
        incar=add_cell_tags(input_set.incar or {}, cell),
        kpoints=get_kpoints(config, "nac"),
        job_script=job_script(config, "nac"),
        copied_files=find_potcar(),
    )


def write_input_files(toml_filename: pathlib.Path) -> None:
    """Generate VASP nac inputs."""
    config = read_velph_toml(toml_filename, "nac")
    job = plan_nac(config)
    write_vasp_job(job)
    click.echo(f'VASP input files were generated in "{job.directory}".')
    assert isinstance(job.poscar, PhonopyAtoms)
    echo_vasp_vca_hint(job.poscar)
