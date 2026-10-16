"""Implementation of velph-el_bands-generate."""

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


def plan_el_bands(config: VelphConfig) -> VaspJob:
    """Return the VASP job of el_bands in the directory "el_bands".

    The cell is [primitive_cell].  KPOINTS_OPT is the band path of
    [vasp.el_bands.kpoints_opt], or the band path that seekpath gives when it
    does not give "path".

    """
    cell = get_cell(config, "primitive")
    return VaspJob(
        directory=pathlib.Path("el_bands"),
        poscar=cell,
        incar=add_cell_tags(get_input_set(config, "el_bands").incar or {}, cell),
        kpoints=get_kpoints(config, "el_bands", cell=cell),
        scheduler=effective_scheduler(config, "el_bands"),
        copied_files=find_potcar(),
    )


def write_input_files(toml_filename: pathlib.Path) -> None:
    """Generate VASP inputs of the electronic band structure."""
    config = read_velph_toml(toml_filename, "el_bands")
    job = plan_el_bands(config)
    write_vasp_job(job)
    click.echo(f'VASP input files were generated in "{job.directory}".')
    echo_vasp_vca_hint(get_cell(config, "primitive"))
