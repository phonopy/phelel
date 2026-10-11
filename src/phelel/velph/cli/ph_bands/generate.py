"""Implementation of velph-ph_bands-generate."""

from __future__ import annotations

import pathlib

import click

from phelel.velph.cli.selfenergy.generate import PHELEL_PARAMS
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


def plan_ph_bands(config: VelphConfig) -> VaspJob:
    """Return the VASP job of ph_bands in the directory "ph_bands".

    The cell is [primitive_cell].  QPOINTS is the band path of
    [vasp.ph_bands.qpoints], or the band path that seekpath gives when it does
    not give "path".  phelel/phelel_params.hdf5 is copied to the directory, and
    click.ClickException is raised when it does not exist.

    """
    if not PHELEL_PARAMS.exists():
        raise click.ClickException(
            f'"{PHELEL_PARAMS}" not found.  It is copied to "ph_bands".\n'
            'Run "velph phelel differentiate" to make it.'
        )
    cell = get_cell(config, "primitive")
    return VaspJob(
        directory=pathlib.Path("ph_bands"),
        poscar=cell,
        incar=add_cell_tags(get_input_set(config, "ph_bands").incar or {}, cell),
        kpoints=get_kpoints(config, "ph_bands", cell=cell),
        scheduler=effective_scheduler(config, "ph_bands"),
        copied_files={**find_potcar(), PHELEL_PARAMS.name: PHELEL_PARAMS},
    )


def write_input_files(toml_filename: pathlib.Path) -> None:
    """Generate VASP inputs of the phonon band structure."""
    config = read_velph_toml(toml_filename, "ph_bands")
    job = plan_ph_bands(config)
    write_vasp_job(job)
    click.echo(f'VASP input files were made in "{job.directory}".')
    echo_vasp_vca_hint(get_cell(config, "primitive"))
