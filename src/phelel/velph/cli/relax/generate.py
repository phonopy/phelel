"""Implementation of velph-relax-generate."""

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


def plan_relax(
    config: VelphConfig,
    directory: pathlib.Path,
    prev_directory: pathlib.Path | None = None,
) -> VaspJob:
    """Return the VASP job of relax in directory.

    With prev_directory, CONTCAR of the previous relax is used as POSCAR, and
    click.ClickException is raised when it does not exist.  The directory
    name, such as "iter2", is appended to job_name of the scheduler settings.

    """
    cell = _get_relax_cell(config)
    poscar: PhonopyAtoms | pathlib.Path = cell
    if prev_directory is not None:
        poscar = prev_directory / "CONTCAR"
        if not poscar.exists():
            raise click.ClickException(
                f'"{poscar}" not found.  It is used as POSCAR of "{directory}".'
            )
    return VaspJob(
        directory=directory,
        poscar=poscar,
        incar=add_cell_tags(get_input_set(config, "relax").incar or {}, cell),
        kpoints=get_kpoints(config, "relax"),
        job_script=job_script(config, "relax", job_id=directory.name),
        copied_files=find_potcar(),
    )


def write_input_files(
    toml_filename: pathlib.Path,
    directory: pathlib.Path,
    prev_directory: pathlib.Path | None = None,
) -> None:
    """Generate VASP relax inputs."""
    config = read_velph_toml(toml_filename, "relax")
    job = plan_relax(config, directory, prev_directory=prev_directory)
    write_vasp_job(job)
    if prev_directory is not None:
        click.echo(f'"{job.poscar}" was copied to "{directory / "POSCAR"}".')
    click.echo(f'VASP input files were written in "{directory}".')
    echo_vasp_vca_hint(_get_relax_cell(config))


def _get_relax_cell(config: VelphConfig) -> PhonopyAtoms:
    """Return the cell given by "cell" of [vasp.relax]."""
    cell_choice = get_input_set(config, "relax").cell
    assert cell_choice is not None
    return get_cell(config, cell_choice)
