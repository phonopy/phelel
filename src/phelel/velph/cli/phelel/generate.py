"""Implementation of velph-phelel-generate."""

from __future__ import annotations

import pathlib

import click

from phelel.velph.cli.supercells import (
    load_phelel_yaml_as_phonopy,
    plan_supercells,
)
from phelel.velph.cli.utils import echo_vasp_vca_hint
from phelel.velph.jobs.project import read_velph_toml
from phelel.velph.jobs.vasp_job import write_vasp_job


def write_supercell_input_files(
    toml_filename: pathlib.Path, yaml_filename: pathlib.Path
) -> None:
    """Generate VASP inputs of the supercells written in yaml_filename."""
    if not yaml_filename.exists():
        raise click.ClickException(
            f'"{yaml_filename}" not found.\nRun "velph phelel init" to make it.'
        )
    project = read_velph_toml(toml_filename, "phelel")
    ph = load_phelel_yaml_as_phonopy(yaml_filename)
    jobs = plan_supercells(project, "phelel", ph)
    for job in jobs:
        write_vasp_job(job)
        click.echo(f'VASP input files were written in "{job.directory}".')
    echo_vasp_vca_hint(jobs[0].cell)
