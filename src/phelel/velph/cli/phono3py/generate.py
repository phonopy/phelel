"""Implementation of velph-phono3py-generate."""

from __future__ import annotations

import pathlib

import click

from phelel.velph.cli.supercells import load_phono3py_yaml, plan_supercells
from phelel.velph.cli.utils import echo_vasp_vca_hint
from phelel.velph.jobs.vasp_job import read_velph_toml, write_vasp_job


def write_supercell_input_files(
    toml_filename: pathlib.Path, yaml_filename: pathlib.Path
) -> None:
    """Generate VASP inputs of the supercells written in yaml_filename."""
    if not yaml_filename.exists():
        raise click.ClickException(
            f'"{yaml_filename}" not found.\nRun "velph phono3py init" to make it.'
        )
    config = read_velph_toml(toml_filename, "phono3py")
    ph = load_phono3py_yaml(yaml_filename)
    jobs = plan_supercells(config, "phono3py", ph)
    for job in jobs:
        write_vasp_job(job)
        click.echo(f'VASP input files were generated in "{job.directory}".')
    assert not isinstance(jobs[0].poscar, pathlib.Path)
    echo_vasp_vca_hint(jobs[0].poscar)
