"""Implementation of velph-phonopy-generate."""

from __future__ import annotations

import pathlib

import click
import phonopy
import tomli

from phelel.velph.cli.phelel.generate import (
    write_supercells,
)


def write_supercell_input_files(
    toml_filename: pathlib.Path,
    phonopy_yaml_filename: pathlib.Path,
) -> None:
    """Generate supercells."""
    if not phonopy_yaml_filename.exists():
        click.echo(f'File "{phonopy_yaml_filename}" not found.', err=True)
        click.echo('Run "velph phonopy init" if necessary.', err=True)
        return None

    ph = phonopy.load(phonopy_yaml_filename)
    with open(toml_filename, "rb") as f:
        toml_dict = tomli.load(f)

    write_supercells(ph, toml_dict, dir_name="phonopy")
