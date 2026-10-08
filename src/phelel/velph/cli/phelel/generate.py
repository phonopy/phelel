"""Implementation of velph-supercell-generate."""

from __future__ import annotations

import pathlib

import click
import tomli

import phelel
from phelel.velph.cli.supercells import write_phonon_supercells, write_supercells


def write_supercell_input_files(
    toml_filename: pathlib.Path,
    phelel_yaml_filename: pathlib.Path,
    dir_name: str = "phelel",
) -> None:
    """Generate supercells."""
    if not phelel_yaml_filename.exists():
        click.echo(f'File "{phelel_yaml_filename}" not found.', err=True)
        click.echo('Run "velph phelel init" if necessary.', err=True)
        return None

    phe = phelel.load(phelel_yaml_filename)
    with open(toml_filename, "rb") as f:
        toml_dict = tomli.load(f)

    write_supercells(phe, toml_dict, dir_name=dir_name)
    if phe.phonon_supercell_matrix is not None:
        if "phonon" in toml_dict["vasp"][dir_name]:
            write_phonon_supercells(phe, toml_dict, dir_name=dir_name)
        else:
            print(f'[vasp.{dir_name}.phonon.*] not found in "{toml_filename}"')
