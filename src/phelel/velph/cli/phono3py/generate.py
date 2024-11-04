"""Implementation of velph-phono3py-generate."""

from __future__ import annotations

import pathlib

import click
import phono3py
import tomli

from phelel.velph.cli.phelel.generate import (
    write_phonon_supercells,
    write_supercells,
)


def write_supercell_input_files(
    toml_filename: pathlib.Path,
    phono3py_yaml_filename: pathlib.Path,
) -> None:
    """Generate supercells."""
    if not phono3py_yaml_filename.exists():
        click.echo(f'File "{phono3py_yaml_filename}" not found.', err=True)
        click.echo('Run "velph phono3py init" if necessary.', err=True)
        return None

    ph3py = phono3py.load(phono3py_yaml_filename)
    with open(toml_filename, "rb") as f:
        toml_dict = tomli.load(f)

    write_supercells(ph3py, toml_dict, dir_name="phono3py")
    if ph3py.phonon_supercell_matrix is not None:
        if "phonon" in toml_dict["vasp"]["phono3py"]:
            write_phonon_supercells(ph3py, toml_dict, dir_name="phono3py")
        else:
            print(f'[vasp.phono3py.phonon.*] not found in "{toml_filename}"')
