"""Implementation of velph-phelel-differentiate."""

import os
import pathlib
from typing import Union

import click

from phelel import Phelel
from phelel.interface.vasp.derivatives import create_derivatives
from phelel.velph.cli.utils import get_num_digits


def run_derivatives(
    phe: Phelel,
    subtract_residual_forces: bool = True,
    dir_name: Union[str, bytes, os.PathLike] = "phelel",
) -> bool:
    """Calculate derivatives and write phelel_params.hdf5."""
    dir_names = []
    nd = get_num_digits(phe.supercells_with_displacements)
    for i, _ in enumerate(
        [
            phe.supercell,
        ]
        + phe.supercells_with_displacements
    ):
        id_number = f"{i:0{nd}d}"
        filepath = pathlib.Path(f"{dir_name}/disp-{id_number}")
        if filepath.exists():
            if _check_files_exist(filepath):
                dir_names.append(filepath)
            else:
                click.echo(f'Necessary file not found in "{filepath}".', err=True)
                return False
        else:
            click.echo(f'"{filepath}" does not exist.', err=True)
            return False

    if phe.phonon_supercell_matrix is not None:
        nd = get_num_digits(phe.phonon_supercells_with_displacements)
        for i, _ in enumerate(
            [
                phe.phonon_supercell,
            ]
            + phe.phonon_supercells_with_displacements
        ):
            id_number = f"{i:0{nd}d}"
            filepath = pathlib.Path(f"{dir_name}/ph-disp-{id_number}")
            if filepath.exists():
                dir_names.append(filepath)
            else:
                click.echo(f'"{filepath}" does not exist.', err=True)
                return False

    create_derivatives(
        phe,
        dir_names,
        subtract_rfs=subtract_residual_forces,
        log_level=0,
    )

    return True


def _check_files_exist(filepath: pathlib.Path) -> bool:
    if not _check_file_exists(filepath, "vasprun.xml"):
        click.echo(f'"{filepath}/vasprun.xml" not found.', err=True)
        return False
    if _check_four_files_exist(filepath):
        return True
    else:
        if (filepath / "vaspout.h5").exists():
            click.echo(f'Found "{filepath}/vaspout.h5".', err=True)
            return True
        else:
            for filename in (
                "inwap.yaml",
                "LOCAL-POTENTIAL.bin",
                "PAW-STRENGTH.bin",
                "PAW-OVERLAP.bin",
            ):
                if not _check_file_exists(filepath, filename):
                    click.echo(f'"{filepath}/{filename}" not found.', err=True)
            return False


def _check_four_files_exist(filepath: pathlib.Path) -> bool:
    for filename in (
        "inwap.yaml",
        "LOCAL-POTENTIAL.bin",
        "PAW-STRENGTH.bin",
        "PAW-OVERLAP.bin",
    ):
        if not _check_file_exists(filepath, filename):
            return False
    return True


def _check_file_exists(filepath: pathlib.Path, filename: str) -> bool:
    """Check if the necessary file exists.

    The file can be compressed with xz, etc.

    The file names that can be checked are:
        inwap.yaml
        LOCAL-POTENTIAL.bin
        PAW-STRENGTH.bin
        PAW-OVERLAP.bin
        vasprun.xml

    """
    return bool(list(pathlib.Path(filepath).glob(f"{filename}*")))
