"""Implementation of velph-phelel-differentiate."""

import os
import pathlib
from typing import Optional, Union

import click

from phelel import Phelel
from phelel.interface.vasp.derivatives import create_derivatives
from phelel.velph.cli.utils import get_num_digits


def run_derivatives(
    phe: Phelel,
    hdf5_filename: Union[str, bytes, os.PathLike] = "supercell/phelel_params.hdf5",
    subtract_residual_forces: bool = True,
    nufft: Optional[str] = None,
    finufft_eps: Optional[float] = None,
    dir_name: Union[str, bytes, os.PathLike] = "supercell",
) -> None:
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
                return None
        else:
            click.echo(f'"{filepath}" does not exist.', err=True)
            return None

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
                return None

    pathlib.Path(hdf5_filename).parent.mkdir(parents=True, exist_ok=True)

    create_derivatives(
        phe,
        dir_names,
        nufft=nufft,
        finufft_eps=finufft_eps,
        subtract_rfs=subtract_residual_forces,
        log_level=0,
    )
    phe.save_hdf5(filename=hdf5_filename)

    click.echo(f'"{hdf5_filename}" has been made.')


def _check_files_exist(filepath: pathlib.Path) -> bool:
    if not (filepath / "vasprun.xml").exists():
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
                if not (filepath / filename).exists():
                    click.echo(f'"{filepath}/{filename}" not found.', err=True)
            return False


def _check_four_files_exist(filepath: pathlib.Path) -> bool:
    """Check if the necessary files exist.

    inwap.yaml
    LOCAL-POTENTIAL.bin
    PAW-STRENGTH.bin
    PAW-OVERLAP.bin

    """
    for filename in (
        "inwap.yaml",
        "LOCAL-POTENTIAL.bin",
        "PAW-STRENGTH.bin",
        "PAW-OVERLAP.bin",
    ):
        if not (filepath / filename).exists():
            return False
    return True
