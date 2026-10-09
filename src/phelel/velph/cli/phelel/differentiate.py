"""Implementation of velph-phelel-differentiate."""

from __future__ import annotations

import os
import pathlib

import click

from phelel import Phelel
from phelel.interface.vasp.derivatives import create_derivatives
from phelel.velph.cli.utils import get_num_digits


def run_derivatives(
    phe: Phelel,
    subtract_residual_forces: bool = True,
    dir_name: str | os.PathLike = "phelel",
    verbose: bool = False,
) -> bool:
    """Calculate derivatives and write phelel_params.hdf5."""
    if phe.unmerged_unitcell is not None:
        click.echo(
            "Derivatives are not supported with the merge scheme of site mixture.",
            err=True,
        )
        return False
    dir_names = []
    if phe.supercells_with_displacements is None:
        raise RuntimeError("supercells_with_displacements is None.")
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
        if phe.phonon_supercells_with_displacements is None:
            raise RuntimeError("phonon_supercells_with_displacements is None.")
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
        log_level=int(verbose),
    )

    return True


def _check_files_exist(filepath: pathlib.Path) -> bool:
    if (filepath / "vaspout.h5").exists():
        click.echo(f'Found "{filepath}/vaspout.h5".', err=False)
        return True
    else:
        if not _check_file_exists(filepath, "vasprun.xml"):
            click.echo(f'"{filepath}/vasprun.xml" not found.', err=True)
            return False
        return _check_four_files_exist(filepath)


def _check_four_files_exist(filepath: pathlib.Path) -> bool:
    four_files_exist = True
    for filename in (
        "inwap.yaml",
        "LOCAL-POTENTIAL.bin",
        "PAW-STRENGTH.bin",
        "PAW-OVERLAP.bin",
    ):
        if not _check_file_exists(filepath, filename):
            click.echo(f'"{filepath}/{filename}" not found.', err=True)
            four_files_exist = False
    return four_files_exist


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


def get_selfenergy_prec(toml_dict: dict, toml_filename: str) -> str | None:
    """Return prec of [vasp.selfenergy.incar] for the FFT mesh of --encut.

    None is returned when prec is not found. CutoffToFFTMesh takes None as
    "normal", which is the default of PREC in VASP.

    """
    try:
        incar = toml_dict["vasp"]["selfenergy"]["incar"]
    except KeyError:
        click.echo(f'[vasp.selfenergy.incar] not found in "{toml_filename}".')
        click.echo('prec = "normal" (VASP default) is assumed.')
        return None
    incar_lower = {key.lower(): value for key, value in incar.items()}
    if "prec" not in incar_lower:
        click.echo(f'"prec" not found in [vasp.selfenergy.incar] of "{toml_filename}".')
        click.echo('prec = "normal" (VASP default) is assumed.')
        return None
    return incar_lower["prec"]
