"""Implementation of velph-phelel-differentiate."""

from __future__ import annotations

import os
import pathlib

import click
import numpy as np
from numpy.typing import NDArray

from phelel import Phelel
from phelel.interface.vasp.derivatives import create_derivatives
from phelel.velph.cli.utils import get_num_digits
from phelel.velph.config.schema import VelphConfig
from phelel.velph.utils.vasp import CutoffToFFTMesh


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


# Input sets whose VASP calculations read phelel_params.hdf5.  Their FFT grid
# has to be the FFT mesh of the local potentials in phelel_params.hdf5.
ELPH_INPUT_SETS = ("selfenergy", "transport", "ph_selfenergy")


def get_fft_mesh(
    config: VelphConfig,
    lattice: NDArray,
    encut: float | None = None,
    toml_filename: str | os.PathLike = "velph.toml",
) -> NDArray:
    """Return the FFT mesh of the local potentials for phelel_params.hdf5.

    The mesh is that of VASP for ENCUT and PREC on the primitive cell, whose
    basis vectors are the rows of lattice.  ENCUT is encut, or that of the INCAR
    of each electron-phonon input set of velph.toml.  PREC is that of the INCAR
    of each electron-phonon input set; a missing PREC is Normal, the default of
    VASP.  The meshes of all these input sets have to be the same, because VASP
    stops when its FFT grid differs from the mesh in phelel_params.hdf5.  With
    encut and no electron-phonon input set, PREC is Normal.

    click.ClickException is raised when ENCUT is not found or when the meshes
    differ.

    """
    input_sets = (config.vasp.input_sets or {}) if config.vasp is not None else {}
    incars = {
        set_name: input_sets[set_name].incar or {}
        for set_name in ELPH_INPUT_SETS
        if set_name in input_sets
    }
    if not incars:
        if encut is None:
            raise click.ClickException(
                f'"{toml_filename}" has no electron-phonon input set, '
                "[vasp.selfenergy], [vasp.transport] or [vasp.ph_selfenergy], from "
                "which ENCUT is taken.\nGive ENCUT by --encut."
            )
        return CutoffToFFTMesh.get_FFTMesh(encut, lattice, None)
    meshes = {}
    for set_name, incar in incars.items():
        set_encut = encut if encut is not None else incar.get("encut")
        if set_encut is None:
            raise click.ClickException(
                f'[vasp.{set_name}.incar] of "{toml_filename}" does not give '
                '"encut".\nGive "encut" there, or give ENCUT by --encut.'
            )
        meshes[set_name] = (
            tuple(
                int(n)
                for n in CutoffToFFTMesh.get_FFTMesh(
                    set_encut, lattice, incar.get("prec")
                )
            ),
            set_encut,
            incar.get("prec"),
        )
    distinct_meshes = {mesh for mesh, _, _ in meshes.values()}
    if len(distinct_meshes) > 1:
        lines = [
            f"  [vasp.{set_name}]: encut = {set_encut}, "
            f"prec = {prec if prec is not None else 'not given (Normal)'}, "
            f"mesh = {list(mesh)}"
            for set_name, (mesh, set_encut, prec) in meshes.items()
        ]
        if encut is None:
            advice = (
                'Make "encut" and "prec" of these input sets give the same mesh, '
                "or give ENCUT by --encut."
            )
        else:
            advice = 'Make "prec" of these input sets give the same mesh.'
        raise click.ClickException(
            "The FFT meshes of the electron-phonon input sets of "
            f'"{toml_filename}" differ:\n' + "\n".join(lines) + "\n" + advice
        )
    return np.array(distinct_meshes.pop())
