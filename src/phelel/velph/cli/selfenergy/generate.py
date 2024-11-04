"""Implementation of velph-selfenergy-generate."""

import pathlib
import shutil
from typing import Literal, Optional

import click
import h5py
import numpy as np
import tomli
from phonopy.interface.calculator import write_crystal_structure
from phonopy.structure.atoms import parse_cell_dict

from phelel.velph.cli.utils import (
    assert_kpoints_mesh_symmetry,
    get_scheduler_dict,
    write_incar,
    write_kpoints_mesh_mode,
    write_launch_script,
)


def write_input_files(toml_filepath: pathlib.Path, dry_run: bool):
    """Generate el-ph input files."""
    write_selfenergy_input_files(toml_filepath, dry_run, calc_type="selfenergy")


def write_selfenergy_input_files(
    toml_filepath: pathlib.Path,
    dry_run: bool,
    calc_type: Optional[Literal["transport", "selfenergy"]] = None,
    current_directory: pathlib.Path = pathlib.Path(""),
    energy_threshold: float = 0.5,
) -> None:
    """Generate el-ph input files.

    current_directory : Path
        Used for test.
    energy_threshold : float
        Energy threshold (gap) to estimate elph_nbands used for
        calc_type="transport".

    """
    directory_path = pathlib.Path(calc_type)

    with open(toml_filepath, "rb") as f:
        toml_dict = tomli.load(f)

    if "vasp" not in toml_dict:
        click.echo(f"[vasp] section not found in {toml_filepath}.", err=True)
        return None

    if "phelel" in toml_dict["vasp"]:
        hdf5_filepath = pathlib.Path("phelel/phelel_params.hdf5")
    elif "supercell" in toml_dict["vasp"]:
        hdf5_filepath = pathlib.Path("supercell/phelel_params.hdf5")
    else:
        click.echo(
            f'[vasp.phelel] section not found in "{toml_filepath}".',
            err=True,
        )
        return None

    if calc_type not in toml_dict["vasp"]:
        click.echo(
            f'[vasp.{calc_type}] section not found in "{toml_filepath}".',
            err=True,
        )
        return None

    # Check phelel_params.hdf5
    if not dry_run and not hdf5_filepath.exists():
        click.echo(f'"{hdf5_filepath}" not found.', err=True)
        click.echo('Run "velph phelel differentiate" if necessary.', err=True)
        return None

    # mkdir, e.g., selfenergy
    directory_path.mkdir(parents=True, exist_ok=True)

    # Dry run
    toml_incar_dict = toml_dict["vasp"][calc_type]["incar"]
    if dry_run:
        toml_incar_dict["nelm"] = 0
        toml_incar_dict["elph_run"] = False

    # Automatic elph_nbands setting for transport.
    if calc_type == "transport":
        # Here toml_incar_dict is updated for setting elph_nbands
        band_index = _find_elph_nbands(current_directory, energy_threshold)
        if band_index is not None:
            click.echo(f'  "elph_nbands={band_index + 1}" in INCAR is set.')
            toml_incar_dict["elph_nbands"] = band_index + 1

    # POSCAR
    primitive = parse_cell_dict(toml_dict["primitive_cell"])
    write_crystal_structure(directory_path / "POSCAR", primitive)

    # INCAR
    write_incar(toml_incar_dict, directory_path, cell=primitive)

    # KPOINTS
    kpoints_dict = toml_dict["vasp"][calc_type]["kpoints"]
    assert_kpoints_mesh_symmetry(toml_dict, kpoints_dict, primitive)
    write_kpoints_mesh_mode(
        toml_incar_dict,
        directory_path,
        f"vasp.{calc_type}.kpoints",
        kpoints_dict,
    )

    # KPOINTS_DENSE
    kpoints_dense_dict = toml_dict["vasp"][calc_type]["kpoints_dense"]
    assert_kpoints_mesh_symmetry(toml_dict, kpoints_dense_dict, primitive)
    write_kpoints_mesh_mode(
        toml_incar_dict,
        directory_path,
        f"vasp.{calc_type}.kpoints_dense",
        kpoints_dense_dict,
        kpoints_filename="KPOINTS_DENSE",
        kspacing_name="elph_kspacing",
    )

    # POTCAR
    potcar_path = pathlib.Path("POTCAR")
    if potcar_path.exists():
        shutil.copy2(potcar_path, directory_path / potcar_path)

    # phelel_params.hdf5
    if not dry_run:
        shutil.copy2(hdf5_filepath, directory_path / hdf5_filepath.name)

    # Scheduler launch script
    if "scheduler" in toml_dict:
        scheduler_dict = get_scheduler_dict(toml_dict, calc_type)
        write_launch_script(scheduler_dict, directory_path)

    click.echo(f'VASP input files were generated in "{directory_path}".')


def _find_elph_nbands(current_directory: pathlib.Path, energy_threshold: float) -> int:
    dos_directory = current_directory / "el_bands" / "dos"
    if dos_directory.exists():
        click.echo('Found "el_bands/dos" directory. Estimate elph_nbands.')
        vaspout_path = dos_directory / "vaspout.h5"
        if vaspout_path.exists():
            possiblly_occupied_band_index = _estimate_elph_nbands(
                vaspout_path, energy_threshold=energy_threshold
            )
            if possiblly_occupied_band_index is None:
                click.echo("Estimation of elph_nbands failed.")
                return None
            return possiblly_occupied_band_index
        else:
            click.echo('Not found "el_bands/dos/vasprun.xml".')
            return None


def _estimate_elph_nbands(
    vaspout_path: pathlib.Path,
    energy_threshold: float = 0.5,
    occupation_condition: float = 1e-5,
    unoccupation_condition: float = 1e-10,
) -> Optional[int]:
    """Estimate elph_nbands from eigenvalues in electronic DOS calculation.

    Parameters
    ----------
    vaspout_path : pathlib.Path
        "vaspout.h5" path.
    energy_threshold : float
        Energy threshold (gap) in eV to estimate elph_nbands. Default is 0.5 eV.
    occupation_condition : float
        Condition to determine either if bands are occupied or not. This value
        is used as occupation_number > occupation_condition. Default is 1e-5.
    unoccupation_condition : float
        Condition to determine either if bands are unoccupied or not. This value
        is used as occupation_number < unoccupation_condition. Default is 1e-10.

    Returns
    -------
    int or None
        Possibly occupied band index by counting from 0. Even if
        occupation number is zero, if some of bands can be occuped by
        excitation, this band index is returned.

    """
    with h5py.File(vaspout_path) as f:
        eigenvalues = f["results"]["electron_eigenvalues_kpoints_opt"]["eigenvalues"][:]
        occupations = f["results"]["electron_eigenvalues_kpoints_opt"]["fermiweights"][
            :
        ]

    nbands = eigenvalues.shape[2]
    unoccupied_band_index = None
    for i in range(nbands):
        if (occupations[:, :, i] < unoccupation_condition).all():
            unoccupied_band_index = i
            break
    if unoccupied_band_index is None:
        return None

    occupied_eigvals = np.sort(
        np.extract(
            occupations[:, :, unoccupied_band_index - 1] > occupation_condition,
            eigenvalues[:, :, unoccupied_band_index - 1],
        )
    )
    max_occupied_eigval = np.max(occupied_eigvals)

    for band_index in range(unoccupied_band_index, nbands):
        min_unoccupied = np.min(eigenvalues[:, :, band_index])
        if max_occupied_eigval + energy_threshold < min_unoccupied:
            return band_index - 1

    return None
