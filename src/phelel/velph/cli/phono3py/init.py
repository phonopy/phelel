"""Implementation of velph-phono3py-init."""

from __future__ import annotations

import pathlib
from typing import Optional, Union

import click
import numpy as np
from phono3py import Phono3py
from phono3py.interface.calculator import get_default_displacement_distance
from phonopy.structure.atoms import parse_cell_dict

from phelel.velph.cli.utils import get_nac_params


def run_init(
    toml_dict: dict,
    current_directory: pathlib.Path = pathlib.Path(""),
    number_of_snapshots: Optional[int] = None,
) -> Optional[Phono3py]:
    """Generate displacements and write phono3py_disp.yaml.

    current_directory : Path
        Used for test.

    """
    convcell = parse_cell_dict(toml_dict["unitcell"])
    supercell_matrix = toml_dict["phelel"].get("supercell_dimension", None)
    if "phonopy" in toml_dict:
        phonon_supercell_matrix = toml_dict["phonopy"].get("supercell_dimension", None)
    else:
        phonon_supercell_matrix = None
    if "primitive_cell" in toml_dict:
        primitive = parse_cell_dict(toml_dict["primitive_cell"])
        primitive_matrix = np.dot(np.linalg.inv(convcell.cell.T), primitive.cell.T)
    else:
        primitive = convcell
        primitive_matrix = None

    is_symmetry = True
    try:
        if toml_dict["phelel"]["nosym"] is True:
            is_symmetry = False
    except KeyError:
        pass

    ph3py = Phono3py(
        convcell,
        supercell_matrix=supercell_matrix,
        phonon_supercell_matrix=phonon_supercell_matrix,
        primitive_matrix=primitive_matrix,
        is_symmetry=is_symmetry,
        calculator="vasp",
    )

    amplitude = toml_dict["phelel"].get("amplitude", None)

    if number_of_snapshots is None:
        is_diagonal = toml_dict["phelel"].get("diagonal", True)
        is_plusminus = toml_dict["phelel"].get("plusminus", "auto")
    else:
        is_diagonal = False
        is_plusminus = False

    _generate_phono3py_supercells(
        ph3py,
        interface_mode="vasp",
        distance=amplitude,
        is_plusminus=is_plusminus,
        is_diagonal=is_diagonal,
        number_of_snapshots=number_of_snapshots,
    )

    nac_directory = current_directory / "nac"
    if nac_directory.exists():
        click.echo('Found "nac" directory. Read NAC params.')
        vasprun_path = nac_directory / "vasprun.xml"
        if vasprun_path.exists():
            nac_params = get_nac_params(
                toml_dict,
                vasprun_path,
                primitive,
                convcell,
                is_symmetry,
            )
            if nac_params is not None:
                ph3py.nac_params = nac_params
        else:
            click.echo('Not found "nac/vasprun.xml". NAC params were not included.')

    return ph3py


def _generate_phono3py_supercells(
    phono3py: Phono3py,
    interface_mode: str = "vasp",
    distance: Optional[float] = None,
    is_plusminus: Union[str, bool] = "auto",
    is_diagonal: bool = True,
    number_of_snapshots: Optional[int] = None,
):
    """Generate phelel supercells."""
    if distance is None:
        _distance = get_default_displacement_distance(interface_mode)
    else:
        _distance = distance

    phono3py.generate_displacements(
        distance=_distance,
        is_plusminus=is_plusminus,
        is_diagonal=is_diagonal,
        number_of_snapshots=number_of_snapshots,
    )
    click.echo(f"Displacement distance: {_distance}")
    click.echo(
        f"Number of displacements: {len(phono3py.supercells_with_displacements)}"
    )

    if phono3py.phonon_supercell_matrix is not None:
        # For estimating number of displacements for harmonic phonon
        phono3py.generate_fc2_displacements(
            distance=distance, is_plusminus=False, is_diagonal=False
        )
        n_disps = len(phono3py.phonon_supercells_with_displacements)
        phono3py.generate_fc2_displacements(
            distance=distance, number_of_snapshots=n_disps, is_plusminus=True
        )
        n_snapshots = len(phono3py.phonon_supercells_with_displacements)
        click.echo(f"Number of displacements for phonon: {n_snapshots}")
