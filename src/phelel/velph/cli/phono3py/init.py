"""Implementation of velph-phono3py-init."""

from __future__ import annotations

import pathlib
from typing import Literal

import click
import numpy as np
from phono3py import Phono3py
from phono3py.interface.calculator import get_default_displacement_distance
from phonopy.structure.atoms import parse_cell_dict

from phelel.velph.cli.utils import get_nac_params


def run_init(
    toml_dict: dict,
    current_directory: pathlib.Path = pathlib.Path(""),
    number_of_snapshots: int | None = None,
) -> Phono3py | None:
    """Generate displacements and write phono3py_disp.yaml.

    current_directory : Path
        Used for test.

    """
    if "phono3py" not in toml_dict:
        raise RuntimeError("[phono3py] section not found in toml file.")

    if "unitcell" not in toml_dict:
        raise RuntimeError("[unitcell] section not found in toml file.")

    convcell = parse_cell_dict(toml_dict["unitcell"])
    assert convcell is not None

    supercell_matrix = None
    for key in ("supercell_dimension", "supercell_matrix"):
        if key in toml_dict["phono3py"]:
            supercell_matrix = toml_dict["phono3py"][key]
    if "primitive_cell" in toml_dict:
        primitive = parse_cell_dict(toml_dict["primitive_cell"])
        assert primitive is not None
        primitive_matrix = np.dot(np.linalg.inv(convcell.cell.T), primitive.cell.T)
    else:
        primitive = convcell
        primitive_matrix = None

    is_symmetry = True
    try:
        if toml_dict["phono3py"]["nosym"] is True:
            is_symmetry = False
    except KeyError:
        pass

    ph3py = Phono3py(
        convcell,
        supercell_matrix=supercell_matrix,
        primitive_matrix=primitive_matrix,
        is_symmetry=is_symmetry,
        calculator="vasp",
    )

    amplitude = toml_dict["phono3py"].get("amplitude", None)
    if number_of_snapshots is None:
        is_diagonal = toml_dict["phono3py"].get("diagonal", True)
        is_plusminus = toml_dict["phono3py"].get("plusminus", "auto")
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
    distance: float | None = None,
    is_plusminus: Literal["auto"] | bool = "auto",
    is_diagonal: bool = True,
    number_of_snapshots: int | None = None,
    number_of_snapshots_fc2: int | None = None,
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
        if number_of_snapshots_fc2 is None:
            phono3py.generate_fc2_displacements(
                distance=distance, is_plusminus="auto", is_diagonal=False
            )
        else:
            phono3py.generate_fc2_displacements(
                distance=distance,
                number_of_snapshots=number_of_snapshots_fc2,
            )
        n_snapshots = len(phono3py.phonon_supercells_with_displacements)
        click.echo(f"Number of displacements for phonon: {n_snapshots}")
