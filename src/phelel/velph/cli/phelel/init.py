"""Implementation of velph-supercell-init."""

from __future__ import annotations

import pathlib

import click
import numpy as np
from phonopy.structure.atoms import parse_cell_dict

from phelel import Phelel
from phelel.cui.create_supercells import generate_phelel_supercells
from phelel.velph.cli.utils import get_nac_params


def run_init(
    toml_dict: dict,
    current_directory: pathlib.Path = pathlib.Path(""),
) -> Phelel:
    """Generate supercell and displacements.

    current_directory : Path
        Used for test.

    """
    convcell = parse_cell_dict(toml_dict["unitcell"])
    assert convcell is not None
    supercell_matrix = None
    phonon_supercell_matrix = None
    for key in ("supercell_dimension", "supercell_matrix"):
        if key in toml_dict["phelel"]:
            supercell_matrix = toml_dict["phelel"][key]
    for key in ("phonon_supercell_dimension", "phonon_supercell_matrix"):
        if key in toml_dict["phelel"]:
            phonon_supercell_matrix = toml_dict["phelel"][key]
    if "primitive_cell" in toml_dict:
        primitive = parse_cell_dict(toml_dict["primitive_cell"])
        assert primitive is not None
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

    phe = Phelel(
        convcell,
        supercell_matrix=supercell_matrix,
        phonon_supercell_matrix=phonon_supercell_matrix,
        primitive_matrix=primitive_matrix,
        is_symmetry=is_symmetry,
        calculator="vasp",
    )

    is_diagonal = toml_dict["phelel"].get("diagonal", True)
    is_plusminus = toml_dict["phelel"].get("plusminus", "auto")
    amplitude = toml_dict["phelel"].get("amplitude", None)

    generate_phelel_supercells(
        phe,
        interface_mode="vasp",
        distance=amplitude,
        is_plusminus=is_plusminus,
        is_diagonal=is_diagonal,
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
                phe.nac_params = nac_params
        else:
            click.echo('Not found "nac/vasprun.xml". NAC params were not included.')

    return phe
