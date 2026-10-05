"""Implementation of velph-phonopy-init."""

from __future__ import annotations

import pathlib

import click
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import parse_cell_dict

from phelel.velph.cli.utils import (
    DisplacementOptions,
    get_displacement_options,
    get_nac_params,
)


def run_init(
    toml_dict: dict,
    current_directory: pathlib.Path = pathlib.Path(""),
) -> Phonopy:
    """Generate displacements and write phonopy_disp.yaml.

    current_directory : Path
        Used for test.

    """
    if "phonopy" not in toml_dict:
        raise RuntimeError("[phonopy] section not found in toml file.")

    convcell = parse_cell_dict(toml_dict["unitcell"])
    assert convcell is not None
    supercell_matrix = None
    for key in ("supercell_dimension", "supercell_matrix"):
        if key in toml_dict["phonopy"]:
            supercell_matrix = toml_dict["phonopy"][key]
    if "primitive_cell" in toml_dict:
        primitive = parse_cell_dict(toml_dict["primitive_cell"])
        assert primitive is not None
        primitive_matrix = np.dot(np.linalg.inv(convcell.cell.T), primitive.cell.T)
    else:
        primitive = convcell
        primitive_matrix = None

    is_symmetry = True
    try:
        if toml_dict["phonopy"]["nosym"] is True:
            is_symmetry = False
    except KeyError:
        pass

    ph = Phonopy(
        convcell,
        supercell_matrix=supercell_matrix,
        primitive_matrix=primitive_matrix,
        is_symmetry=is_symmetry,
        calculator="vasp",
    )

    displacement_options = get_displacement_options(
        toml_dict["phonopy"],
        "phonopy",
        number_of_snapshots=toml_dict["phonopy"].get("number_of_snapshots"),
    )

    _generate_phonopy_supercells(ph, displacement_options)

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
                ph.nac_params = nac_params
        else:
            click.echo('Not found "nac/vasprun.xml". NAC params were not included.')

    return ph


def _generate_phonopy_supercells(
    phonopy: Phonopy, displacement_options: DisplacementOptions
):
    """Generate phonopy supercells with displacements."""
    phonopy.generate_displacements(
        distance=displacement_options.amplitude,
        is_plusminus=displacement_options.plusminus,
        is_diagonal=displacement_options.diagonal,
        number_of_snapshots=displacement_options.number_of_snapshots,
    )
    assert phonopy.supercells_with_displacements is not None
    click.echo(f"Displacement distance: {displacement_options.amplitude}")
    click.echo(f"Number of displacements: {len(phonopy.supercells_with_displacements)}")
