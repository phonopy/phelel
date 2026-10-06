"""Implementation of velph-phono3py-init."""

from __future__ import annotations

import pathlib

import click
import numpy as np
from phono3py import Phono3py
from phonopy.structure.atoms import parse_cell_dict

from phelel.velph.cli.utils import (
    DisplacementOptions,
    get_displacement_options,
    get_nac_params,
)


def run_init(
    toml_dict: dict,
    current_directory: pathlib.Path = pathlib.Path(""),
    number_of_snapshots: int | None = None,
) -> Phono3py:
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
    if convcell.has_mixtures or convcell.has_weighted_species:
        raise click.ClickException(
            "phono3py does not support site-mixture cells (mixture or weight in "
            "[unitcell] of velph.toml)."
        )

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

    displacement_options = get_displacement_options(
        toml_dict["phono3py"], "phono3py", number_of_snapshots=number_of_snapshots
    )

    _generate_phono3py_supercells(ph3py, displacement_options)

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
    displacement_options: DisplacementOptions,
    number_of_snapshots_fc2: int | None = None,
):
    """Generate phono3py supercells with displacements."""
    distance = displacement_options.amplitude
    phono3py.generate_displacements(
        distance=distance,
        is_plusminus=displacement_options.plusminus,
        is_diagonal=displacement_options.diagonal,
        number_of_snapshots=displacement_options.number_of_snapshots,
    )
    click.echo(f"Displacement distance: {distance}")
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
