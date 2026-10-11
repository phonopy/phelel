"""Implementation of velph-phonopy-init."""

from __future__ import annotations

import pathlib

import click
from phonopy import Phonopy

from phelel.velph.cli.supercells import (
    get_cells,
    get_code_section,
    get_supercell_matrix,
    set_nac_params,
)
from phelel.velph.config.schema import VelphConfig


def run_init(
    config: VelphConfig,
    current_directory: pathlib.Path = pathlib.Path(""),
    toml_filename: str | pathlib.Path = "velph.toml",
) -> Phonopy:
    """Return Phonopy with the supercells with displacements of [phonopy].

    With number_of_snapshots, the displacements are random, and diagonal and
    plusminus are not used.  current_directory is where the directory "nac"
    is looked for.  toml_filename is used in error messages.

    """
    section = get_code_section(config, "phonopy", toml_filename)
    # The defaults of velph init give these fields.
    assert section.plusminus is not None and section.diagonal is not None
    unitcell, _, primitive_matrix = get_cells(config)
    is_symmetry = not section.nosym
    ph = Phonopy(
        unitcell,
        supercell_matrix=get_supercell_matrix(section),
        primitive_matrix=primitive_matrix,
        is_symmetry=is_symmetry,
        calculator="vasp",
        site_mixture_scheme=section.site_mixture_scheme or "merge",
    )
    number_of_snapshots = section.number_of_snapshots
    ph.generate_displacements(
        distance=section.amplitude,
        is_plusminus=False if number_of_snapshots else section.plusminus,
        is_diagonal=False if number_of_snapshots else section.diagonal,
        number_of_snapshots=number_of_snapshots,
    )
    assert ph.supercells_with_displacements is not None
    click.echo(f"Displacement distance: {section.amplitude}")
    click.echo(f"Number of displacements: {len(ph.supercells_with_displacements)}")
    set_nac_params(ph, config, is_symmetry, current_directory)
    return ph
