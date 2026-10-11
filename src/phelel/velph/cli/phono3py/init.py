"""Implementation of velph-phono3py-init."""

from __future__ import annotations

import pathlib

import click
from phono3py import Phono3py

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
    number_of_snapshots: int | None = None,
    toml_filename: str | pathlib.Path = "velph.toml",
) -> Phono3py:
    """Return Phono3py with the supercells with displacements of [phono3py].

    number_of_snapshots, given by --rd of velph phono3py init, has priority over
    that of [phono3py].  With it, the displacements are random, and diagonal and
    plusminus are not used.  current_directory is where the directory "nac" is
    looked for.  toml_filename is used in error messages.

    """
    section = get_code_section(config, "phono3py", toml_filename)
    # The defaults of velph init give these fields.
    assert section.plusminus is not None and section.diagonal is not None
    unitcell, _, primitive_matrix = get_cells(config)
    if unitcell.has_mixtures or unitcell.has_weighted_species:
        raise click.ClickException(
            "phono3py does not support site-mixture cells (mixture or weight in "
            f'[unitcell] of "{toml_filename}").'
        )
    is_symmetry = not section.nosym
    ph3py = Phono3py(
        unitcell,
        supercell_matrix=get_supercell_matrix(section),
        primitive_matrix=primitive_matrix,
        is_symmetry=is_symmetry,
        calculator="vasp",
    )
    if number_of_snapshots is None:
        number_of_snapshots = section.number_of_snapshots
    ph3py.generate_displacements(
        distance=section.amplitude,
        is_plusminus=False if number_of_snapshots else section.plusminus,
        is_diagonal=False if number_of_snapshots else section.diagonal,
        number_of_snapshots=number_of_snapshots,
    )
    click.echo(f"Displacement distance: {section.amplitude}")
    click.echo(f"Number of displacements: {len(ph3py.supercells_with_displacements)}")
    set_nac_params(ph3py, config, is_symmetry, current_directory)
    return ph3py
