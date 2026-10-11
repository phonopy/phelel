"""Implementation of velph-phelel-init."""

from __future__ import annotations

import pathlib

from phelel import Phelel
from phelel.cui.create_supercells import generate_phelel_supercells
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
) -> Phelel:
    """Return Phelel with the supercells with displacements of [phelel].

    current_directory is where the directory "nac" is looked for.  toml_filename
    is used in error messages.

    """
    section = get_code_section(config, "phelel", toml_filename)
    # The defaults of velph init give these fields.
    assert section.plusminus is not None and section.diagonal is not None
    unitcell, _, primitive_matrix = get_cells(config)
    is_symmetry = not section.nosym
    phe = Phelel(
        unitcell,
        supercell_matrix=get_supercell_matrix(section),
        primitive_matrix=primitive_matrix,
        is_symmetry=is_symmetry,
        calculator="vasp",
        site_mixture_scheme=section.site_mixture_scheme or "merge",
    )
    generate_phelel_supercells(
        phe,
        interface_mode="vasp",
        distance=section.amplitude,
        is_plusminus=section.plusminus,
        is_diagonal=section.diagonal,
    )
    set_nac_params(phe, config, is_symmetry, current_directory)
    return phe
