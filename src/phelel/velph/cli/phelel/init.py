"""Implementation of velph-phelel-init."""

from __future__ import annotations

import pathlib

import click

from phelel import Phelel
from phelel.cui.create_supercells import generate_phelel_supercells
from phelel.velph.cli.supercells import (
    PHONOPY_DISP_YAML,
    get_cells,
    get_code_section,
    get_supercell_matrix,
    read_phonopy_phonon,
    set_nac_params,
)
from phelel.velph.config.schema import VelphConfig


def run_init(
    config: VelphConfig,
    current_directory: pathlib.Path = pathlib.Path(""),
    phonon_from_phonopy: bool = False,
    toml_filename: str | pathlib.Path = "velph.toml",
) -> Phelel:
    """Return Phelel with the supercells with displacements of [phelel].

    With phonon_from_phonopy, the phonon supercell matrix and the phonon dataset
    are those of phonopy/phonopy_disp.yaml.  current_directory is where the
    directories "nac" and "phonopy" are looked for.  toml_filename is used in
    error messages.

    """
    section = get_code_section(config, "phelel", toml_filename)
    # The defaults of velph init give these fields.
    assert section.plusminus is not None and section.diagonal is not None
    unitcell, _, primitive_matrix = get_cells(config)
    ph_phonon = (
        read_phonopy_phonon(config, current_directory, toml_filename)
        if phonon_from_phonopy
        else None
    )
    is_symmetry = not section.nosym
    phe = Phelel(
        unitcell,
        supercell_matrix=get_supercell_matrix(section),
        primitive_matrix=primitive_matrix,
        phonon_supercell_matrix=(
            None if ph_phonon is None else ph_phonon.supercell_matrix
        ),
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
    if ph_phonon is not None:
        # generate_phelel_supercells also generates phonon displacements, which
        # are replaced by those of phonopy.
        assert ph_phonon.dataset is not None
        phe.phonon_dataset = dict(ph_phonon.dataset)
        assert phe.phonon_supercells_with_displacements is not None
        click.echo(
            "Number of phonon displacements: "
            f"{len(phe.phonon_supercells_with_displacements)}, taken from "
            f'"{current_directory / PHONOPY_DISP_YAML}".'
        )
    set_nac_params(phe, config, is_symmetry, current_directory)
    return phe
