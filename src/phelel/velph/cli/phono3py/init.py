"""Implementation of velph-phono3py-init."""

from __future__ import annotations

import pathlib

import click
from phono3py import Phono3py

from phelel.velph.cli.supercells import (
    PHONOPY_DISP_YAML,
    read_phonopy_phonon,
    set_nac_params,
)
from phelel.velph.jobs.project import Project


def run_init(
    project: Project,
    current_directory: pathlib.Path = pathlib.Path(""),
    number_of_snapshots: int | None = None,
    phonon_from_phonopy: bool = False,
) -> Phono3py:
    """Return Phono3py with the supercells with displacements of [phono3py].

    number_of_snapshots, given by --rd of velph phono3py init, has priority over
    that of [phono3py].  With it, the displacements are random, and diagonal and
    plusminus are not used.  With phonon_from_phonopy, the phonon supercell
    matrix and the phonon dataset, those of fc2, are those of
    phonopy/phonopy_disp.yaml.  current_directory is where the directories
    "nac" and "phonopy" are looked for.

    """
    section = project.code_section("phono3py")
    # code_section fills these fields with the defaults of velph init.
    assert section.plusminus is not None and section.diagonal is not None
    unitcell = project.unitcell
    if unitcell.has_mixtures or unitcell.has_weighted_species:
        raise click.ClickException(
            "phono3py does not support site-mixture cells (mixture or weight in "
            f'[unitcell] of "{project.toml_filename}").'
        )
    ph_phonon = (
        read_phonopy_phonon(project, current_directory) if phonon_from_phonopy else None
    )
    is_symmetry = not section.nosym
    ph3py = Phono3py(
        unitcell,
        supercell_matrix=project.supercell_matrix("phono3py"),
        primitive_matrix=project.primitive_matrix,
        phonon_supercell_matrix=(
            None if ph_phonon is None else ph_phonon.supercell_matrix
        ),
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
    if ph_phonon is not None:
        # generate_displacements also generates phonon displacements, which are
        # replaced by those of phonopy.
        ph3py.phonon_dataset = ph_phonon.dataset
        assert ph3py.phonon_supercells_with_displacements is not None
        click.echo(
            "Number of phonon displacements: "
            f"{len(ph3py.phonon_supercells_with_displacements)}, taken from "
            f'"{current_directory / PHONOPY_DISP_YAML}".'
        )
    set_nac_params(ph3py, project, is_symmetry, current_directory)
    return ph3py
