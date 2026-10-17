"""Implementation of velph-phonopy-init."""

from __future__ import annotations

import pathlib

import click
from phonopy import Phonopy

from phelel.velph.cli.supercells import set_nac_params
from phelel.velph.jobs.project import Project


def run_init(
    project: Project,
    current_directory: pathlib.Path = pathlib.Path(""),
) -> Phonopy:
    """Return Phonopy with the supercells with displacements of [phonopy].

    With number_of_snapshots, the displacements are random, and diagonal and
    plusminus are not used.  current_directory is where the directory "nac"
    is looked for.

    """
    section = project.code_section("phonopy")
    # code_section fills these fields with the defaults of velph init.
    assert section.plusminus is not None and section.diagonal is not None
    is_symmetry = not section.nosym
    ph = Phonopy(
        project.unitcell,
        supercell_matrix=project.supercell_matrix("phonopy"),
        primitive_matrix=project.primitive_matrix,
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
    set_nac_params(ph, project, is_symmetry, current_directory)
    return ph
