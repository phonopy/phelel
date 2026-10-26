"""Implementation of velph-phelel-init."""

from __future__ import annotations

import click

from phelel import Phelel
from phelel.cui.create_supercells import generate_phelel_supercells
from phelel.velph.cli.supercells import (
    read_phonopy_phonon,
    set_nac_params,
)
from phelel.velph.jobs.project import Project
from phelel.velph.jobs.project_directory import ProjectDirectory


def run_init(
    project: Project,
    project_dir: ProjectDirectory | None = None,
    phonon_from_phonopy: bool = False,
) -> Phelel:
    """Return Phelel with the supercells with displacements of [phelel].

    With phonon_from_phonopy, the phonon supercell matrix and the phonon dataset
    are those of phonopy/phonopy_disp.yaml.  project_dir gives the directories
    "nac" and "phonopy", in the current directory by default.

    """
    if project_dir is None:
        project_dir = ProjectDirectory()
    section = project.code_section("phelel")
    # code_section fills these fields with the defaults of velph init.
    assert section.plusminus is not None and section.diagonal is not None
    ph_phonon = (
        read_phonopy_phonon(project, project_dir) if phonon_from_phonopy else None
    )
    is_symmetry = not section.nosym
    phe = Phelel(
        project.unitcell,
        supercell_matrix=project.supercell_matrix("phelel"),
        primitive_matrix=project.primitive_matrix,
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
    assert phe.supercells_with_displacements is not None
    click.echo(f"Displacement distance: {section.amplitude}")
    click.echo(f"Number of displacements: {len(phe.supercells_with_displacements)}")
    if ph_phonon is not None:
        # generate_phelel_supercells also generates phonon displacements, which
        # are replaced by those of phonopy.
        assert ph_phonon.dataset is not None
        phe.phonon_dataset = ph_phonon.dataset
        assert phe.phonon_supercells_with_displacements is not None
        click.echo(
            "Number of phonon displacements: "
            f"{len(phe.phonon_supercells_with_displacements)}, taken from "
            f'"{project_dir.disp_yaml("phonopy")}".'
        )
    set_nac_params(phe, project, is_symmetry, project_dir)
    return phe
