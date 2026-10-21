"""Implementation of velph-phelel-phonopy."""

from __future__ import annotations

import pathlib

import click

from phelel.interface.vasp.derivatives import (
    get_vasp_force_filenames,
    read_vasp_forces,
)
from phelel.velph.cli.supercells import load_phelel_yaml_as_phonopy
from phelel.velph.jobs.project import read_velph_toml
from phelel.velph.jobs.project_directory import ProjectDirectory


def create_phonopy_yaml(
    toml_filename: pathlib.Path,
    project_dir: ProjectDirectory | None = None,
    subtract_residual_forces: bool = True,
):
    """Write phelel/phonopy_params.yaml with the forces of the phelel supercells.

    The supercells are those of phelel/phelel_disp.yaml, and the forces are
    read from phelel/disp-*, also when phelel_disp.yaml has the phonon
    supercell, which is calculated by velph phonopy.  The directories are those
    of project_dir, in the current directory by default.  With the merge scheme of
    site mixture, the forces are on the atoms of the unmerged supercell.
    click.ClickException is raised when neither vaspout.h5 nor vasprun.xml is
    found in a directory.

    """
    if project_dir is None:
        project_dir = ProjectDirectory()
    yaml_filename = project_dir.disp_yaml("phelel")
    phonopy_yaml_filename = project_dir.phonopy_params

    project = read_velph_toml(toml_filename)
    is_symmetry = not project.code_section("phelel").nosym
    if not is_symmetry:
        click.echo(
            'Found "nosym = true" in [phelel] section. Symmetrization is turned off.'
        )

    # NAC params are those of phelel_disp.yaml.
    ph = load_phelel_yaml_as_phonopy(yaml_filename, is_symmetry=is_symmetry)
    supercell = ph.supercell if ph.unmerged_supercell is None else ph.unmerged_supercell
    cells = ph.supercells_with_displacements
    assert cells is not None
    dir_names = project_dir.disp_directories("phelel", len(cells))
    try:
        filenames = get_vasp_force_filenames(dir_names)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    ph.forces = read_vasp_forces(
        filenames[1:],
        supercell,
        perfect_filename=filenames[0] if subtract_residual_forces else None,
    )

    phonopy_yaml_filename.parent.mkdir(parents=True, exist_ok=True)
    ph.save(filename=phonopy_yaml_filename)
    click.echo(
        f'"{phonopy_yaml_filename}" was written with the forces of the phelel '
        "supercells."
    )
    if "phonopy" in project.input_sets:
        click.echo(
            'The phonopy calculation with the [phonopy] supercell is in "phonopy".'
        )
