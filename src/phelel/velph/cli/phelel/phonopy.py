"""Implementation of velph-phelel-phonopy."""

from __future__ import annotations

import pathlib

import click

from phelel.interface.vasp.derivatives import (
    get_vasp_force_filenames,
    read_vasp_forces,
)
from phelel.velph.cli.supercells import load_phelel_yaml_as_phonopy
from phelel.velph.cli.utils import get_num_digits
from phelel.velph.jobs.project import read_velph_toml


def create_phonopy_yaml(
    toml_filename: pathlib.Path,
    yaml_filename: pathlib.Path,
    dir_name: str,
    subtract_residual_forces: bool = True,
):
    """Write phonopy_params.yaml with the forces of the supercells of phelel.

    The forces are read from <dir_name>/disp-*, also when yaml_filename has the
    phonon supercell, which is calculated by velph phonopy.  With the merge
    scheme of site mixture, the forces are on the atoms of the unmerged
    supercell.  click.ClickException is raised when neither vaspout.h5 nor
    vasprun.xml is found in a directory.

    """
    phonopy_yaml_filename = pathlib.Path(dir_name) / "phonopy_params.yaml"

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
    nd = get_num_digits(cells)
    dir_names = [
        pathlib.Path(dir_name) / f"disp-{i:0{nd}d}" for i in range(len(cells) + 1)
    ]
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
