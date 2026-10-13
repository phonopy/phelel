"""Implementation of velph-phelel-differentiate."""

from __future__ import annotations

import pathlib

import click

import phelel
from phelel.interface.vasp.derivatives import (
    get_vasp_force_filenames,
    read_vasp_forces,
)
from phelel.velph.cli.utils import get_num_digits
from phelel.velph.jobs.vasp_job import read_velph_toml


def create_phonopy_yaml(
    toml_filename: pathlib.Path,
    yaml_filename: pathlib.Path,
    dir_name: str,
    subtract_residual_forces: bool = True,
):
    """Calculate derivatives and write phelel_params.hdf5."""
    phonopy_yaml_filename = pathlib.Path(f"{dir_name}/phonopy_params.yaml")

    config = read_velph_toml(toml_filename)
    is_symmetry = True
    if config.phelel is not None and config.phelel.nosym:
        click.echo(
            'Found "nosym = true" in [phelel] section. Symmetrization is turned off.'
        )
        is_symmetry = False

    phe = phelel.load(
        str(yaml_filename),
        is_symmetry=is_symmetry,
    )
    # Forces are those of the phonon supercells when phonon_supercell_matrix is
    # given, otherwise those of the supercells of phelel. With the merge scheme
    # of site mixture, the forces are on the atoms of the unmerged supercell.
    if phe.phonon_supercell_matrix is None:
        supercell = phe.unmerged_supercell
        if supercell is None:
            supercell = phe.supercell
        cells = phe.supercells_with_displacements
        prefix = "disp"
    else:
        supercell = phe.phonon_unmerged_supercell
        if supercell is None:
            supercell = phe.phonon_supercell
        cells = phe.phonon_supercells_with_displacements
        prefix = "ph-disp"
    assert supercell is not None
    assert cells is not None
    nd = get_num_digits(cells)
    dir_names = [
        pathlib.Path(f"{dir_name}/{prefix}-{i:0{nd}d}") for i in range(len(cells) + 1)
    ]

    phonopy_yaml_filename.parent.mkdir(parents=True, exist_ok=True)

    # NAC params should be contained phelel_disp.yaml.
    # Therefore nac_params is not set to phe here.
    filenames = get_vasp_force_filenames(dir_names)
    forces = read_vasp_forces(
        filenames[1:],
        supercell,
        perfect_filename=filenames[0] if subtract_residual_forces else None,
    )
    phe.forces = forces
    phe.save_phonon(filename=phonopy_yaml_filename)
    click.echo(f'"{phonopy_yaml_filename}" has been made.')
