"""Implementation of velph-phelel-differentiate."""

from __future__ import annotations

import pathlib

import click
import tomli

import phelel
from phelel.interface.vasp.derivatives import read_forces_from_vasprunxmls
from phelel.velph.cli.utils import get_num_digits


def create_phonopy_yaml(
    toml_filename: pathlib.Path,
    yaml_filename: pathlib.Path,
    dir_name: str,
    subtract_residual_forces: bool = True,
):
    """Calculate derivatives and write phelel_params.hdf5."""
    phonopy_yaml_filename = pathlib.Path(f"{dir_name}/phonopy_params.yaml")

    with open(toml_filename, "rb") as f:
        toml_dict = tomli.load(f)

    is_symmetry = True
    try:
        if toml_dict["phelel"]["nosym"] is True:
            click.echo(
                'Found "nosym = true" in [phelel] section. '
                "Symmetrization is turned off."
            )
            is_symmetry = False
    except KeyError:
        pass

    phe = phelel.load(
        str(yaml_filename),
        is_symmetry=is_symmetry,
    )
    dir_names = []

    assert phe.supercells_with_displacements is not None
    nd = get_num_digits(phe.supercells_with_displacements)
    for i, _ in enumerate(
        [
            phe.supercell,
        ]
        + phe.supercells_with_displacements
    ):
        id_number = f"{i:0{nd}d}"
        dir_names.append(pathlib.Path(f"{dir_name}/disp-{id_number}"))

    if phe.phonon_supercell_matrix is not None:
        assert phe.phonon_supercells_with_displacements is not None
        nd = get_num_digits(phe.phonon_supercells_with_displacements)
        for i, _ in enumerate(
            [
                phe.phonon_supercell,
            ]
            + phe.phonon_supercells_with_displacements
        ):
            id_number = f"{i:0{nd}d}"
            dir_names.append(pathlib.Path(f"{phelel}/ph-disp-{id_number}"))

    phonopy_yaml_filename.parent.mkdir(parents=True, exist_ok=True)

    # NAC params should be contained phelel_disp.yaml.
    # Therefore nac_params is not set to phe here.
    if phe.phonon_supercell_matrix is None:
        supercell = phe.supercell
    else:
        supercell = phe.phonon_supercell
    assert supercell is not None
    forces = read_forces_from_vasprunxmls(
        [d / "vasprun.xml" for d in dir_names],
        supercell,
        subtract_rfs=subtract_residual_forces,
        log_level=0,
    )
    phe.forces = forces
    phe.save_phonon(filename=phonopy_yaml_filename)
    click.echo(f'"{phonopy_yaml_filename}" has been made.')
