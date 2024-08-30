"""Implementation of velph-supercell-generate."""

from __future__ import annotations

import pathlib
import shutil
from typing import Union

import click
import tomli
from phono3py import Phono3py
from phonopy.interface.calculator import write_crystal_structure

import phelel
from phelel import Phelel
from phelel.velph.cli.utils import (
    get_num_digits,
    get_scheduler_dict,
    kspacing_to_mesh,
    write_incar,
    write_kpoints_mesh_mode,
    write_launch_script,
)


def write_supercell_input_files(
    toml_filename: pathlib.Path,
    phelel_yaml_filename: pathlib.Path,
    dir_name: str = "phelel",
) -> None:
    """Generate supercells."""
    if not phelel_yaml_filename.exists():
        click.echo(f'File "{phelel_yaml_filename}" not found.', err=True)
        click.echo('Run "velph phelel init" if necessary.', err=True)
        return None

    phe = phelel.load(phelel_yaml_filename)
    with open(toml_filename, "rb") as f:
        toml_dict = tomli.load(f)

    write_supercells(phe, toml_dict, dir_name=dir_name)
    if phe.phonon_supercell_matrix is not None:
        if "phonon" in toml_dict["vasp"][dir_name]:
            write_phonon_supercells(phe, toml_dict, dir_name=dir_name)
        else:
            print(f'[vasp.{dir_name}.phonon.*] not found in "{toml_filename}"')


def write_supercells(
    phe: Union[Phelel, Phono3py], toml_dict: dict, dir_name: str = "phelel"
):
    """Write VASP input for supercells.

    This is alos used by velph-phono3py-generate.

    """
    kpoints_dict = toml_dict["vasp"][dir_name]["kpoints"]
    if "kspacing" in kpoints_dict:
        symmetry_dataset = kspacing_to_mesh(kpoints_dict, phe.supercell)
        if "symmetry" in toml_dict and "spacegroup_type" in toml_dict["symmetry"]:
            assert (
                symmetry_dataset.international
                == toml_dict["symmetry"]["spacegroup_type"]
            )
    nd = get_num_digits(phe.supercells_with_displacements)

    for i, cell in enumerate(
        [
            phe.supercell,
        ]
        + phe.supercells_with_displacements
    ):
        id_number = f"{i:0{nd}d}"
        disp_dir_name = f"{dir_name}/disp-{id_number}"
        directory = pathlib.Path(disp_dir_name)
        directory.mkdir(parents=True, exist_ok=True)

        _write_vasp_files(
            directory,
            cell,
            toml_dict["vasp"][dir_name]["incar"],
            dir_name,
            kpoints_dict,
        )

        # Scheduler launch script
        if "scheduler" in toml_dict:
            scheduler_dict = get_scheduler_dict(toml_dict, dir_name)
            write_launch_script(scheduler_dict, directory, job_id=id_number)

        click.echo(f'VASP input files were generated in "{disp_dir_name}".')


def write_phonon_supercells(
    phe: Union[Phelel, Phono3py], toml_dict: dict, dir_name: str = "phelel"
):
    """Write VASP input for phonon supercells.

    This is alos used by velph-phono3py-generate.

    """
    kpoints_dict = toml_dict["vasp"][dir_name]["phonon"]["kpoints"]
    nd = get_num_digits(phe.phonon_supercells_with_displacements)

    for i, cell in enumerate(
        [
            phe.phonon_supercell,
        ]
        + phe.phonon_supercells_with_displacements
    ):
        id_number = f"{i:0{nd}d}"
        disp_dir_name = f"{dir_name}/ph-disp-{id_number}"
        directory = pathlib.Path(disp_dir_name)
        directory.mkdir(parents=True, exist_ok=True)

        _write_vasp_files(
            directory,
            cell,
            toml_dict["vasp"][dir_name]["phonon"]["incar"],
            f"{dir_name}.phonon",
            kpoints_dict,
        )

        # Scheduler launch script
        if "scheduler" in toml_dict:
            scheduler_dict = get_scheduler_dict(toml_dict, f"{dir_name}.phonon")
            write_launch_script(scheduler_dict, directory, job_id=id_number)

        click.echo(f'VASP input files were generated in "{disp_dir_name}".')


def _write_vasp_files(directory, cell, toml_incar_dict, dir_name, kpoints_dict):
    # POSCAR
    write_crystal_structure(directory / "POSCAR", cell)

    # INCAR
    write_incar(toml_incar_dict, directory, cell=cell)

    # KPOINTS
    write_kpoints_mesh_mode(
        toml_incar_dict,
        directory,
        f"vasp.{dir_name}.kpoints",
        kpoints_dict,
    )

    # POTCAR
    potcar_path = pathlib.Path("POTCAR")
    if potcar_path.exists():
        shutil.copy2(potcar_path, directory / potcar_path)
