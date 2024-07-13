"""Implementation of velph-supercell-generate."""

from __future__ import annotations

import pathlib
import shutil

import click
import tomli
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
) -> None:
    """Generate supercells."""
    if not phelel_yaml_filename.exists():
        click.echo(f'File "{phelel_yaml_filename}" not found.', err=True)
        click.echo('Run "velph supercell init" if necessary.', err=True)
        return None

    phe = phelel.load(phelel_yaml_filename)
    with open(toml_filename, "rb") as f:
        toml_dict = tomli.load(f)

    _write_supercells(phe, toml_dict)
    if phe.phonon_supercell_matrix is not None:
        if "phonon" in toml_dict["vasp"]["supercell"]:
            _write_phonon_supercells(phe, toml_dict)
        else:
            print(f'[vasp.supercell.phonon.*] not found in "{toml_filename}"')


def _write_supercells(phe: Phelel, toml_dict: dict):
    kpoints_dict = toml_dict["vasp"]["supercell"]["kpoints"]
    if "kspacing" in kpoints_dict:
        symmetry_dataset = kspacing_to_mesh(kpoints_dict, phe.supercell)
        if "symmetry" in toml_dict and "spacegroup_type" in toml_dict["symmetry"]:
            assert (
                symmetry_dataset["international"]
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
        dir_name = f"supercell/disp-{id_number}"
        directory = pathlib.Path(dir_name)
        directory.mkdir(parents=True, exist_ok=True)

        # POSCAR
        write_crystal_structure(directory / "POSCAR", cell)

        # INCAR
        write_incar(toml_dict["vasp"]["supercell"]["incar"], directory, cell=cell)

        # KPOINTS
        write_kpoints_mesh_mode(
            toml_dict["vasp"]["supercell"]["incar"],
            directory,
            "vasp.supercell.kpoints",
            kpoints_dict,
        )

        # POTCAR
        potcar_path = pathlib.Path("POTCAR")
        if potcar_path.exists():
            shutil.copy2(potcar_path, directory / potcar_path)

        # Scheduler launch script
        if "scheduler" in toml_dict:
            scheduler_dict = get_scheduler_dict(toml_dict, "supercell")
            write_launch_script(scheduler_dict, directory, job_id=id_number)

        click.echo(f'VASP input files were generated in "{dir_name}".')


def _write_phonon_supercells(phe: Phelel, toml_dict: dict):
    kpoints_dict = toml_dict["vasp"]["supercell"]["phonon"]["kpoints"]
    nd = get_num_digits(phe.phonon_supercells_with_displacements)

    for i, cell in enumerate(
        [
            phe.phonon_supercell,
        ]
        + phe.phonon_supercells_with_displacements
    ):
        id_number = f"{i:0{nd}d}"
        dir_name = f"supercell/ph-disp-{id_number}"
        directory = pathlib.Path(dir_name)
        directory.mkdir(parents=True, exist_ok=True)

        # POSCAR
        write_crystal_structure(directory / "POSCAR", cell)

        # INCAR
        write_incar(
            toml_dict["vasp"]["supercell"]["phonon"]["incar"], directory, cell=cell
        )

        # KPOINTS
        write_kpoints_mesh_mode(
            toml_dict["vasp"]["supercell"]["phonon"]["incar"],
            directory,
            "vasp.supercell.phonon.kpoints",
            kpoints_dict,
        )

        # POTCAR
        potcar_path = pathlib.Path("POTCAR")
        if potcar_path.exists():
            shutil.copy2(potcar_path, directory / potcar_path)

        # Scheduler launch script
        if "scheduler" in toml_dict:
            scheduler_dict = get_scheduler_dict(toml_dict, ["supercell", "phonon"])
            write_launch_script(scheduler_dict, directory, job_id=id_number)

        click.echo(f'VASP input files were generated in "{dir_name}".')
