"""Write VASP input files of supercells with displacements.

These are used by velph phelel generate, velph phonopy generate and velph
phono3py generate.

"""

from __future__ import annotations

import pathlib
import shutil

import click
from phono3py import Phono3py
from phonopy import Phonopy
from phonopy.structure.cells import Supercell

from phelel import Phelel
from phelel.velph.cli.utils import (
    echo_vasp_vca_hint,
    get_num_digits,
    get_scheduler_dict,
    kspacing_to_mesh,
    write_incar,
    write_kpoints_mesh_mode,
    write_launch_script,
    write_poscar,
)


def write_supercells(
    phe: Phelel | Phonopy | Phono3py, toml_dict: dict, dir_name: str = "phelel"
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
    assert phe.supercells_with_displacements is not None
    nd = get_num_digits(phe.supercells_with_displacements)
    # With the merge scheme of site mixture, the atoms of the supercell for the
    # calculator are those of the unmerged supercell.
    if isinstance(phe, (Phelel, Phonopy)) and phe.unmerged_supercell is not None:
        supercell = phe.unmerged_supercell
    else:
        supercell = phe.supercell

    for i, cell in enumerate([supercell] + phe.supercells_with_displacements):
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

    echo_vasp_vca_hint(supercell)


def write_phonon_supercells(
    phe: Phelel | Phono3py, toml_dict: dict, dir_name: str = "phelel"
):
    """Write VASP input for phonon supercells.

    This is alos used by velph-phono3py-generate.

    """
    kpoints_dict = toml_dict["vasp"][dir_name]["phonon"]["kpoints"]
    assert phe.phonon_supercells_with_displacements is not None
    nd = get_num_digits(phe.phonon_supercells_with_displacements)
    phonon_supercell: Supercell | None
    if isinstance(phe, Phelel) and phe.phonon_unmerged_supercell is not None:
        phonon_supercell = phe.phonon_unmerged_supercell
    else:
        phonon_supercell = phe.phonon_supercell
    assert phonon_supercell is not None

    for i, cell in enumerate(
        [phonon_supercell] + phe.phonon_supercells_with_displacements
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

    echo_vasp_vca_hint(phonon_supercell)


def _write_vasp_files(directory, cell, toml_incar_dict, dir_name, kpoints_dict):
    # POSCAR
    write_poscar(directory, cell)

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
