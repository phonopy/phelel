"""Implementation of velph-relax-generate."""

import pathlib
import shutil
from typing import Optional

import click
import tomli
from phonopy.interface.calculator import write_crystal_structure

from phelel.velph.cli.utils import (
    assert_kpoints_mesh_symmetry,
    choose_cell_in_dict,
    get_scheduler_dict,
    write_incar,
    write_kpoints_mesh_mode,
    write_launch_script,
)


def write_input_files(
    toml_filename: pathlib.Path,
    directory: pathlib.Path,
    prev_directory=Optional[pathlib.Path],
) -> None:
    """Generate VASP relax inputs."""
    with open(toml_filename, "rb") as f:
        toml_dict = tomli.load(f)

    directory.mkdir(parents=True, exist_ok=True)

    # POSCAR
    cell = choose_cell_in_dict(toml_dict, toml_filename, "relax")
    if prev_directory is None:
        write_crystal_structure(directory / "POSCAR", cell)
    else:
        contcar_path = prev_directory / "CONTCAR"
        if contcar_path.exists():
            shutil.copy2(contcar_path, directory / "POSCAR")
            click.echo(f'"{contcar_path}" will be as new "POSCAR".', err=True)
        else:
            click.echo(f'"{contcar_path}" not found.', err=True)
            return None

    # INCAR
    write_incar(toml_dict["vasp"]["relax"]["incar"], directory, cell=cell)

    # KPOINTS
    kpoints_dict = toml_dict["vasp"]["relax"]["kpoints"]
    assert_kpoints_mesh_symmetry(toml_dict, kpoints_dict, cell)
    write_kpoints_mesh_mode(
        toml_dict["vasp"]["relax"]["incar"],
        directory,
        "vasp.relax.kpoints",
        toml_dict["vasp"]["relax"]["kpoints"],
    )

    # POTCAR
    if pathlib.Path("POTCAR").exists():
        shutil.copy2(pathlib.Path("POTCAR"), directory / "POTCAR")

    # Scheduler launch script
    if "scheduler" in toml_dict:
        scheduler_dict = get_scheduler_dict(toml_dict, "relax")
        write_launch_script(
            scheduler_dict, directory, job_id=str(directory).split("/")[-1].strip()
        )

    click.echo(f'VASP input files were made in "{directory}".')
