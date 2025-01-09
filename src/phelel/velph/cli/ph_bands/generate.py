"""Implementation of velph-ph_bands-generate."""

import pathlib
import shutil

import click
import tomli
from phonopy.interface.calculator import write_crystal_structure
from phonopy.structure.atoms import parse_cell_dict

from phelel.velph.cli.utils import (
    get_scheduler_dict,
    write_incar,
    write_kpoints_line_mode,
    write_kpoints_mesh_mode,
    write_launch_script,
)


def write_input_files(
    toml_filename: pathlib.Path,
    dir_name: str = "phelel",
) -> None:
    """Generate VASP inputs to generate phonon band structure."""
    with open(toml_filename, "rb") as f:
        toml_dict = tomli.load(f)

    main_directory_name = "ph_bands"
    main_directory = pathlib.Path(main_directory_name)
    main_directory.mkdir(parents=True, exist_ok=True)

    directory_name = f"{main_directory_name}/bands"
    directory = pathlib.Path(directory_name)
    directory.mkdir(parents=True, exist_ok=True)

    # POSCAR
    primitive = parse_cell_dict(toml_dict["primitive_cell"])
    write_crystal_structure(directory / "POSCAR", primitive)

    # INCAR
    write_incar(toml_dict["vasp"]["ph_bands"]["incar"], directory, cell=primitive)

    # KPOINTS
    write_kpoints_mesh_mode(
        toml_dict["vasp"]["ph_bands"]["incar"],
        directory,
        "vasp.ph_bands.kpoints",
        toml_dict["vasp"]["ph_bands"]["kpoints"],
    )

    # QPOINTS
    if "path" in toml_dict["vasp"]["ph_bands"]["qpoints"]:
        click.echo("Seek-path (https://github.com/giovannipizzi/seekpath) is used.")

    write_kpoints_line_mode(
        primitive,
        directory,
        "vasp.ph_bands.qpoints",
        toml_dict["vasp"]["ph_bands"]["qpoints"],
        kpoints_filename="QPOINTS",
    )

    # phelel_params.hdf5
    if pathlib.Path(f"{dir_name}/phelel_params.hdf5").exists():
        shutil.copy2(
            f"{dir_name}/phelel_params.hdf5", "ph_bands/bands/phelel_params.hdf5"
        )
    else:
        click.echo(f'"{dir_name}/bands/phelel_params.hdf5" not found.', err=True)
        return None

    # POTCAR
    if pathlib.Path("POTCAR").exists():
        shutil.copy2(pathlib.Path("POTCAR"), directory / "POTCAR")

    # Scheduler launch script
    if "scheduler" in toml_dict:
        scheduler_dict = get_scheduler_dict(toml_dict, "ph_bands")
        write_launch_script(scheduler_dict, directory)

    click.echo(f'VASP input files were made in "{directory_name}".')
