"""Implementation of velph-nac-generate."""

import pathlib
import shutil

import click
import tomli

from phelel.velph.cli.utils import (
    assert_kpoints_mesh_symmetry,
    choose_cell_in_dict,
    echo_vasp_vca_hint,
    get_scheduler_dict,
    write_incar,
    write_kpoints_mesh_mode,
    write_launch_script,
    write_poscar,
)


def write_input_files(toml_filename: pathlib.Path) -> None:
    """Generate VASP nac inputs."""
    with open(toml_filename, "rb") as f:
        toml_dict = tomli.load(f)

    directory_name = "nac"
    directory = pathlib.Path(directory_name)
    directory.mkdir(parents=True, exist_ok=True)

    # POSCAR
    cell = choose_cell_in_dict(toml_dict, toml_filename, "nac")
    assert cell is not None
    write_poscar(directory, cell)

    # INCAR
    write_incar(toml_dict["vasp"]["nac"]["incar"], directory, cell=cell)

    # KPOINTS
    kpoints_dict = toml_dict["vasp"]["nac"]["kpoints"]
    assert_kpoints_mesh_symmetry(toml_dict, kpoints_dict, cell)
    write_kpoints_mesh_mode(
        toml_dict["vasp"]["nac"]["incar"],
        directory,
        "vasp.nac.kpoints",
        toml_dict["vasp"]["nac"]["kpoints"],
    )

    # POTCAR
    if pathlib.Path("POTCAR").exists():
        shutil.copy2(pathlib.Path("POTCAR"), directory / "POTCAR")

    # Scheduler launch script
    if "scheduler" in toml_dict:
        scheduler_dict = get_scheduler_dict(toml_dict, "nac")
        write_launch_script(scheduler_dict, directory)

    click.echo(f'VASP input files were made in "{directory_name}".')
    echo_vasp_vca_hint(cell)
