"""Implementation of velph-relax-generate."""

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


def write_input_files(
    toml_filename: pathlib.Path,
    directory: pathlib.Path,
    prev_directory: pathlib.Path | None = None,
) -> None:
    """Generate VASP relax inputs."""
    with open(toml_filename, "rb") as f:
        toml_dict = tomli.load(f)

    directory.mkdir(parents=True, exist_ok=True)

    # POSCAR
    cell = choose_cell_in_dict(toml_dict, toml_filename, "relax")
    assert cell is not None
    if prev_directory is None:
        write_poscar(directory, cell)
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
    echo_vasp_vca_hint(cell)
