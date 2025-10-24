"""Implementation of velph-relax-generate."""

import copy
import pathlib
import shutil

import click
import tomli
from phonopy.interface.calculator import write_crystal_structure
from phonopy.structure.atoms import parse_cell_dict

from phelel.velph.cli.utils import (
    assert_kpoints_mesh_symmetry,
    get_scheduler_dict,
    write_incar,
    write_kpoints_line_mode,
    write_kpoints_mesh_mode,
    write_launch_script,
)


def write_input_files(toml_filename: pathlib.Path) -> None:
    """Generate VASP inputs to generate electronc band structure."""
    with open(toml_filename, "rb") as f:
        toml_dict = tomli.load(f)

    main_directory_name = "el_bands"
    main_directory = pathlib.Path(main_directory_name)
    main_directory.mkdir(parents=True, exist_ok=True)

    for calc_type in ("bands", "dos"):
        directory_name = f"{main_directory_name}/{calc_type}"
        directory = pathlib.Path(directory_name)
        directory.mkdir(parents=True, exist_ok=True)

        # POSCAR
        primitive = parse_cell_dict(toml_dict["primitive_cell"])
        write_crystal_structure(directory / "POSCAR", primitive)

        # INCAR
        incar_dict = copy.deepcopy(toml_dict["vasp"]["el_bands"][calc_type]["incar"])
        write_incar(incar_dict, directory, cell=primitive)

        # KPOINTS
        kpoints_dict = toml_dict["vasp"]["el_bands"][calc_type]["kpoints"]
        assert_kpoints_mesh_symmetry(toml_dict, kpoints_dict, primitive)
        write_kpoints_mesh_mode(
            toml_dict["vasp"]["el_bands"][calc_type]["incar"],
            directory,
            f"vasp.el_bands.{calc_type}.kpoints",
            toml_dict["vasp"]["el_bands"][calc_type]["kpoints"],
        )

        # KPOINTS_OPT
        if calc_type == "bands":
            if "path" in toml_dict["vasp"]["el_bands"][calc_type]["kpoints_opt"]:
                click.echo(
                    "Seek-path (https://github.com/giovannipizzi/seekpath) is used."
                )
            write_kpoints_line_mode(
                primitive,
                directory,
                "vasp.el_bands.bands.kpoints_opt",
                toml_dict["vasp"]["el_bands"][calc_type]["kpoints_opt"],
                kpoints_filename="KPOINTS_OPT",
            )
        elif calc_type == "dos":
            if "kpoints_dense" not in toml_dict["vasp"]["el_bands"][calc_type]:
                raise RuntimeError(
                    "[vasp.el_bands.dos.kpoints_dense] section is necessary "
                    "for electronic DOS calculation."
                )
            kpoints_dense_dict = toml_dict["vasp"]["el_bands"][calc_type][
                "kpoints_dense"
            ]
            assert_kpoints_mesh_symmetry(toml_dict, kpoints_dense_dict, primitive)
            write_kpoints_mesh_mode(
                toml_dict["vasp"]["el_bands"][calc_type]["incar"],
                directory,
                "vasp.el_bands.dos.kpoints_dense",
                kpoints_dense_dict,
                kpoints_filename="KPOINTS_OPT",
            )

        # POTCAR
        if pathlib.Path("POTCAR").exists():
            shutil.copy2(pathlib.Path("POTCAR"), directory / "POTCAR")

        # Scheduler launch script
        if "scheduler" in toml_dict:
            scheduler_dict = get_scheduler_dict(toml_dict, "el_bands")
            write_launch_script(scheduler_dict, directory)

        click.echo(f'VASP input files were made in "{directory_name}".')
