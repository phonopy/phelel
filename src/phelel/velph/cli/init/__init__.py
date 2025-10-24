"""velph command line tool / velph-init."""

from __future__ import annotations

import pathlib
import shutil
from typing import Literal

import click

from phelel.velph.cli import cmd_root
from phelel.velph.cli.init.init import run_init
from phelel.velph.cli.utils import (
    DefaultCellChoices,
    DisplacementOptions,
    PrimitiveCellChoice,
    VelphFilePaths,
    VelphInitOptions,
    VelphInitParams,
)
from phelel.velph.utils.vasp import VaspPotcar


#
# velph init
#
@cmd_root.command("init")
@click.argument("cell_filename", nargs=1, type=click.Path())
@click.argument("project_folder", nargs=1, type=click.Path())
@click.option(
    "--amplitude",
    nargs=1,
    type=float,
    default=None,
    help=(
        "Distance of displacements in Angstrom. "
        f"(amplitude: float, default={DisplacementOptions.amplitude})"
    ),
)
@click.option(
    "--cell-for-nac",
    "cell_for_nac",
    type=str,
    default=None,
    help=(
        'Cell choice for NAC, "primitive" or "unitcell" '
        f'(cell_for_nac: str, default="{DefaultCellChoices.nac.value}")'
    ),
)
@click.option(
    "--cell-for-relax",
    "cell_for_relax",
    type=str,
    default=None,
    help=(
        'Cell choice for relax, "primitive" or "unitcell" '
        f'(cell_for_relax: str, default="{DefaultCellChoices.relax.value}")'
    ),
)
@click.option(
    "--diagonal",
    "diagonal",
    type=bool,
    default=None,
    help=(
        "Generate displacements in diagonal directions or only along axes."
        f"(diagonal: bool, default={DisplacementOptions.diagonal})"
    ),
)
@click.option(
    "--dim",
    "supercell_dimension",
    nargs=3,
    type=int,
    default=None,
    help=(
        "Specify supercell dimensions by three integers like 2 2 2. "
        "For specifiying nine integers, use --supercell-matrix. "
        "This option is a shortcut for diagonal supercell matrix, where "
        "three integers correspond to the diagonal elements. "
        "(supercell_dimension: tuple[int, int, int], "
        f"default={DisplacementOptions.supercell_matrix})"
    ),
)
@click.option(
    "--force",
    "force_create",
    is_flag=True,
    default=None,
    help="Create velph.toml even if it already exists.",
)
@click.option(
    "--kspacing",
    "kspacing",
    nargs=1,
    type=float,
    default=None,
    help=(
        "Define k-point grid by KSPACING. "
        f"(kspacing: float, default={VelphInitParams.kspacing})"
    ),
)
@click.option(
    "--kspacing-dense",
    "kspacing_dense",
    nargs=1,
    type=float,
    default=None,
    help=(
        "Define dense k-point grid by KSPACING_DENSE. "
        f"(kspacing_dense: float, default={VelphInitParams.kspacing_dense})"
    ),
)
@click.option(
    "--magmom",
    "magmom",
    type=str,
    default=None,
    help=(
        'String corresponding to INCAR MAGMOM tag value, e.g., "24*1" or "0 0 1"'
        f"(magmom: str, default={VelphInitParams.magmom})"
    ),
)
@click.option(
    "--max-num-atoms",
    "max_num_atoms",
    nargs=1,
    default=None,
    type=int,
    help=(
        "Determine supercell dimension so that number of atoms in supercell "
        "is less than this number. This must be used with --symmmetrize-cell. "
        f"(max_num_atoms: int, default={DisplacementOptions.max_num_atoms})"
    ),
)
@click.option(
    "--no-find-primitive",
    "find_primitive",
    is_flag=True,
    flag_value=False,
    default=None,
    help=(
        "Disable finding primitive cell in input cell. "
        f"(find_primitive: bool, default={VelphInitParams.find_primitive})"
    ),
)
@click.option(
    "--phelel-dir-name",
    "phelel_dir_name",
    type=str,
    default="phelel",
    help=(
        'Used for backward compatibility, for which set "supercell". '
        '(phelel_dir_name: str, default="phelel")'
    ),
)
@click.option(
    "--phelel-nosym",
    "phelel_nosym",
    is_flag=True,
    default=None,
    help=(
        'Invoke "phelel --nosym". '
        f"(phelel_nosym: bool, default={VelphInitParams.phelel_nosym})"
    ),
)
@click.option(
    "--plusminus/--auto",
    "plusminus",
    type=bool,
    default=None,
    help=(
        "Plus-minus displacements in supercell, otherwise auto. "
        f"(plusminus: bool, default={DisplacementOptions.plusminus})"
    ),
)
@click.option(
    "--primitive-cell-choice",
    "primitive_cell_choice",
    type=str,
    default=None,
    help=(
        'Primitive cell choice, "standardized" or "reduced" '
        "(primitive_cell_choice: str, "
        f'default="{PrimitiveCellChoice.STANDARDIZED.value}")'
    ),
)
@click.option(
    "--supercell-matrix",
    "supercell_matrix",
    nargs=9,
    type=int,
    default=None,
    help=(
        "Specify supercell dimensions by nine integers like 0 2 2 2 0 2 2 2 0."
        "(supercell_matrix: tuple[int, ...], "
        f"default={DisplacementOptions.supercell_matrix})"
    ),
)
@click.option(
    "--symmetrize-cell",
    "symmetrize_cell",
    is_flag=True,
    default=None,
    help=(
        "Symmetrize input crystal structure. "
        f"(symmetrize_cell: bool, default={VelphInitParams.symmetrize_cell})"
    ),
)
@click.option(
    "--template-toml",
    "template_toml_filename",
    nargs=1,
    type=click.Path(),
    default=None,
    help=(
        "File name of template in toml to update velph.toml in python's "
        "dict.update() style."
    ),
)
@click.option(
    "--tolerance",
    "tolerance",
    nargs=1,
    type=float,
    default=None,
    help=(
        "Tolerance to find crystal symmetry. "
        f"(tolerance: float, default={VelphInitParams.tolerance})"
    ),
)
@click.option(
    "--toml-filename",
    "toml_filename",
    nargs=1,
    type=click.Path(),
    default=None,
    help="File name of velph.toml type file to be created.",
)
@click.option(
    "--use-grg",
    "use_grg",
    is_flag=True,
    default=None,
    help=(
        "Use generalized regular grid when needed. "
        f"(use_grg: bool, default={VelphInitParams.use_grg})"
    ),
)
@click.help_option("-h", "--help")
def cmd_init(
    amplitude: float | None,
    cell_filename: str,
    cell_for_nac: Literal["primitive", "unitcell"] | None,
    cell_for_relax: Literal["primitive", "unitcell"] | None,
    diagonal: bool | None,
    find_primitive: bool | None,
    force_create: bool | None,
    kspacing: float | None,
    kspacing_dense: float | None,
    magmom: str | None,
    max_num_atoms: int | None,
    phelel_dir_name: str,
    phelel_nosym: bool | None,
    plusminus: bool | None,
    primitive_cell_choice: str | None,
    project_folder: str,
    supercell_dimension: tuple[int, int, int] | None,
    supercell_matrix: tuple[int, int, int, int, int, int, int, int, int] | None,
    symmetrize_cell: bool | None,
    template_toml_filename: str | None,
    toml_filename: str | None,
    tolerance: float | None,
    use_grg: bool | None,
):
    """Initialize an electron phonon calculation project.

    Crystal structure (CELL_FILENAME) and new folder where new velph project
    is created have to be specified as command-line arguments.

    Some command options can be specified in the [init.options] section of the
    velph.toml-template file. Each option's key and its corresponding type are
    indicated in parentheses within echo help documentation, for example,

    \b
    [init.options]
    kspacing = 0.2
    kspacing_dense = 0.1
    max_num_atoms = 120

    """  # noqa: D301
    if not pathlib.Path(cell_filename).exists():
        click.echo(f'"{cell_filename}" not found.', err=True)
        return

    vip_cmd_options = VelphInitOptions(
        **{
            "amplitude": amplitude,
            "cell_for_nac": cell_for_nac,
            "cell_for_relax": cell_for_relax,
            "diagonal": diagonal,
            "find_primitive": find_primitive,
            "kspacing": kspacing,
            "kspacing_dense": kspacing_dense,
            "magmom": magmom,
            "max_num_atoms": max_num_atoms,
            "phelel_nosym": phelel_nosym,
            "plusminus": plusminus,
            "primitive_cell_choice": primitive_cell_choice,
            "supercell_dimension": supercell_dimension,
            "supercell_matrix": supercell_matrix,
            "symmetrize_cell": symmetrize_cell,
            "tolerance": tolerance,
            "use_grg": use_grg,
        }
    )

    cell_filepath = pathlib.Path(cell_filename)
    if cell_filepath.exists():
        vfp_dict = {"cell_filepath": cell_filepath}
    else:
        click.echo(f'"{cell_filename}" not found.', err=True)
        return

    project_folder_path = pathlib.Path(project_folder)
    if project_folder_path.exists():
        if project_folder_path.samefile(pathlib.Path()):
            click.echo(f'Project directory: "{project_folder}".')
        else:
            click.echo(f'File or folder "{project_folder}" already exists.')
            if not force_create:
                return

    if template_toml_filename is not None:
        velph_tmpl_filepath = pathlib.Path(template_toml_filename)
        if velph_tmpl_filepath.exists():
            vfp_dict["velph_template_filepath"] = velph_tmpl_filepath
        else:
            click.echo(f'"{template_toml_filename}" not found.', err=True)
            return

    if toml_filename is None:
        toml_filepath = project_folder_path / "velph.toml"
    else:
        toml_filepath = project_folder_path / toml_filename

    if toml_filepath.exists():
        if force_create:
            click.echo(f'"{toml_filepath}" exists, but will be overwritten.', err=True)
        else:
            click.echo(
                f'"{toml_filepath}" was not overwritten because it exists.', err=True
            )
            return

    vfp = VelphFilePaths(**vfp_dict)
    toml_lines = run_init(vip_cmd_options, vfp, phelel_dir_name=phelel_dir_name)

    # Write velph.toml.
    if toml_lines:
        if not project_folder_path.exists():
            project_folder_path.mkdir()
            click.echo(f'Created new folder "{project_folder}".')
        with open(toml_filepath, "w") as w:
            w.write("\n".join(toml_lines))
            w.write("\n")
        click.echo(f'Initial settings were written to "{toml_filepath}".')
        if pathlib.Path("POTCAR").exists():
            vasp_potcar = VaspPotcar("POTCAR")
            click.echo('Found "POTCAR".')
            for elem in vasp_potcar.titel:
                print(f"  {elem}")
            enmax = max(vasp_potcar.enmax)
            click.echo(f'  Max ENMAX in "POTCAR" is {enmax}.')
            if project_folder is not None:
                if not (project_folder_path / "POTCAR").exists():
                    shutil.copy2(pathlib.Path("POTCAR"), project_folder_path / "POTCAR")
                    click.echo(f'"POTCAR" was copied to "{project_folder}/POTCAR".')
    else:
        click.echo("")
        click.echo(f'"{toml_filepath}" was not created.')
