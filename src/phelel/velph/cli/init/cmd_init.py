"""velph command line tool / velph-init."""

from __future__ import annotations

import pathlib
import shutil
from typing import Any, Literal

import click
from phonopy.interface.calculator import read_crystal_structure

from phelel.velph.cli.init.build import build_velph_config
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.defaults import get_default_config
from phelel.velph.config.io import (
    VelphConfigError,
    load_template,
    parse_init,
    write_velph_toml,
)
from phelel.velph.config.schema import (
    CODES,
    CodeName,
    InitSection,
    InputSetName,
    VelphConfig,
)
from phelel.velph.utils.vasp import VaspPotcar

_DEFAULTS = get_default_config()

# Keys of [init] for the supercell and their options.
_SUPERCELL_OPTIONS = {
    "supercell_dimension": "--dim",
    "supercell_matrix": "--supercell-matrix",
    "max_num_atoms": "--max-num-atoms",
}


def _init_default(name: str) -> Any:
    """Return the default of a key of [init] that the defaults give."""
    assert _DEFAULTS.init is not None
    return getattr(_DEFAULTS.init, name)


def _code_default(name: str) -> str:
    """Return the default of a field of [phelel], [phonopy] and [phono3py]."""
    values = {code: getattr(getattr(_DEFAULTS, code), name) for code in CODES}
    if len(set(values.values())) == 1:
        return str(values[CODES[0]])
    return ", ".join(f"{value} for {code}" for code, value in values.items())


def _cell_default(set_name: InputSetName) -> str:
    """Return the default cell of an input set."""
    assert _DEFAULTS.vasp is not None and _DEFAULTS.vasp.input_sets is not None
    cell = _DEFAULTS.vasp.input_sets[set_name].cell
    assert cell is not None
    return cell


#
# velph init
#
@cmd_root.command("init")
@click.argument(
    "cell_filename",
    nargs=1,
    type=click.Path(),
)
@click.argument(
    "project_folder",
    nargs=1,
    type=click.Path(),
)
@click.option(
    "--amplitude",
    "amplitude",
    type=float,
    default=None,
    help=(
        "Distance of displacements in Angstrom. "
        f"(amplitude: float, default={_code_default('amplitude')})"
    ),
)
@click.option(
    "--cell-for-nac",
    "cell_for_nac",
    type=click.Choice(["primitive", "unitcell"], case_sensitive=False),
    default=None,
    help=(
        'Cell choice for NAC, "primitive" or "unitcell" '
        f'(cell_for_nac: str, default="{_cell_default("nac")}")'
    ),
)
@click.option(
    "--cell-for-relax",
    "cell_for_relax",
    type=click.Choice(["primitive", "unitcell"], case_sensitive=False),
    default=None,
    help=(
        'Cell choice for relax, "primitive" or "unitcell" '
        f'(cell_for_relax: str, default="{_cell_default("relax")}")'
    ),
)
@click.option(
    "--code",
    "codes",
    type=click.Choice(CODES),
    multiple=True,
    help=(
        "Code whose section is written in velph.toml, repeated for more than "
        "one code, e.g., --code phelel --code phonopy. "
        f"(codes: list[str], default={list(_init_default('codes'))})"
    ),
)
@click.option(
    "--diagonal",
    "diagonal",
    type=bool,
    default=None,
    help=(
        "Generate displacements in diagonal directions or only along axes. "
        f"(diagonal: bool, default={_code_default('diagonal')})"
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
        "To give nine integers, use --supercell-matrix. "
        "This option is a shortcut for diagonal supercell matrix, where "
        "three integers correspond to the diagonal elements. "
        "(supercell_dimension: tuple[int, int, int], default=None)"
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
    type=float,
    default=None,
    help=(
        "Spacing of k-points in 1/Angstrom. velph init computes the mesh of "
        "[vasp.<input set>.kpoints] from it when the block does not give mesh "
        "or kspacing, by the formula of the VASP INCAR tag KSPACING. "
        f"(kspacing: float, default={_init_default('kspacing')})"
    ),
)
@click.option(
    "--kspacing-dense",
    "kspacing_dense",
    type=float,
    default=None,
    help=(
        "Spacing of k-points in 1/Angstrom for [vasp.<input set>.kpoints_dense], "
        "used in the same way as --kspacing. "
        f"(kspacing_dense: float, default={_init_default('kspacing_dense')})"
    ),
)
@click.option(
    "--magmom",
    "magmom",
    type=str,
    default=None,
    help=(
        'String corresponding to INCAR MAGMOM tag value, e.g., "24*1" or "0 0 1" '
        f"(magmom: str, default={_init_default('magmom')})"
    ),
)
@click.option(
    "--max-num-atoms",
    "max_num_atoms",
    default=None,
    type=int,
    help=(
        "Determine supercell dimension so that the supercell has at most this "
        "number of atoms. This must be used with --symmetrize-cell. "
        "(max_num_atoms: int, default=None)"
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
        f"(find_primitive: bool, default={_init_default('find_primitive')})"
    ),
)
@click.option(
    "--nosym",
    "nosym",
    is_flag=True,
    default=None,
    help=(
        "Generate displacements without symmetry, as phonopy --nosym. "
        f"(nosym: bool, default={_code_default('nosym')})"
    ),
)
@click.option(
    "--plusminus/--auto",
    "plusminus",
    type=bool,
    default=None,
    help=(
        "Plus-minus displacements in supercell, otherwise auto. "
        f"(plusminus: bool or 'auto', default={_code_default('plusminus')})"
    ),
)
@click.option(
    "--primitive-cell-choice",
    "primitive_cell_choice",
    type=click.Choice(["standardized", "reduced"], case_sensitive=False),
    default=None,
    help=(
        'Primitive cell choice, "standardized" or "reduced" '
        "(primitive_cell_choice: str, "
        f'default="{_init_default("primitive_cell_choice")}")'
    ),
)
@click.option(
    "--site-mixture",
    "site_mixture",
    type=str,
    default=None,
    help=(
        "[Experimental] Per-atom concentration weights in input-cell atom "
        'order, e.g., "0.5 0.5 0.5 0.5". Weights of co-located atoms (same '
        "fractional position) must sum to 1.0; an isolated atom must be 1.0. "
        "By default the co-located atoms are merged into mixed-species sites, "
        "as in phonopy; pass --split-site-mixture to keep them as separate "
        "weighted species. "
        "Cannot be combined with --magmom. "
        f"(site_mixture: str, default={_init_default('site_mixture')})"
    ),
)
@click.option(
    "--split-site-mixture",
    "split_site_mixture",
    is_flag=True,
    default=None,
    help=(
        "[Experimental] With --site-mixture, keep co-located atoms as separate "
        "weighted species instead of merging them into mixed-species sites. "
        f"(split_site_mixture: bool, default={_init_default('split_site_mixture')})"
    ),
)
@click.option(
    "--supercell-matrix",
    "supercell_matrix",
    nargs=9,
    type=int,
    default=None,
    help=(
        "Specify supercell dimensions by nine integers like 0 2 2 2 0 2 2 2 0. "
        "(supercell_matrix: tuple[int, ...], default=None)"
    ),
)
@click.option(
    "--symmetrize-cell",
    "symmetrize_cell",
    is_flag=True,
    default=None,
    help=(
        "Symmetrize input crystal structure. "
        f"(symmetrize_cell: bool, default={_init_default('symmetrize_cell')})"
    ),
)
@click.option(
    "--template-toml",
    "template_toml_filename",
    type=click.Path(),
    default=None,
    help="File name of velph init template in toml.",
)
@click.option(
    "--tolerance",
    "tolerance",
    type=float,
    default=None,
    help=(
        "Tolerance to find crystal symmetry. "
        f"(tolerance: float, default={_init_default('tolerance')})"
    ),
)
@click.option(
    "--toml-filename",
    "toml_filename",
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
        f"(use_grg: bool, default={_init_default('use_grg')})"
    ),
)
@click.help_option("-h", "--help")
def cmd_init(
    amplitude: float | None,
    cell_filename: str,
    cell_for_nac: Literal["primitive", "unitcell"] | None,
    cell_for_relax: Literal["primitive", "unitcell"] | None,
    codes: tuple[CodeName, ...],
    diagonal: bool | None,
    find_primitive: bool | None,
    force_create: bool | None,
    kspacing: float | None,
    kspacing_dense: float | None,
    magmom: str | None,
    max_num_atoms: int | None,
    nosym: bool | None,
    plusminus: bool | None,
    primitive_cell_choice: Literal["standardized", "reduced"] | None,
    project_folder: str,
    site_mixture: str | None,
    split_site_mixture: bool | None,
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

    The command options except --force, --template-toml and --toml-filename
    can be specified in the [init] section of the velph init template.  Each
    option's key and its corresponding type are indicated in parentheses
    within the help of the option, for example,

    \b
    [init]
    codes = ["phelel", "phonopy"]
    kspacing = 0.2
    kspacing_dense = 0.1
    max_num_atoms = 120

    """  # noqa: D301
    options = _init_section(
        {
            "codes": codes or None,
            "supercell_dimension": supercell_dimension,
            "supercell_matrix": supercell_matrix,
            "max_num_atoms": max_num_atoms,
            "amplitude": amplitude,
            "diagonal": diagonal,
            "plusminus": {True: True, False: "auto", None: None}[plusminus],
            "nosym": nosym,
            "cell_for_relax": cell_for_relax,
            "cell_for_nac": cell_for_nac,
            "tolerance": tolerance,
            "symmetrize_cell": symmetrize_cell,
            "find_primitive": find_primitive,
            "primitive_cell_choice": primitive_cell_choice,
            "use_grg": use_grg,
            "kspacing": kspacing,
            "kspacing_dense": kspacing_dense,
            "magmom": magmom,
            "site_mixture": site_mixture,
            "split_site_mixture": split_site_mixture,
        }
    )

    cell_filepath = pathlib.Path(cell_filename)
    if not cell_filepath.exists():
        raise click.ClickException(f'"{cell_filename}" not found.')
    template_filepath = None
    if template_toml_filename is not None:
        template_filepath = pathlib.Path(template_toml_filename)
        if not template_filepath.exists():
            raise click.ClickException(f'"{template_toml_filename}" not found.')

    project_folder_path = pathlib.Path(project_folder)
    if project_folder_path.exists():
        if not project_folder_path.is_dir():
            raise click.ClickException(f'"{project_folder}" is not a folder.')
        if project_folder_path.samefile(pathlib.Path()):
            click.echo(f'Project directory: "{project_folder}".')
        elif not force_create:
            raise click.ClickException(
                f'Folder "{project_folder}" already exists.  Use --force to write '
                "velph.toml in it."
            )

    if toml_filename is None:
        toml_filepath = project_folder_path / "velph.toml"
    else:
        toml_filepath = project_folder_path / toml_filename
    if toml_filepath.exists():
        if not force_create:
            raise click.ClickException(
                f'"{toml_filepath}" exists.  Use --force to overwrite it.'
            )
        click.echo(f'"{toml_filepath}" exists, but will be overwritten.')

    input_cell, _ = read_crystal_structure(cell_filepath, interface_mode="vasp")
    if input_cell is None:
        raise click.ClickException(
            f'Crystal structure file "{cell_filename}" could not be read.'
        )
    click.echo(f'Read crystal structure file "{cell_filename}".')

    template: VelphConfig | None = None
    if template_filepath is not None:
        template = load_template(template_filepath)
        click.echo(f'Read velph init template "{template_toml_filename}".')

    config = build_velph_config(input_cell, template, options)

    if not project_folder_path.exists():
        project_folder_path.mkdir()
        click.echo(f'Created new folder "{project_folder}".')
    write_velph_toml(config, toml_filepath)
    click.echo(f'Initial settings were written to "{toml_filepath}".')

    if pathlib.Path("POTCAR").exists():
        vasp_potcar = VaspPotcar("POTCAR")
        click.echo('Found "POTCAR".')
        for elem in vasp_potcar.titel:
            click.echo(f"  {elem}")
        enmax = max(vasp_potcar.enmax)
        click.echo(f'  Max ENMAX in "POTCAR" is {enmax}.')
        if not (project_folder_path / "POTCAR").exists():
            shutil.copy2(pathlib.Path("POTCAR"), project_folder_path / "POTCAR")
            click.echo(f'"POTCAR" was copied to "{project_folder}/POTCAR".')


def _init_section(options: dict[str, Any]) -> InitSection:
    """Return InitSection of the options given on the command line.

    options maps the keys of [init] to the values of the options, None for an
    option not given.

    """
    given = [
        option for key, option in _SUPERCELL_OPTIONS.items() if options[key] is not None
    ]
    if len(given) > 1:
        raise click.UsageError(f"Give only one of {', '.join(given)}.")
    try:
        return parse_init({k: v for k, v in options.items() if v is not None})
    except VelphConfigError as e:
        raise click.UsageError(str(e)) from e
