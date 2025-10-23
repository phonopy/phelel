"""Implementation of velph-init."""

from __future__ import annotations

import copy
import dataclasses
import io
import os
import typing
from typing import Literal

import click
import numpy as np
import spglib
import tomli
import tomli_w
from numpy.typing import NDArray
from phono3py.phonon.grid import GridMatrix
from phonopy.interface.calculator import read_crystal_structure
from phonopy.interface.vasp import get_vasp_structure_lines
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import (
    estimate_supercell_matrix,
    get_supercell,
    shape_supercell_matrix,
)
from spglib import SpglibDataset, SpglibMagneticDataset

from phelel.velph.cli.utils import (
    CellChoice,
    DefaultCellChoices,
    DisplacementOptions,
    PrimitiveCellChoice,
    VelphFilePaths,
    VelphInitOptions,
    VelphInitParams,
    generate_standardized_cells,
    get_primitive_cell,
    get_reduced_cell,
    get_symmetry_dataset,
)
from phelel.velph.templates import default_template_dict
from phelel.velph.utils.vasp import CutoffToFFTMesh, VaspIncar
from phelel.version import __version__


def run_init(
    cmd_init_options: VelphInitOptions,
    vfp: VelphFilePaths,
    phelel_dir_name: str = "phelel",
) -> list[str] | None:
    """Run velph-init.

    Main part of init processes are implemented in the function _run_init.

    Preference order of configurations is as follows.

    1. Detailed configurations in [phelel] and [vasp] in velph-template.
    2. Command line options.
    3. [init.options] (alternative of command-line-options) in velph-template.

    Parameters
    ----------
    cmd_options : dict
        Command line options.
    vfp : VelphFilePaths
        Input and output file names required for velph init. Default path to
        scheduler-toml-template file is defined in VelphFilePaths.
    phelel_dir_name : str, optional
        Directory name for [vasp.{phelel_dir_name}]. The default is "phelel",
        which used to be "supercell". This parameter exists for backward
        compatibility.

    Returns
    -------
    list[str, ...]
        velph-toml lines.

    """
    #
    # Parse files.
    #
    input_cell, _ = read_crystal_structure(vfp.cell_filepath, interface_mode="vasp")
    click.echo(f'Read crystal structure file "{vfp.cell_filepath}".')

    return _run_init(
        input_cell,
        cmd_init_options,
        velph_template_fp=vfp.velph_template_filepath,
        template_toml_filepath=vfp.velph_template_filepath,
        phelel_dir_name=phelel_dir_name,
    )


def _run_init(
    input_cell: PhonopyAtoms,
    cmd_init_options: VelphInitOptions,
    velph_template_fp: str | os.PathLike | typing.IO | None = None,
    template_toml_filepath: str | os.PathLike | None = None,
    phelel_dir_name: str = "phelel",
) -> list[str] | None:
    """Run init process and return velph-toml lines.

    Parameters
    ----------
    input_cell : PhonopyAtoms
        Input crystal structure.
    cmd_init_options : VelphInitOptions
        Parameters provided by velph-init command options.
    velph_template_fp : str, os.PathLike, io.IOBase, or None
        velph toml template path. The parameter in str, bytes, or os.PathLike
        represents file name.
    template_toml_filepath : str, os.PathLike
        File name of velph-toml-template.
    phelel_dir_name : str, optional
        Directory name for [vasp.{phelel_dir_name}]. The default is "phelel",
        which used to be "supercell". This parameter exists for backward
        compatibility.

    """
    #
    # Prepare velph configurations: default + template files
    #
    try:
        velph_template_dict = _parse_velph_template(velph_template_fp)
    except tomli.TOMLDecodeError as e:
        click.echo(f'Error in reading "{velph_template_fp}": {e}', err=True)
        return None

    #
    # Collect velph-init command line options.
    #
    template_init_params = _get_template_init_params(
        velph_template_dict, template_toml_filepath
    )
    vip = _collect_init_params(
        cmd_init_options, template_init_params, template_toml_filepath
    )
    if vip is None:
        return None

    #
    # Set magnetic moments
    #
    if vip.magmom is not None:
        magmom_vals = VaspIncar().expand(vip.magmom.strip())
        input_cell.magnetic_moments = magmom_vals

    #
    # Define cells and find crystal symmetry.
    #
    unitcell, primitive, sym_dataset = _get_cells(
        input_cell,
        vip.tolerance,
        vip.symmetrize_cell,
        vip.find_primitive,
        vip.primitive_cell_choice,
    )

    #
    # Parse velph configurations.
    #
    velph_dict = _get_velph_dict(velph_template_dict)

    #
    # Determine cell choices for calculations such as nac, relax, etc.
    #
    cell_choices = _determine_cell_choices(vip, velph_dict)

    #
    # Create velph-toml lines.
    #
    toml_lines = _get_toml_lines(
        velph_dict,
        vip,
        unitcell,
        primitive,
        cell_choices,
        sym_dataset,
        phelel_dir_name=phelel_dir_name,
    )

    return toml_lines


def _get_supercell_matrix(
    vip: VelphInitParams,
    velph_dict: dict,
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
    calc_type: Literal["phelel", "phonopy", "phono3py"],
) -> NDArray | None:
    displacement_options = vip[f"{calc_type}_displacement_options"]
    if displacement_options is None:
        displacement_options = vip.displacement_options
    return _select_supercell_matrix(
        velph_dict.get(calc_type, {}),
        sym_dataset,
        vip.find_primitive,
        max_num_atoms=displacement_options.max_num_atoms,
        supercell_dimension=displacement_options.supercell_dimension,
        supercell_matrix=displacement_options.supercell_matrix,
    )


def _select_supercell_matrix(
    velph_dict_calc_type: dict,
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
    find_primitive: bool,
    max_num_atoms: int | None = None,
    supercell_dimension: tuple[int, int, int] | None = None,
    supercell_matrix: tuple[int, int, int, int, int, int, int, int, int] | None = None,
) -> NDArray | None:
    """Return 3x3 supercell matrix.

    This function is used to determine supercell dimension for velph-init.

    Parameters
    ----------
    velph_dict_calc_type : dict
        velph-toml data of calc_type.
    sym_dataset : SpglibDataset
        Symmetry dataset of the input cell.
    find_primitive : bool
        When True, supercell is constructured
    max_num_atoms : int, optional
        Maximum number of atoms in the supercell. Supercell is constructed to
        preserve the point group of the lattice.
    supercell_dimension : tuple of 3 int, optional
        Supercell dimension. If this is given, it is used as diagonal elements
        of the supercell matrix.
    supercell_matrix : tuple of 9 int, optional
        Supercell matrix. If this is given, it is used as the supercell matrix.

    """
    if max_num_atoms is not None:
        if find_primitive is False:
            _max_num_atoms = max_num_atoms * np.rint(
                1.0 / np.linalg.det(sym_dataset.transformation_matrix)
            ).astype(int)
        else:
            _max_num_atoms = max_num_atoms
        _supercell_matrix = shape_supercell_matrix(
            estimate_supercell_matrix(sym_dataset, max_num_atoms=_max_num_atoms)
        )
    elif supercell_dimension is not None:
        _supercell_matrix = shape_supercell_matrix(supercell_dimension)
    elif supercell_matrix is not None:
        _supercell_matrix = shape_supercell_matrix(supercell_matrix)
    else:
        _supercell_matrix = None
        for key in ("supercell_dimension", "supercell_matrix"):
            if key in velph_dict_calc_type:
                _supercell_matrix = shape_supercell_matrix(velph_dict_calc_type[key])

    return _supercell_matrix


def _determine_cell_choices(vip: VelphInitParams, velph_dict: dict) -> dict:
    """Determine cell choices for calculations such as nac, relax, etc.

    Cell choices are collected from VelphInitParams. When unspecified in
    VelphInitParams, [vasp.*.cell] in velph dict are examined.

    """
    cell_choices = dataclasses.asdict(DefaultCellChoices())
    for key in ("nac", "relax"):
        value = vip[f"cell_for_{key}"]
        if value is CellChoice.UNSPECIFIED:
            if (
                "vasp" in velph_dict
                and key in velph_dict["vasp"]
                and "cell" in velph_dict["vasp"][key]
            ):
                for _cell_choice in CellChoice:
                    if _cell_choice is CellChoice.UNSPECIFIED:
                        continue
                    if _cell_choice.value in velph_dict["vasp"][key]["cell"]:
                        cell_choices[key] = _cell_choice
        else:
            cell_choices[key] = value
        assert cell_choices[key] in CellChoice
        assert cell_choices[key] is not CellChoice.UNSPECIFIED
    return cell_choices


def _get_template_init_params(
    velph_template_dict: dict | None, template_toml_filepath: str | os.PathLike | None
) -> VelphInitOptions:
    """Collect init params in [init.options] in velph-toml-template file."""
    if not velph_template_dict:
        return VelphInitOptions()

    try:
        vip_keys = velph_template_dict["init"]["options"].keys()
    except KeyError:
        return VelphInitOptions()

    template_init_params = {}
    for _dataclass in (VelphInitParams, DisplacementOptions):
        for field in dataclasses.fields(_dataclass):
            key = field.name
            if key not in vip_keys:
                continue
            template_init_params[key] = velph_template_dict["init"]["options"][key]

    # Show parameters specified in velph-toml-template file.
    if template_init_params:
        click.echo("Following init-options were found", nl=False)
        if template_toml_filepath:
            click.echo(f" in {template_toml_filepath}", nl=False)
        click.echo(":")
        click.echo(
            "\n".join(
                [f"  {key} = {value}" for key, value in template_init_params.items()]
            )
        )

    return VelphInitOptions(**template_init_params)


def _collect_init_params(
    cmd_init_options: VelphInitOptions,
    template_init_params: VelphInitOptions,
    template_toml_filepath: str | os.PathLike | None,
) -> VelphInitParams | None:
    """Merge init params defined different places.

    Init parameters were collected in the following order. For the same
    parameters, the latters override the formers.
    1. Defalut VelphInitParams
    2. template_dict["init"]["options"]
    3. Command line options

    Returns
    -------
    VelphInitParams
        Init parameters.

    """
    displacement_options_keys = [
        field.name for field in dataclasses.fields(DisplacementOptions)
    ]
    vip_dict = {}
    displacement_options = {}

    # Set parameters specified in velph-toml-template file.
    for key, value in template_init_params.items():
        if value is None:
            continue
        elif key in displacement_options_keys:
            displacement_options.update({key: value})
        elif key in ("cell_for_nac", "cell_for_relax"):
            for cell_choice in CellChoice:
                if value.lower() == cell_choice.value:
                    vip_dict[key] = cell_choice
        elif key == "primitive_cell_choice":
            for primitive_cell_choice in PrimitiveCellChoice:
                if value.lower() == primitive_cell_choice.value:
                    vip_dict[key] = primitive_cell_choice
        else:
            vip_dict[key] = value

    # Collect parameters specified by command-line options.
    # Filling by None for all keys in VelphInitParams is for test mimicing
    # the behaviour of command-line-options that would already fill so.
    cmd_displacement_options = {}
    for key in displacement_options_keys:
        if key in cmd_init_options:
            value = cmd_init_options[key]
            if value is not None:
                cmd_displacement_options.update({key: value})

    if cmd_displacement_options:
        num_active_cmd_params = 1
        displacement_options.update(cmd_displacement_options)
    else:
        num_active_cmd_params = 0

    cmd_params: dict = {}
    for field in dataclasses.fields(VelphInitParams):
        key = field.name
        if key in cmd_init_options:
            value = cmd_init_options[key]
            if value is not None:
                num_active_cmd_params += 1
                cmd_params.update({key: value})

    # Show parameters specified by command-line options.
    if num_active_cmd_params > 0:
        click.echo("The following command-line options were given:")
    for key, value in cmd_displacement_options.items():
        click.echo(f"  {key} = {value}")
    for key, value in cmd_params.items():
        if value is None:
            continue
        if key in ("cell_for_nac", "cell_for_relax"):
            for cell_choice in CellChoice:
                if cell_choice == CellChoice.UNSPECIFIED:
                    continue
                if value.lower() == cell_choice.value:
                    cmd_params[key] = cell_choice
                    click.echo(f"  {key} = {value.lower()}")
        elif key == "primitive_cell_choice":
            for primitive_cell_choice in PrimitiveCellChoice:
                if value.lower() == primitive_cell_choice.value:
                    cmd_params[key] = primitive_cell_choice
        else:
            click.echo(f"  {key} = {value}")

    shared_params = [
        key_c == key_t for key_t in template_init_params for key_c in cmd_params
    ]
    if template_toml_filepath:
        if sum(shared_params) == 1:
            click.echo(
                "The command option was preferred to [init.options] in "
                f'"{template_toml_filepath}".'
            )
        if sum(shared_params) > 1:
            click.echo(
                "The command options were preferred to [init.options] in "
                f'"{template_toml_filepath}".'
            )

    # Set parameters specified by command-line options. vip_dict is already
    # filled by [init.options]. But command-line-options have higher preference.
    for key, value in cmd_params.items():
        if value is not None:
            vip_dict[key] = value

    # DisplacementOptions is treated specially.
    if displacement_options:
        vip_dict["displacement_options"] = DisplacementOptions(**displacement_options)

    # Treatment of correlation among parameters
    if "max_num_atoms" in displacement_options:
        if "symmetrize_cell" not in vip_dict or vip_dict["symmetrize_cell"] is False:
            msg = """
------------------------------- ERROR -------------------------------
"max_num_atoms" requires "symmetrize_cell=true" (--symmetrize-cell).
For "symmetrize_cell=false", use "supercell_dimension" (--dim) or
"supercell_matrix" (--supercell-matrix).
---------------------------------------------------------------------"""
            click.echo(msg, err=True)
            return None

    vip = VelphInitParams(**vip_dict)

    return vip


def _get_cells(
    input_cell: PhonopyAtoms,
    tolerance: float,
    symmetrize_cell: bool,
    find_primitive: bool,
    primitive_cell_choice: PrimitiveCellChoice,
) -> tuple[PhonopyAtoms, PhonopyAtoms, SpglibDataset | SpglibMagneticDataset]:
    """Return unit cell, primitive cell, and symmetry dataset.

    This function is complicated due to complicated requests.

    The unit cell and primitive cell returned can change depending on
    `symmetrize_cell`, `find_primitive` and `primitive_cell_choice`. When
    `primitive_cell_choice==PrimitiveCellChoice.REDUCED`, the primitive cell
    is finally redued.

    * `symmetrize_cell=True` and `find_primitive=True`.

    The unit cell and primitive cell are those standardized by spglib.

    * `symmetrize_cell=False` and `find_primitive=True`.

    The unit cell is the input cell. If the input cell is a primitive cell, it
    is used as the primitive cell. Otherwise, the primitive cell standardized by
    spglib is used as the primitive cell.

    * `symmetrize_cell=False` and `find_primitive=False`.

    The unit cell and primitive cell are the same as the input cell.

    Parameters
    ----------
    cell_filename : Path
        Input cell file name.
    tolerance : float
        Symmetry search tolerance.
    symmetrize_cell : bool
        If True, a standardize unit cell is generated.
    find_primitive : bool
        If False, the standardized unit cell (``symmetrize_cell=True``) or the
        input cell (``symmetrize_cell=False``) is used as the primitive cell. If
        True, primitive cell is made from the standardized cell
        (``symmetrize_cell=True``) or the input cell
        (``symmetrize_cell=False``).
    primitive_cell_choice : PrimitiveCellChoice
        When `primitive_cell_choice==PrimitiveCellChoice.REDUCED` The primitive
        cell is reduced by spglib.

    """
    sym_dataset = get_symmetry_dataset(input_cell, tolerance=tolerance)
    if isinstance(sym_dataset, SpglibDataset):
        click.echo(f"Space-group: {sym_dataset.international}")
    else:
        spg_type = spglib.get_spacegroup_type(sym_dataset.hall_number)
        assert spg_type is not None
        click.echo(f"Magnetic-space-group type-{sym_dataset.msg_type}")
        click.echo(f"  Uni-number: {sym_dataset.uni_number}")
        click.echo(f"  Reference space-group-type: {spg_type.international_short}")

    if symmetrize_cell:
        unitcell, _primitive, tmat = generate_standardized_cells(
            sym_dataset, tolerance=tolerance
        )
        if find_primitive:
            primitive = _primitive
            if len(_primitive) != len(unitcell):
                click.echo(
                    "Transformation matrix from conventional unit cell to "
                    "prmitive cell is"
                )
                for v in tmat:
                    click.echo(f"  [{v[0]:6.3f} {v[1]:6.3f} {v[2]:6.3f}]")
        else:
            primitive = unitcell
    else:
        unitcell = input_cell
        _primitive, tmat = get_primitive_cell(
            unitcell, sym_dataset, tolerance=tolerance
        )
        if find_primitive:
            if len(_primitive) == len(input_cell):
                primitive = input_cell
                click.echo(
                    "Input cell was already a primitive cell, and it is used as "
                    "the primitive cell."
                )
            else:
                primitive = _primitive
                click.echo("Found a primitive cell whose transformation matrix is")
                for v in tmat:
                    click.echo(f"  [{v[0]:6.3f} {v[1]:6.3f} {v[2]:6.3f}]")
        else:
            primitive = unitcell
            if len(_primitive) == len(unitcell):
                pass
            else:
                click.echo(
                    "Input cell is not a primitive cell from the symmetry point of "
                    "view. "
                )
                click.echo(
                    "But velph will consider the input cell as the primitive cell. "
                )
                click.echo(
                    "For reference, below is a potential primitive cell found in "
                    "the input cell."
                )
                click.echo("-" * 70)
                click.echo("\n".join(get_vasp_structure_lines(_primitive)).strip())
                click.echo("-" * 70)

    click.echo("Supercell is generated with respect to the cell below.")
    click.echo("-" * 80)
    click.echo(str(unitcell))
    click.echo("-" * 80)

    if primitive_cell_choice is PrimitiveCellChoice.REDUCED:
        primitive = get_reduced_cell(primitive, tolerance=tolerance)

    return unitcell, primitive, sym_dataset


def _parse_velph_template(
    velph_template_fp: str | os.PathLike | typing.IO | None,
) -> dict | None:
    """Read velph-toml template file.

    The type PathLike is used to represent file name, while toml_str does not
    serve this purpose. If there is a need to pass toml_str, it can be achieved
    by using io.BytesIO(toml_str.encode('utf-8')).

    """
    if velph_template_fp is None:
        return None

    if isinstance(velph_template_fp, io.BytesIO):
        return tomli.load(velph_template_fp)
    else:
        assert isinstance(velph_template_fp, (str, os.PathLike))
        with open(velph_template_fp, "rb") as f:
            template_dict = tomli.load(f)

    click.echo(f'Read velph template file "{velph_template_fp}".')
    return template_dict


def _get_velph_dict(
    template_dict: dict | None,
) -> dict:
    """Return velph_dict.

    velph_dict is made by default_template_dict, template_dict and
    schedular_template_dict.

    This function implements very human-like actions.

    * [vasp.{key}] is simply replaced by template_dict["vasp"][key] unless
      key=="incar".
    * [vasp.incar] is treated specially by merging but not replacing.

    """
    velph_dict = copy.deepcopy(default_template_dict)
    if template_dict is not None:
        _update_velph_dict_by_template_dict(velph_dict, template_dict)

    return velph_dict


def _update_velph_dict_by_template_dict(velph_dict: dict, template_dict: dict):
    """Update velph_dict by template dict.

    [phelel], [vasp], ...

    """
    for key in template_dict:
        if key in velph_dict:
            if key == "vasp":
                _update_vasp_dict_by_template_dict(velph_dict, template_dict)
            else:
                velph_dict[key] = template_dict[key]


def _update_vasp_dict_by_template_dict(velph_dict: dict, template_dict: dict):
    """Update vasp_dict by template dict.

    [vasp.incar], [vasp.calc_key.incar], [vasp.calc_key.kpoints], ...

    """
    vasp_incar = _merge_vasp_incar_section(velph_dict, template_dict)
    for calc_key in template_dict["vasp"]:
        if calc_key == "incar":
            continue
        if calc_key in velph_dict["vasp"]:
            _update_vasp_calc_types(
                velph_dict["vasp"][calc_key],
                template_dict["vasp"][calc_key],
            )
        else:
            velph_dict["vasp"][calc_key] = template_dict["vasp"][calc_key]
    velph_dict["vasp"]["incar"] = vasp_incar


def _merge_vasp_incar_section(velph_dict: dict, template_dict: dict) -> dict:
    """Merge [vasp.incar] sections in default and template dicts."""
    vasp_incar = velph_dict["vasp"]["incar"]
    if "incar" in template_dict["vasp"]:
        for key in template_dict["vasp"]["incar"]:
            vasp_incar[key] = template_dict["vasp"]["incar"][key]

    vasp_incar_str = "[vasp.incar] (basic INCAR settings)"
    special_tags = [f"  {key} = {val}" for key, val in vasp_incar.items()]
    click.echo("\n".join([f"{vasp_incar_str}"] + special_tags))

    return vasp_incar


def _update_vasp_calc_types(vasp_calc_dict: dict, template_calc_dict: dict):
    """Update [vasp.calc_type].

    Treatment of [vasp.calc_type.*].

    """
    for key, val in template_calc_dict.items():
        if key in vasp_calc_dict and isinstance(val, dict):
            if isinstance(vasp_calc_dict[key], dict):
                vasp_calc_dict[key].update(val)
                continue
            else:
                raise RuntimeError(
                    "velph template and default template are inconsistent."
                )
        vasp_calc_dict[key] = template_calc_dict[key]


def _get_toml_lines(
    velph_dict: dict,
    vip: VelphInitParams,
    unitcell: PhonopyAtoms,
    primitive: PhonopyAtoms,
    cell_choices: dict,
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
    phelel_dir_name: str = "phelel",
) -> list[str] | None:
    """Return velph-toml lines."""
    assert vip.displacement_options is not None
    supercell_matrices = {}
    for calc_type in ("phelel", "phonopy", "phono3py"):
        supercell_matrices[calc_type] = _get_supercell_matrix(
            vip,
            velph_dict,
            sym_dataset,
            calc_type,
        )
        click.echo(f"[{calc_type}]")
        if supercell_matrices[calc_type] is not None:
            _show_supercell_dimension(supercell_matrices[calc_type])

    (
        kpoints_dict,
        kpoints_dense_dict,
        qpoints_dict,
        kpoints_opt_dict,
    ) = _get_kpoints_dict(
        vip.kspacing,
        vip.kspacing_dense,
        vip.use_grg,
        vip.tolerance,
        unitcell,
        primitive,
        sym_dataset,
        supercell_matrices,
        cell_choices["nac"],
        cell_choices["relax"],
        phelel_dir_name=phelel_dir_name,
    )

    if "vasp" in velph_dict:
        _update_kpoints_by_vasp_dict(
            kpoints_dict,
            kpoints_dense_dict,
            qpoints_dict,
            kpoints_opt_dict,
            velph_dict["vasp"],
        )
        _show_kpoints_lines(
            kpoints_dict,
            kpoints_dense_dict,
            velph_dict["vasp"],
            vip.kspacing,
            vip.kspacing_dense,
        )

    #
    # velph.toml
    #
    lines = []

    # [phelel]
    if "phelel" in velph_dict:
        lines += _get_phelel_lines(
            velph_dict,
            supercell_matrices["phelel"],
            primitive,
            vip.displacement_options.amplitude,
            vip.displacement_options.diagonal,
            vip.displacement_options.plusminus,
            vip.phelel_nosym,
        )

    # [phonopy], [phono3py]
    for calc_type in ("phonopy", "phono3py"):
        displacement_options = vip[f"{calc_type}_displacement_options"]
        if displacement_options is None:
            displacement_options = vip.displacement_options
        if supercell_matrices[calc_type] is not None:
            lines += [f"[{calc_type}]"]
            lines += _get_supercell_matrix_lines(supercell_matrices[calc_type])
            lines += _get_displacement_settings_lines(
                velph_dict,
                calc_type,
                displacement_options.amplitude,
                displacement_options.diagonal,
                displacement_options.plusminus,
            )
            lines.append("")

    # [vasp.*]
    if "vasp" in velph_dict:
        lines += _get_vasp_lines(
            velph_dict["vasp"],
            kpoints_dict,
            kpoints_dense_dict,
            kpoints_opt_dict,
            qpoints_dict,
            cell_choices["nac"],
            cell_choices["relax"],
            phelel_dir_name=phelel_dir_name,
        )

    # [scheduler]
    if "scheduler" in velph_dict:
        scheduler_dict = {
            key: val
            for key, val in velph_dict["scheduler"].items()
            if key != "scheduler_template"
        }
        scheduler_template_str = velph_dict["scheduler"].get("scheduler_template")
        lines.append("[scheduler]")
        lines.append(tomli_w.dumps(scheduler_dict).strip())
        if scheduler_template_str is not None:
            lines.append('scheduler_template = """\\')
            lines.append(f'{scheduler_template_str}"""')
        lines.append("")

    # [symmetry]
    if sym_dataset is not None:
        lines.append("[symmetry]")
        if isinstance(sym_dataset, SpglibDataset):
            spg_type = sym_dataset.international
            lines.append(f'spacegroup_type = "{spg_type}"')
        else:
            lines.append(f"uni_number = {sym_dataset.uni_number}")
        lines.append(f"tolerance = {vip.tolerance}")
        if len(unitcell) != len(primitive):
            pmat = (primitive.cell @ np.linalg.inv(unitcell.cell)).T
            lines.append("primitive_matrix = [")
            for v in pmat:
                lines.append(f"  [ {v[0]:18.15f}, {v[1]:18.15f}, {v[2]:18.15f} ],")
            lines.append("]")
            lines.append("")

    # [unitcell]
    lines += _get_cell_toml_lines(unitcell, cell_name="unitcell")

    # [primitive_cell]
    if primitive is not None:
        lines += _get_cell_toml_lines(primitive, cell_name="primitive_cell")

    return lines


def _get_fft_mesh(velph_dict: dict, primitive: PhonopyAtoms) -> NDArray | None:
    """FFT mesh is computed from encut and prec in INCAR dict.

    Two possiblity of encut sourse, [vasp.selfenergy.incar] and [vasp.incar].

    """
    cutoff_eV = None
    prec = None

    incar_dict_selfenergy = _get_incar_dict_with_encut(
        velph_dict, ["vasp", "selfenergy", "incar"]
    )
    incar_dict = _get_incar_dict_with_encut(velph_dict, ["vasp", "incar"])
    incar_dict.update(incar_dict_selfenergy)
    if "encut" in incar_dict:
        cutoff_eV = incar_dict["encut"]
        if "prec" in incar_dict:
            prec = incar_dict["prec"]
        if cutoff_eV is not None:
            fft_mesh = CutoffToFFTMesh.get_FFTMesh(
                cutoff_eV, primitive.cell, incar_prec=prec
            )
            return fft_mesh
    return None


def _get_incar_dict_with_encut(velph_dict: dict, nested_keys: list[str]) -> dict:
    """Return INCAR dict containing encut and prec keys."""
    incar_dict = velph_dict
    for key in nested_keys:
        try:
            incar_dict = incar_dict[key]
        except KeyError:
            return {}

    return {
        key.lower(): val
        for key, val in incar_dict.items()
        if key.lower() in ("encut", "prec")
    }


def _get_kpoints_dict(
    vip_kspacing: float,
    vip_kspacing_dense: float,
    vip_use_grg: bool,
    vip_tolerance: float,
    unitcell: PhonopyAtoms,
    primitive: PhonopyAtoms,
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
    supercell_matrices: dict[Literal["phelel", "phonopy", "phono3py"], NDArray],
    cell_for_nac: CellChoice,
    cell_for_relax: CellChoice,
    phelel_dir_name: str = "phelel",
) -> tuple[dict, dict, dict, dict]:
    """Return kpoints dicts."""
    use_grg_unitcell = vip_use_grg and (len(unitcell) == len(primitive))
    sym_dataset_prim = get_symmetry_dataset(primitive, tolerance=vip_tolerance)

    # Grid matrix for unitcell
    gm = _get_grid_matrix(vip_kspacing, unitcell, sym_dataset, use_grg_unitcell)

    # Grid matrix for primitive cell
    gm_prim = _get_grid_matrix(vip_kspacing, primitive, sym_dataset_prim, vip_use_grg)

    # Grid matrix for supercell
    supercell_grid_matrices = {}
    for calc_type in ("phelel", "phonopy", "phono3py"):
        if supercell_matrices[calc_type] is not None:
            _supercell = get_supercell(unitcell, supercell_matrices[calc_type])
            _sym_dataset = get_symmetry_dataset(_supercell, tolerance=vip_tolerance)
            supercell_grid_matrices[calc_type] = _get_grid_matrix(
                vip_kspacing,
                _supercell,
                _sym_dataset,
                use_grg=False,
            )
        else:
            supercell_grid_matrices[calc_type] = None

    # Dense grid matrix for primitive cell
    gm_dense_prim = _get_grid_matrix(
        vip_kspacing_dense, primitive, sym_dataset_prim, vip_use_grg
    )

    # Build return values
    kpoints_dict = _get_kpoints_by_kspacing(
        gm,
        gm_prim,
        supercell_grid_matrices,
        cell_for_nac,
        cell_for_relax,
        phelel_dir_name=phelel_dir_name,
    )
    kpoints_dense_dict = _get_kpoints_by_kspacing_dense(
        gm_dense_prim, with_phelel=phelel_dir_name in kpoints_dict
    )
    qpoints_dict: dict = {}
    kpoints_opt_dict: dict = {}

    return kpoints_dict, kpoints_dense_dict, qpoints_dict, kpoints_opt_dict


def _get_grid_matrix(
    kspacing: float,
    primitive: PhonopyAtoms,
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
    use_grg: bool,
) -> GridMatrix:
    try:
        gm = GridMatrix(
            2 * np.pi / kspacing,
            lattice=primitive.cell,
            symmetry_dataset=sym_dataset,
            use_grg=use_grg,
        )
    except RuntimeError as e:
        if "Grid symmetry is broken." in str(e):
            click.echo(
                "Warning: Grid symmetry is broken. "
                "Switching to use_grg=True for the primitive cell."
            )
            gm = GridMatrix(
                2 * np.pi / kspacing,
                lattice=primitive.cell,
                symmetry_dataset=sym_dataset,
                use_grg=True,
            )
        else:
            raise e

    return gm


def _update_kpoints_by_vasp_dict(
    kpoints_dict: dict,
    kpoints_dense_dict: dict,
    qpoints_dict: dict,
    kpoints_opt_dict: dict,
    vasp_dict: dict,
) -> None:
    """Overwrite kpoints_(dense_)dict if kpoints are defined in vasp_dict.

    "phelel.phonon" are "phono3py.phonon" should be understood as
    vasp_dict["phelel"]["phonon"] and vasp_dict["phono3py"]["phonon"].

    """
    for key, calc_type_dict in vasp_dict.items():
        if "kpoints" in calc_type_dict:
            kpoints_dict[key] = calc_type_dict["kpoints"]
        if "kpoints_dense" in calc_type_dict:
            kpoints_dense_dict[key] = calc_type_dict["kpoints_dense"]
        if "qpoints" in calc_type_dict:
            qpoints_dict[key] = calc_type_dict["qpoints"]
        if "kpoints_opt" in calc_type_dict:
            kpoints_opt_dict[key] = calc_type_dict["kpoints_opt"]


def _show_kpoints_lines(
    kpoints_dict: dict,
    kpoints_dense_dict: dict,
    vasp_dict: dict,
    vip_kspacing: float,
    vip_kspacing_dense: float,
):
    k_mesh_lines = [f"[vasp.*.kpoints.mesh] (*kspacing={vip_kspacing})"]
    for key in vasp_dict:
        if key == "ph_bands":
            continue
        if key in kpoints_dict:
            mesh = kpoints_dict[key].get("mesh")
            if mesh is not None:
                if "D_diag" in kpoints_dict[key]:
                    D_diag = kpoints_dict[key]["D_diag"]
                    line = f"  {key}: {D_diag}"
                else:
                    line = f"  {key}: {np.array(mesh)}"
                if "kpoints" not in vasp_dict[key]:
                    line += "*"
                k_mesh_lines.append(line)
                if "D_diag" in kpoints_dict[key]:
                    k_mesh_lines += [f"     {v}" for v in np.array(mesh)]

    k_mesh_lines.append(
        f"[vasp.*.kpoints_dense.mesh] (*kspacing_dense={vip_kspacing_dense})"
    )
    for key in vasp_dict:
        if key in kpoints_dense_dict:
            mesh = kpoints_dense_dict[key].get("mesh")
            if mesh is not None:
                if "D_diag" in kpoints_dense_dict[key]:
                    D_diag = kpoints_dense_dict[key]["D_diag"]
                    line = f"  {key}: {np.array(D_diag)}"
                else:
                    line = f"  {key}: {np.array(mesh)}"
                if "kpoints_dense" not in vasp_dict[key]:
                    line += "*"
                k_mesh_lines.append(line)
                if "D_diag" in kpoints_dense_dict[key]:
                    k_mesh_lines += [f"     {v}" for v in np.array(mesh)]
    if k_mesh_lines:
        click.echo("\n".join(k_mesh_lines))


def _get_kpoints_by_kspacing(
    gm: GridMatrix,
    gm_prim: GridMatrix,
    supercell_grid_matrices: dict[Literal["phelel", "phonopy", "phono3py"], GridMatrix],
    cell_for_nac: CellChoice,
    cell_for_relax: CellChoice,
    phelel_dir_name: str = "phelel",
) -> dict:
    """Return kpoints dict.

    Parameters
    ----------
    gm : GridMatrix
        Grid matrix of unit cell.
    gm_prim : GridMatrix
        Grid matrix of primitive cell. The primitive cell can be a reduced cell.
    supercell_grid_matrices : dict[Literal["phelel", "phonopy", "phono3py"], GridMatrix]
        Grid matrices of phelel, phonopy, and phono3py supercells.
    cell_for_nac : CellChoice
        Cell choice for NAC calculation among unit cell or primitive cell. The
        primitive cell can be a reduced cell.
    cell_for_relax : CellChoice
        Cell choice for relax calculation among unit cell or primitive cell. The
        primitive cell can be a reduced cell.

    Returns
    -------
    dict
        kpoints information for each calculation type is stored. The keys must
        be calc_type names such as "phelel", "phelel.phonon", "selfenergy", etc.

    """
    kpoints_of_supercells = {}
    for calc_type in ("phelel", "phonopy", "phono3py"):
        gm_super = supercell_grid_matrices[calc_type]
        if gm_super is not None:
            if gm_super.grid_matrix is None:
                kpoints_of_supercells[calc_type] = {"mesh": gm_super.D_diag}
            else:
                kpoints_of_supercells[calc_type] = {
                    "mesh": gm_super.grid_matrix,
                    "D_diag": gm_super.D_diag,
                }
        else:
            kpoints_of_supercells[calc_type] = None

    if gm_prim.grid_matrix is None:
        selfenergy_kpoints = {"mesh": gm_prim.D_diag}
        el_bands_kpoints = {"mesh": gm_prim.D_diag}
    else:
        selfenergy_kpoints = {"mesh": gm_prim.grid_matrix, "D_diag": gm_prim.D_diag}
        el_bands_kpoints = {"mesh": gm_prim.grid_matrix, "D_diag": gm_prim.D_diag}
    ph_bands_kpoints = {"mesh": [1, 1, 1]}

    # nac
    if cell_for_nac is CellChoice.UNITCELL:
        gm_nac = gm
    elif cell_for_nac is CellChoice.PRIMITIVE:
        gm_nac = gm_prim
    else:
        raise RuntimeError("This is something that sholud not happen.")
    if gm_nac.grid_matrix is None:
        nac_kpoints = {"mesh": gm_nac.D_diag}
    else:
        nac_kpoints = {"mesh": gm_nac.grid_matrix, "D_diag": gm_nac.D_diag}

    # relax
    if cell_for_relax is CellChoice.UNITCELL:
        gm_relax = gm
    elif cell_for_relax is CellChoice.PRIMITIVE:
        gm_relax = gm_prim
    else:
        raise RuntimeError("This is something that sholud not happen.")
    if gm_relax.grid_matrix is None:
        relax_kpoints = {"mesh": gm_relax.D_diag}
    else:
        relax_kpoints = {"mesh": gm_relax.grid_matrix, "D_diag": gm_relax.D_diag}

    # keys are calc_types.
    kpoints_dict = {
        phelel_dir_name: kpoints_of_supercells["phelel"],
        "phonopy": kpoints_of_supercells["phonopy"],
        "phono3py": kpoints_of_supercells["phono3py"],
        "relax": relax_kpoints,
        "nac": nac_kpoints,
        "el_bands.dos": el_bands_kpoints,
        "el_bands.bands": el_bands_kpoints,
        "ph_bands": ph_bands_kpoints,
    }
    if gm_super:
        kpoints_dict.update(
            {
                "selfenergy": selfenergy_kpoints,
                "transport": selfenergy_kpoints,
            }
        )

    return_kpoints_dict = {}
    for key, val in kpoints_dict.items():
        if val is not None:
            return_kpoints_dict[key] = val

    return return_kpoints_dict


def _get_kpoints_by_kspacing_dense(
    gm_dense_prim: GridMatrix, with_phelel: bool = True
) -> dict:
    """Return kpoints dict of dense grid.

    Returns
    -------
    dict
        kpoints information for each calculation type is stored. The keys must
        be calc_type names such as "selfenergy", etc.

    """
    if gm_dense_prim.grid_matrix is None:
        selfenergy_kpoints_dense = {"mesh": gm_dense_prim.D_diag}
        el_bands_kpoints_dense = {"mesh": gm_dense_prim.D_diag}
    else:
        selfenergy_kpoints_dense = {
            "mesh": gm_dense_prim.grid_matrix,
            "D_diag": gm_dense_prim.D_diag,
        }
        el_bands_kpoints_dense = {
            "mesh": gm_dense_prim.grid_matrix,
            "D_diag": gm_dense_prim.D_diag,
        }

    if with_phelel:
        return {
            "selfenergy": selfenergy_kpoints_dense,
            "transport": selfenergy_kpoints_dense,
            "el_bands.dos": el_bands_kpoints_dense,
        }
    else:
        return {
            "el_bands.dos": el_bands_kpoints_dense,
        }


def _get_vasp_lines(
    vasp_dict: dict,
    kpoints_dict: dict,
    kpoints_dense_dict: dict,
    kpoints_opt_dict: dict,
    qpoints_dict: dict,
    cell_for_nac: CellChoice,
    cell_for_relax: CellChoice,
    phelel_dir_name: str = "phelel",
) -> list:
    incar_commons = _get_incar_commons(vasp_dict)

    lines = []

    for calc_type in (phelel_dir_name, "phonopy", "phono3py"):
        if calc_type in vasp_dict and calc_type in kpoints_dict:
            _vasp_dict = vasp_dict[calc_type]
            _add_incar_lines(lines, _vasp_dict, incar_commons, calc_type)
            lines.append(f"[vasp.{calc_type}.kpoints]")
            _add_kpoints_lines(lines, kpoints_dict[calc_type])
            _add_calc_type_scheduler_lines(lines, _vasp_dict, calc_type)
            lines.append("")

    for calc_type in ("selfenergy", "transport"):
        if (
            calc_type in vasp_dict
            and calc_type in kpoints_dict
            and calc_type in kpoints_dense_dict
        ):
            # primitive cell
            _vasp_dict = vasp_dict[calc_type]
            _add_incar_lines(lines, _vasp_dict, incar_commons, calc_type)
            lines.append(f"[vasp.{calc_type}.kpoints]")
            _add_kpoints_lines(lines, kpoints_dict[calc_type])
            lines.append(f"[vasp.{calc_type}.kpoints_dense]")
            _add_kpoints_lines(lines, kpoints_dense_dict[calc_type])
            _add_calc_type_scheduler_lines(lines, _vasp_dict, calc_type)
            lines.append("")

    if "relax" in vasp_dict:
        lines.append("[vasp.relax]")
        assert cell_for_relax in CellChoice
        assert cell_for_relax is not CellChoice.UNSPECIFIED
        _vasp_dict = vasp_dict["relax"]
        lines.append(f'cell = "{cell_for_relax.value}"')
        _add_incar_lines(lines, _vasp_dict, incar_commons, "relax")
        lines.append("[vasp.relax.kpoints]")
        _add_kpoints_lines(lines, kpoints_dict["relax"])
        _add_calc_type_scheduler_lines(lines, _vasp_dict, "relax")
        lines.append("")

    if "nac" in vasp_dict:
        lines.append("[vasp.nac]")
        assert cell_for_nac in CellChoice
        assert cell_for_nac is not CellChoice.UNSPECIFIED
        _vasp_dict = vasp_dict["nac"]
        lines.append(f'cell = "{cell_for_nac.value}"')
        _add_incar_lines(lines, _vasp_dict, incar_commons, "nac")
        lines.append("[vasp.nac.kpoints]")
        _add_kpoints_lines(lines, kpoints_dict["nac"])
        _add_calc_type_scheduler_lines(lines, _vasp_dict, "nac")
        lines.append("")

    for calc_subtype in ("bands", "dos"):
        if f"el_bands.{calc_subtype}" not in vasp_dict:
            continue

        # primitive cell
        _vasp_dict = vasp_dict[f"el_bands.{calc_subtype}"]
        _add_incar_lines(lines, _vasp_dict, incar_commons, f"el_bands.{calc_subtype}")
        lines.append(f"[vasp.el_bands.{calc_subtype}.kpoints]")
        _add_kpoints_lines(lines, kpoints_dict[f"el_bands.{calc_subtype}"])
        if calc_subtype == "bands":
            el_bands_kpoints_opt = kpoints_opt_dict.get("el_bands.bands")
            if el_bands_kpoints_opt and "line" in el_bands_kpoints_opt:
                lines.append("[vasp.el_bands.bands.kpoints_opt]")
                _add_kpoints_lines_bands(lines, el_bands_kpoints_opt)
        if calc_subtype == "dos":
            el_bands_kpoints_dense = kpoints_dense_dict.get("el_bands.dos")
            if el_bands_kpoints_dense and "mesh" in el_bands_kpoints_dense:
                lines.append("[vasp.el_bands.dos.kpoints_dense]")
                _add_kpoints_lines(lines, el_bands_kpoints_dense)
        _add_calc_type_scheduler_lines(lines, _vasp_dict, f"el_bands.{calc_subtype}")
        lines.append("")

    if "ph_bands" in vasp_dict:
        # primitive cell
        ph_bands_qpoints = qpoints_dict.get("ph_bands")
        _vasp_dict = vasp_dict["ph_bands"]
        _add_incar_lines(lines, _vasp_dict, incar_commons, "ph_bands")
        lines.append("[vasp.ph_bands.kpoints]")
        _add_kpoints_lines(lines, kpoints_dict["ph_bands"])
        if ph_bands_qpoints:
            lines.append("[vasp.ph_bands.qpoints]")
            _add_kpoints_lines_bands(lines, ph_bands_qpoints)
        _add_calc_type_scheduler_lines(lines, _vasp_dict, "ph_bands")
        lines.append("")

    return lines


def _get_incar_commons(vasp_dict: dict) -> dict:
    """Extract common INCAR parameters.

    Parameters written in ``vasp_dict['incar']`` are considered as common
    INCAR parameters.

    Default common INCAR parameters are hard coded in ``default_template_dict``.

    """
    assert "incar" in vasp_dict
    return vasp_dict["incar"]


def _merge_incar_commons(incar: dict, incar_commons: dict):
    """Merge INCAR parameters in template and common INCAR parameters."""
    incar_copy = {key.lower(): copy.deepcopy(value) for key, value in incar.items()}
    for key in [key.lower() for key in incar_commons]:
        if key not in incar_copy:
            incar_copy[key] = incar_commons[key]
    return {
        key: value
        for key, value in incar_copy.items()
        if not isinstance(value, dict) or value
    }


def _get_phelel_lines(
    velph_dict: dict,
    supercell_matrix: NDArray | None,
    primitive: PhonopyAtoms,
    amplitude: float,
    diagonal: bool,
    plusminus: Literal["auto"] | bool,
    phelel_nosym: bool,
) -> list:
    lines = []
    lines.append("[phelel]")
    lines.append(f'version = "{__version__}"')

    if supercell_matrix is not None:
        lines += _get_supercell_matrix_lines(supercell_matrix)
        lines += _get_displacement_settings_lines(
            velph_dict, "phelel", amplitude, diagonal, plusminus
        )

        if phelel_nosym:
            lines.append("nosym = true")

        fft_mesh = _get_fft_mesh(velph_dict, primitive)
        try:
            if fft_mesh is None and "fft_mesh" in velph_dict["phelel"]:
                fft_mesh = velph_dict["phelel"]["fft_mesh"]
            if fft_mesh is not None:
                lines.append("fft_mesh = [{:d}, {:d}, {:d}]".format(*fft_mesh))
        except KeyError:
            pass

    lines.append("")
    return lines


def _add_incar_lines(lines: list, vasp_dict: dict, incar_commons: dict, calc_type: str):
    if "incar" in vasp_dict:
        lines.append(f"[vasp.{calc_type}.incar]")
        lines.append(
            tomli_w.dumps(
                _merge_incar_commons(vasp_dict["incar"], incar_commons)
            ).strip()
        )


def _add_kpoints_lines_bands(lines: list, kpt_dict: dict) -> None:
    lines.append("line = {:d}".format(kpt_dict["line"]))
    if "path" in kpt_dict:
        lines.append("path = " + str(kpt_dict["path"]))
        points = kpt_dict["path"]
        for i in np.unique(points):
            lines.append(
                str(i) + " = " + "[{:f}, {:f}, {:f}]".format(*kpt_dict[str(i)])
            )


def _add_kpoints_lines(lines: list, kpt_dict: dict) -> None:
    if len(np.ravel(kpt_dict["mesh"])) == 3:
        lines.append("mesh = [{:d}, {:d}, {:d}]".format(*kpt_dict["mesh"]))
    elif len(np.ravel(kpt_dict["mesh"])) == 9:
        lines.append("mesh = [")
        lines.append("  [{:d}, {:d}, {:d}],".format(*kpt_dict["mesh"][0]))
        lines.append("  [{:d}, {:d}, {:d}],".format(*kpt_dict["mesh"][1]))
        lines.append("  [{:d}, {:d}, {:d}]".format(*kpt_dict["mesh"][2]))
        lines.append("]")
    if "shift" in kpt_dict:
        lines.append("shift = [{:f}, {:f}, {:f}]".format(*kpt_dict["shift"]))


def _add_calc_type_scheduler_lines(lines: list, vasp_dict: dict, calc_type: str):
    if "scheduler" in vasp_dict:
        lines.append(f"[vasp.{calc_type}.scheduler]")
        lines.append(tomli_w.dumps(vasp_dict["scheduler"]).strip())


def _get_cell_toml_lines(
    unitcell: PhonopyAtoms, cell_name: str = "cell", show_masses: bool = False
) -> list:
    """Return crystal structure lines in toml.

    Masses are not presented by default because they are inconsistent with those
    in VASP.

    """
    if cell_name:
        lines = [f"[{cell_name}]"]
    lines.append("lattice = [")
    for v, a in zip(unitcell.cell, ("a", "b", "c")):
        lines.append("  [ %21.15f, %21.15f, %21.15f ], # %s" % (v[0], v[1], v[2], a))
    lines.append("]")
    if unitcell.masses is None:
        masses = [None] * len(unitcell.symbols)
    else:
        masses = unitcell.masses
    if unitcell.magnetic_moments is None:
        magnetic_moments = [None] * len(unitcell.symbols)
    else:
        magnetic_moments = unitcell.magnetic_moments
    for i, (s, v, m, mag) in enumerate(
        zip(unitcell.symbols, unitcell.scaled_positions, masses, magnetic_moments)
    ):
        lines.append(f"[[{cell_name}.points]]  # {i + 1}")
        lines.append(f'symbol = "{s}"')
        lines.append(f"coordinates = [ {v[0]:18.15f}, {v[1]:18.15f}, {v[2]:18.15f} ]")
        if show_masses and m is not None:
            lines.append(f"mass = {m:f}")
        if mag is not None:
            if mag.ndim == 0:
                mag_str = f"{mag:.8f}"
            else:
                mag_str = f"[ {mag[0]:.8f}, {mag[1]:.8f}, {mag[2]:.8f} ]"
            lines.append(f"magnetic_moment = {mag_str}")
    return lines


def _show_supercell_dimension(dim: NDArray) -> None:
    if np.array_equal(dim, np.diag(dim.diagonal())):
        click.echo(f"  supercell_dimension: {dim.diagonal()}")
    else:
        click.echo("  supercell_matrix:")
        for v in dim:
            click.echo(f"    {v}")


def _get_supercell_matrix_lines(
    supercell_dimension: NDArray,
) -> list:
    lines = []
    if (np.diag(np.diag(supercell_dimension)) == supercell_dimension).all():
        lines.append(
            "supercell_dimension = [{:d}, {:d}, {:d}]".format(
                *np.diag(supercell_dimension)
            )
        )
    else:
        fmt_str = (
            "supercell_matrix = "
            "[[{:d}, {:d}, {:d}], [{:d}, {:d}, {:d}], [{:d}, {:d}, {:d}]]"
        )
        lines.append(fmt_str.format(*np.ravel(supercell_dimension)))
    return lines


def _get_displacement_settings_lines(
    velph_dict: dict,
    calc_type: Literal["phelel", "phonopy", "phono3py"],
    amplitude: float,
    diagonal: bool,
    plusminus: Literal["auto"] | bool,
) -> list:
    lines = []
    calc_dict = velph_dict.get(calc_type, {})
    if "amplitude" in calc_dict:
        lines.append(f"amplitude = {calc_dict['amplitude']}")
    else:
        lines.append(f"amplitude = {amplitude}")

    if "diagonal" in calc_dict:
        _diagonal = calc_dict["diagonal"]
    else:
        _diagonal = diagonal
    assert isinstance(_diagonal, bool)
    if _diagonal:
        lines.append("diagonal = true")
    else:
        lines.append("diagonal = false")

    if "plusminus" in calc_dict:
        _plusminus = calc_dict["plusminus"]
    else:
        if plusminus is False:
            _plusminus = "auto"
        else:
            _plusminus = True
    if isinstance(_plusminus, bool):
        if _plusminus:
            lines.append("plusminus = true")
        else:
            lines.append("plusminus = false")
    elif isinstance(_plusminus, str):
        if _plusminus == "auto":
            lines.append('plusminus = "auto"')
        else:  # Fall back to default
            lines.append("plusminus = true")
    return lines
