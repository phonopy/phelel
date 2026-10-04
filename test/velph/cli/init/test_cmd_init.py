"""Tests CLIs."""

from __future__ import annotations

import dataclasses
import io
import itertools
import pathlib
from collections.abc import Callable
from typing import Literal

import click
import numpy as np
import pytest
import tomli
from phonopy.interface.calculator import read_crystal_structure
from phonopy.interface.phonopy_yaml import load_phonopy_yaml

try:
    from phonopy.phonon.grid import BZGrid, get_ir_grid_points
except ModuleNotFoundError:
    from phono3py.phonon.grid import BZGrid, get_ir_grid_points

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_primitive

from phelel.velph.cli.init.init import (
    _collect_init_params,
    _determine_cell_choices,
    _get_cells,
    _get_kpoints_dict,
    _get_supercell_matrices,
    _get_template_init_params,
    _get_toml_lines,
    _get_velph_dict,
    _parse_velph_template,
    _run_init,
    run_init,
)
from phelel.velph.cli.utils import (
    CellChoice,
    DefaultCellChoices,
    DisplacementOptions,
    PrimitiveCellChoice,
    VelphFilePaths,
    VelphInitOptions,
    VelphInitParams,
)
from phelel.velph.templates import default_template_dict
from phelel.velph.utils.structure import get_symmetry_dataset

cwd = pathlib.Path(__file__).parent


@pytest.mark.parametrize(
    "symmetrize_cell,find_primitive", itertools.product([True, False], repeat=2)
)
def test_run_init_read_cell(
    helper_methods: Callable, symmetrize_cell: bool, find_primitive: bool
):
    """Test combinatons of symmetrize_cell and find_primitive options in run_init .

    Command options: --symmetrize-cell --no-find-primitive

    """
    cell_filepath = cwd / "POSCAR_NaCl"
    cmd_init_options = VelphInitOptions(
        **{
            "symmetrize_cell": symmetrize_cell,
            "find_primitive": find_primitive,
            "supercell_dimension": [2, 2, 2],
        }
    )
    vfp = VelphFilePaths(cell_filepath=cell_filepath)
    toml_lines = run_init(cmd_init_options, vfp)
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    unitcell = load_phonopy_yaml(velph_dict["unitcell"]).unitcell
    assert unitcell is not None
    pcell = load_phonopy_yaml(velph_dict["primitive_cell"]).unitcell
    assert pcell is not None
    cell_ref, _ = read_crystal_structure(cell_filepath, interface_mode="vasp")
    if symmetrize_cell and find_primitive:
        helper_methods.compare_cells(unitcell, cell_ref)
        pcell_ref = get_primitive(unitcell, velph_dict["symmetry"]["primitive_matrix"])
        helper_methods.compare_cells(pcell, pcell_ref)
    elif symmetrize_cell and not find_primitive:
        helper_methods.compare_cells(unitcell, cell_ref)
        helper_methods.compare_cells(pcell, cell_ref)
    elif not symmetrize_cell and find_primitive:
        helper_methods.compare_cells(unitcell, cell_ref)
        pcell_ref = get_primitive(unitcell, velph_dict["symmetry"]["primitive_matrix"])
        helper_methods.compare_cells(pcell, pcell_ref)
    elif not symmetrize_cell and not find_primitive:
        helper_methods.compare_cells_with_order(unitcell, cell_ref)
        helper_methods.compare_cells_with_order(pcell, cell_ref)


def test_run_init_read_cell_and_magmom():
    """Test read_magmom."""
    cell_filepath = cwd / "POSCAR_NaCl"
    magmom = "1 1 1 1 -1 -1 -1 -1"
    cmd_init_options = VelphInitOptions(
        find_primitive=True,
        supercell_dimension=(2, 2, 2),
        magmom=magmom,
    )
    vfp = VelphFilePaths(cell_filepath=cell_filepath)
    toml_lines = run_init(cmd_init_options, vfp)
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    unitcell = load_phonopy_yaml(velph_dict["unitcell"]).unitcell
    assert unitcell is not None
    assert unitcell.magnetic_moments is not None
    pcell = load_phonopy_yaml(velph_dict["primitive_cell"]).unitcell
    assert pcell is not None
    assert pcell.magnetic_moments is not None
    np.testing.assert_allclose(unitcell.magnetic_moments, [1, 1, 1, 1, -1, -1, -1, -1])
    np.testing.assert_allclose(pcell.magnetic_moments, [1, -1])


def _site_mixture_cell() -> PhonopyAtoms:
    """Return a CsCl-like cell with a co-located Ge/Sn site and a Te site."""
    return PhonopyAtoms(
        symbols=["Ge", "Sn", "Te"],
        cell=np.eye(3) * 4.0,
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
    )


def _run_init_site_mixture(options: VelphInitOptions) -> dict:
    toml_lines = _run_init(
        _site_mixture_cell(), options, velph_template_fp=io.BytesIO(b"")
    )
    assert toml_lines is not None
    return tomli.loads("\n".join(toml_lines))


@pytest.mark.parametrize("symmetrize_cell", [False, True])
def test_run_init_site_mixture(symmetrize_cell: bool):
    """Test --site-mixture writes per-atom weights into velph.toml.

    Weights survive both the default (find_primitive) path and the
    --symmetrize-cell standardization path, and a pure site carries an
    explicit weight of 1.0.

    """
    velph_dict = _run_init_site_mixture(
        VelphInitOptions(
            site_mixture="0.5 0.5 1.0",
            split_site_mixture=True,
            symmetrize_cell=symmetrize_cell,
            supercell_dimension=(2, 2, 2),
        )
    )
    for cell_key in ("unitcell", "primitive_cell"):
        cell = load_phonopy_yaml(velph_dict[cell_key]).unitcell
        assert cell is not None
        assert cell.symbols == ["Ge", "Sn", "Te"]
        assert cell.mixture_weights is not None
        np.testing.assert_allclose(cell.mixture_weights, [0.5, 0.5, 1.0])


def test_run_init_site_mixture_via_template():
    """Test site_mixture / split_site_mixture flow through [init.options]."""
    template = (
        b'[init.options]\nsite_mixture = "0.5 0.5 1.0"\nsplit_site_mixture = true\n'
    )
    toml_lines = _run_init(
        _site_mixture_cell(),
        VelphInitOptions(supercell_dimension=(2, 2, 2)),
        velph_template_fp=io.BytesIO(template),
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    unitcell = load_phonopy_yaml(velph_dict["unitcell"]).unitcell
    assert unitcell is not None
    assert unitcell.mixture_weights is not None
    np.testing.assert_allclose(unitcell.mixture_weights, [0.5, 0.5, 1.0])


def test_run_init_site_mixture_with_magmom_raises():
    """Test --site-mixture cannot be combined with --magmom."""
    with pytest.raises(click.ClickException):
        _run_init_site_mixture(
            VelphInitOptions(
                site_mixture="0.5 0.5 1.0",
                magmom="1 1 1",
                supercell_dimension=(2, 2, 2),
            )
        )


def test_run_init_without_max_num_atoms(
    nacl_cell: PhonopyAtoms,
):
    """Test run_init without max_num_atoms."""
    template_str = "\n".join([]).encode("utf-8")
    velph_template_fp = io.BytesIO(template_str)
    toml_lines = _run_init(
        nacl_cell, VelphInitOptions(), velph_template_fp=velph_template_fp
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    assert "phelel" in velph_dict
    assert "supercell_dimension" not in velph_dict["phelel"]
    assert "supercell_matrix" not in velph_dict["phelel"]


def test_get_toml_lines_minimum(nacl_cell: PhonopyAtoms):
    """Minimum test of _get_toml_lines.

    Minimum number of the functions in _run_init are called in this test.

    """
    velph_dict = {}
    input_cell = nacl_cell
    unitcell, primitive, sym_dataset = _get_cells(
        input_cell, 1e-5, True, True, PrimitiveCellChoice.STANDARDIZED
    )
    vip = VelphInitParams(displacement_options=DisplacementOptions(max_num_atoms=100))
    cell_choices = dataclasses.asdict(DefaultCellChoices())

    supercell_matrices = _get_supercell_matrices(vip, velph_dict, sym_dataset)
    (
        kpoints_dict,
        kpoints_dense_dict,
        qpoints_dict,
        kpoints_opt_dict,
    ) = _get_kpoints_dict(
        vip,
        velph_dict,
        unitcell,
        primitive,
        sym_dataset,
        supercell_matrices,
        cell_choices,
    )
    toml_lines = _get_toml_lines(
        velph_dict,
        vip,
        unitcell,
        primitive,
        cell_choices,
        supercell_matrices,
        kpoints_dict,
        kpoints_dense_dict,
        qpoints_dict,
        kpoints_opt_dict,
        sym_dataset,
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))


def test_get_toml_lines_medium(nacl_cell: PhonopyAtoms):
    """Medium test of _get_toml_lines.

    Most of the functions in _run_init are called in this test.

    """
    input_cell = nacl_cell
    velph_template_dict = _parse_velph_template(velph_template_fp=io.BytesIO(b""))
    velph_dict = _get_velph_dict(velph_template_dict)
    template_init_params = _get_template_init_params(velph_template_dict, None)
    vip = _collect_init_params(
        cmd_init_options=VelphInitOptions(supercell_dimension=(2, 2, 2)),
        template_init_params=template_init_params,
        template_toml_filepath=pathlib.Path(""),
    )
    assert vip is not None
    unitcell, primitive, sym_dataset = _get_cells(
        input_cell,
        vip.tolerance,
        vip.symmetrize_cell,
        vip.find_primitive,
        vip.primitive_cell_choice,
    )
    cell_choices = _determine_cell_choices(vip, velph_dict)
    supercell_matrices = _get_supercell_matrices(vip, velph_dict, sym_dataset)
    (
        kpoints_dict,
        kpoints_dense_dict,
        qpoints_dict,
        kpoints_opt_dict,
    ) = _get_kpoints_dict(
        vip,
        velph_dict,
        unitcell,
        primitive,
        sym_dataset,
        supercell_matrices,
        cell_choices,
    )
    toml_lines = _get_toml_lines(
        velph_dict,
        vip,
        unitcell,
        primitive,
        cell_choices,
        supercell_matrices,
        kpoints_dict,
        kpoints_dense_dict,
        qpoints_dict,
        kpoints_opt_dict,
        sym_dataset,
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))


@pytest.mark.parametrize(
    "cell_for_relax,cell_for_nac",
    itertools.product(["primitive", "unitcell", None], repeat=2),
)
def test_run_init_cmd_option_cell_choices(
    cell_for_relax: Literal["primitive", "unitcell"] | None,
    cell_for_nac: Literal["primitive", "unitcell"] | None,
):
    """Simple test of cell_for_nac and cell_for_relax from cmd-line options.

    Choices are primitive or unitcell.

    """
    cell_filepath = cwd / "POSCAR_NaCl"
    cmd_init_options: dict = {"supercell_dimension": (2, 2, 2)}
    if cell_for_relax:
        cmd_init_options["cell_for_relax"] = cell_for_relax
    if cell_for_nac:
        cmd_init_options["cell_for_nac"] = cell_for_nac
    vfp = VelphFilePaths(cell_filepath=cell_filepath)
    toml_lines = run_init(VelphInitOptions(**cmd_init_options), vfp)
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))

    _test_velph_dict_cell_choices(velph_dict, "relax", cell_for_relax)
    _test_velph_dict_cell_choices(velph_dict, "nac", cell_for_nac)


@pytest.mark.parametrize(
    "cell_for_relax,cell_for_nac",
    itertools.product(["primitive", "unitcell", None], repeat=2),
)
def test_run_init_template_init_options_cell_choices(
    nacl_cell: PhonopyAtoms,
    cell_for_relax: Literal["primitive", "unitcell"] | None,
    cell_for_nac: Literal["primitive", "unitcell"] | None,
):
    """Test of cell_for_nac and cell_for_relax from [init.options].

    Choices are primitive or unitcell.

    """
    input_cell = nacl_cell
    template_lines = ["[init.options]", "supercell_dimension = [2, 2, 2]"]
    if cell_for_relax:
        template_lines += [f'cell_for_relax = "{cell_for_relax}"']
    if cell_for_nac:
        template_lines += [f'cell_for_nac = "{cell_for_nac}"']
    template_str = "\n".join(template_lines).encode("utf-8")
    velph_template_fp = io.BytesIO(template_str)
    toml_lines = _run_init(
        input_cell, VelphInitOptions(), velph_template_fp=velph_template_fp
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))

    _test_velph_dict_cell_choices(velph_dict, "relax", cell_for_relax)
    _test_velph_dict_cell_choices(velph_dict, "nac", cell_for_nac)


@pytest.mark.parametrize(
    "cell_for_relax,cell_for_nac",
    itertools.product(["primitive", "unitcell", None], repeat=2),
)
def test_run_init_template_vasp_calc_cell_cell_choices(
    nacl_cell: PhonopyAtoms,
    cell_for_relax: Literal["primitive", "unitcell"] | None,
    cell_for_nac: Literal["primitive", "unitcell"] | None,
):
    """Test of cell_for_nac and cell_for_relax from [vasp.*.cell].

    Choices are primitive or unitcell.

    """
    input_cell = nacl_cell
    template_lines = ["[init.options]", "supercell_dimension = [2, 2, 2]"]
    if cell_for_relax:
        template_lines += ["[vasp.relax]", f'cell = "{cell_for_relax}"']
    if cell_for_nac:
        template_lines += ["[vasp.nac]", f'cell = "{cell_for_nac}"']
    template_str = "\n".join(template_lines).encode("utf-8")
    velph_template_fp = io.BytesIO(template_str)
    toml_lines = _run_init(
        input_cell, VelphInitOptions(), velph_template_fp=velph_template_fp
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))

    _test_velph_dict_cell_choices(velph_dict, "relax", cell_for_relax)
    _test_velph_dict_cell_choices(velph_dict, "nac", cell_for_nac)


@pytest.mark.parametrize(
    "calc_type,cell_choices",
    itertools.product(
        ["relax", "nac"],
        itertools.product(["primitive", "unitcell", None], repeat=3),
    ),
)
def test_run_init_combination_relax_options_and_tag_cell_choices(
    nacl_cell: PhonopyAtoms,
    calc_type: Literal["relax", "nac"],
    cell_choices: tuple[
        Literal["primitive", "unitcell"] | None,
        Literal["primitive", "unitcell"] | None,
        Literal["primitive", "unitcell"] | None,
    ],
):
    """Test for three ways and their combinations to specify cell_for-calcs.

    calc_type :
        ["relax", "nac"]
    cell_choices for cmd-line-options, [init.options], [vasp.*.cell] :
        ["primitive", "unitcell", None]

    """
    cell_for_calc_cmd, cell_for_calc, vasp_calc_cell = cell_choices
    input_cell = nacl_cell
    velph_template_fp = None

    template_lines = ["[init.options]", "supercell_dimension = [2, 2, 2]"]
    if cell_for_calc:
        template_lines += [f'cell_for_{calc_type} = "{cell_for_calc}"']
    if vasp_calc_cell:
        template_lines += [f"[vasp.{calc_type}]"]
        template_lines += [f'cell = "{vasp_calc_cell}"']
    velph_template_fp = io.BytesIO("\n".join(template_lines).encode("utf-8"))

    cmd_init_options: dict = {"supercell_dimension": (2, 2, 2)}
    if cell_for_calc_cmd:
        cmd_init_options[f"cell_for_{calc_type}"] = cell_for_calc_cmd
    toml_lines = _run_init(
        input_cell,
        VelphInitOptions(**cmd_init_options),
        velph_template_fp=velph_template_fp,
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))

    if cell_for_calc_cmd:
        assert velph_dict["vasp"][f"{calc_type}"]["cell"] == cell_for_calc_cmd
    elif cell_for_calc and cell_for_calc_cmd is None:
        assert velph_dict["vasp"][f"{calc_type}"]["cell"] == cell_for_calc
    elif vasp_calc_cell and cell_for_calc is None and cell_for_calc_cmd is None:
        assert velph_dict["vasp"][f"{calc_type}"]["cell"] == vasp_calc_cell
    else:
        _test_velph_dict_cell_choices(velph_dict, calc_type, None)


@pytest.mark.parametrize("symmetrize_cell", [True, False])
def test_run_init_show_toml(symmetrize_cell: bool):
    """Show toml."""
    cell_filepath = cwd / "POSCAR_NaCl"
    vio = VelphInitOptions(**{"symmetrize_cell": symmetrize_cell, "max_num_atoms": 120})
    vfp = VelphFilePaths(cell_filepath=cell_filepath)
    toml_lines = run_init(vio, vfp)
    if symmetrize_cell:
        assert toml_lines is not None
        print("\n".join(toml_lines))
    else:
        assert toml_lines is None


@pytest.mark.parametrize(
    "in_template,in_options,in_phelel",
    itertools.product([True, False], repeat=3),
)
def test_run_init_template_amplitude(
    nacl_cell: PhonopyAtoms, in_template: bool, in_options: bool, in_phelel: bool
):
    """Test of preference of amplitude in init option and [phelel].

    Preference order:
        cmd_options > [init.options] > [phelel] > default

    See test_run_init_template_displacement_options for "diagonal" and
    "plusminus". "nosym" in [phelel] of template is not read.

    """
    input_cell = nacl_cell
    template_lines = ["[init.options]", "supercell_dimension = [2, 2, 2]"]
    cmd_options = {}
    if in_template:
        template_lines += ["amplitude = 0.06"]
    if in_options:
        cmd_options["amplitude"] = 0.04
    if in_phelel:
        template_lines += ["[phelel]", "amplitude = 0.05"]
    template_str = "\n".join(template_lines).encode("utf-8")
    velph_template_fp = io.BytesIO(template_str)
    toml_lines = _run_init(
        input_cell, VelphInitOptions(**cmd_options), velph_template_fp=velph_template_fp
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    if in_options:
        amplitude_ref = 0.04
    elif in_template:
        amplitude_ref = 0.06
    elif in_phelel:
        amplitude_ref = 0.05
    else:
        amplitude_ref = 0.03
    np.testing.assert_allclose(velph_dict["phelel"]["amplitude"], amplitude_ref)


@pytest.mark.parametrize(
    "calc_type,key,values,sources",
    [
        (calc_type, key, values, sources)
        for calc_type in ("phelel", "phonopy", "phono3py")
        for key, values in (
            ("amplitude", (0.05, 0.04)),
            ("diagonal", (True, False)),
            ("plusminus", ("auto", False)),
        )
        for sources in (
            ("template_section",),
            ("init_options", "template_section"),
            ("cmd_options", "template_section"),
            ("cmd_options", "init_options"),
        )
    ],
)
def test_run_init_template_displacement_options(
    nacl_cell: PhonopyAtoms,
    calc_type: str,
    key: str,
    values: tuple,
    sources: tuple[str, ...],
):
    """Test of preference of amplitude, diagonal, and plusminus.

    Preference order:
        cmd_options > [init.options] > [phelel], [phonopy], [phono3py] > default

    The first source in sources is given values[0], which differs from the
    default, and the second source is given values[1]. values[0] is expected.

    """

    def toml_value(value: float | bool | str) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, str):
            return f'"{value}"'
        return str(value)

    source_values = dict(zip(sources, values, strict=False))
    template_lines = ["[init.options]", "supercell_dimension = [2, 2, 2]"]
    cmd_init_options: dict = {}
    if "init_options" in source_values:
        template_lines += [f"{key} = {toml_value(source_values['init_options'])}"]
    if "cmd_options" in source_values:
        cmd_init_options[key] = source_values["cmd_options"]
    if "template_section" in source_values:
        template_lines += [
            f"[{calc_type}]",
            f"{key} = {toml_value(source_values['template_section'])}",
        ]
    toml_lines = _run_init(
        nacl_cell,
        VelphInitOptions(**cmd_init_options),
        velph_template_fp=io.BytesIO("\n".join(template_lines).encode("utf-8")),
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    assert velph_dict[calc_type][key] == values[0]


@pytest.mark.parametrize("plusminus", [True, False, "auto"])
def test_run_init_template_init_options_plusminus(
    nacl_cell: PhonopyAtoms, plusminus: bool | str
):
    """Test that plusminus in [init.options] is not overridden by default.

    VelphInitOptions() is used without plusminus, i.e., no command-line option.

    """
    template_lines = ["[init.options]", "supercell_dimension = [2, 2, 2]"]
    if plusminus == "auto":
        template_lines += ['plusminus = "auto"']
    else:
        template_lines += [f"plusminus = {str(plusminus).lower()}"]
    toml_lines = _run_init(
        nacl_cell,
        VelphInitOptions(),
        velph_template_fp=io.BytesIO("\n".join(template_lines).encode("utf-8")),
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    assert velph_dict["phelel"]["plusminus"] == plusminus


@pytest.mark.parametrize(
    "plusminus,diagonal",
    itertools.product([True, False, "auto", None], [True, False, None]),
)
def test_run_init_plusminus_diagonal(
    plusminus: bool | Literal["auto"] | None, diagonal: bool | None
):
    """Test of plusminus and diagonal command line options."""
    cell_filepath = cwd / "POSCAR_Ti"
    command_options = {
        "plusminus": plusminus,
        "diagonal": diagonal,
        "amplitude": 0.05,
        "max_num_atoms": 80,
        "symmetrize_cell": True,
    }
    default_values = DisplacementOptions()
    vfp = VelphFilePaths(cell_filepath=cell_filepath)
    toml_lines = run_init(VelphInitOptions(**command_options), vfp)
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    if diagonal is None:
        _diagonal = default_values.diagonal
    else:
        _diagonal = diagonal
    assert velph_dict["phelel"]["diagonal"] is _diagonal
    if plusminus == "auto":
        _plusminus = "auto"
    elif plusminus is None:
        _plusminus = default_values.plusminus
    else:
        _plusminus = plusminus
    if isinstance(_plusminus, str):
        assert velph_dict["phelel"]["plusminus"] == _plusminus
    else:
        assert velph_dict["phelel"]["plusminus"] is _plusminus
    np.testing.assert_almost_equal(velph_dict["phelel"]["amplitude"], 0.05)
    np.testing.assert_array_equal(
        velph_dict["phelel"]["supercell_dimension"], [4, 4, 2]
    )


@pytest.mark.parametrize("index,kspacing,num_ir_kpts", [(0, 0.1, 360), (1, 0.2, 50)])
def test_run_init_with_use_grg(
    tio2_prim_cell: PhonopyAtoms, index: int, kspacing: float, num_ir_kpts: int
):
    """Return velph_dict by running _run_init with use_grg.

    VASP returns 360 and 50 ir-kpoints. These values are compared with those
    obtained from phono3py.

    """
    ref_grid = (
        [[0, 17, 17], [17, 0, 17], [7, 7, 0]],
        [[0, 8, 8], [8, 0, 8], [3, 3, 0]],
    )
    input_cell = tio2_prim_cell
    toml_lines = _run_init(
        input_cell,
        VelphInitOptions(
            use_grg=True,
            kspacing=kspacing,
            max_num_atoms=120,
            symmetrize_cell=True,
        ),
    )
    # print("\n".join(toml_lines))
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    for calc in ("relax", "nac"):
        try:
            if velph_dict["vasp"][calc]["cell"] == "primitive":
                mesh = velph_dict["vasp"][calc]["kpoints"]["mesh"]
                assert np.array(mesh).shape == (3, 3)
                np.testing.assert_array_equal(mesh, ref_grid[index])
        except KeyError:
            pass

    sym_dataset = get_symmetry_dataset(tio2_prim_cell)
    bzgrid = BZGrid(
        ref_grid[index],
        lattice=tio2_prim_cell.cell,
        symmetry_dataset=sym_dataset,
        use_grg=True,
    )
    assert len(get_ir_grid_points(bzgrid)[0]) == num_ir_kpts


@pytest.mark.parametrize("primitive_cell_choice", ["standardized", "reduced"])
def test_run_init_with_primitive_cell_choice(
    bi2te3_prim_cell: PhonopyAtoms, primitive_cell_choice: str
):
    """Return velph_dict by running _run_init with primitive_cell_choice.

    VASP returns 360 and 50 ir-kpoints. These values are compared with those
    obtained from phono3py.

    """
    input_cell = bi2te3_prim_cell
    toml_lines = _run_init(
        input_cell,
        VelphInitOptions(
            **{
                "primitive_cell_choice": primitive_cell_choice,
                "max_num_atoms": 12,
                "symmetrize_cell": True,
            }
        ),
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))

    # Check lenghts of basis vectors.
    lattice = velph_dict["primitive_cell"]["lattice"]
    lengths = np.linalg.norm(lattice, axis=1)
    if primitive_cell_choice == "standardized":
        ref_lengths = [10.78873291, 10.78873291, 10.78873291]
    elif primitive_cell_choice == "reduced":
        ref_lengths = [4.44300549, 4.44300549, 10.78873291]
    is_found = False
    for ref_perm in itertools.permutations(ref_lengths):
        for perm in itertools.permutations(lengths):
            if np.allclose(ref_perm, perm):
                is_found = True
                break
    assert is_found


@pytest.mark.parametrize("encut", [300, 400, None])
def test_run_init_template_with_vasp_incar(
    nacl_cell: PhonopyAtoms, encut: float | None
):
    """Test of [vasp.incar] settings."""
    input_cell = nacl_cell
    if encut is None:
        template_lines = ["[vasp.incar]"]
    else:
        template_lines = ["[vasp.incar]", f"encut = {encut}"]
    template_str = "\n".join(template_lines).encode("utf-8")
    velph_template_fp = io.BytesIO(template_str)
    toml_lines = _run_init(
        input_cell,
        VelphInitOptions(**{"max_num_atoms": 120, "symmetrize_cell": True}),
        velph_template_fp=velph_template_fp,
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    if encut is None:
        encut_ref = default_template_dict["vasp"]["incar"]["encut"]
    else:
        encut_ref = encut
    for incar in _get_incar_dicts(velph_dict).values():
        assert incar["encut"] == pytest.approx(encut_ref)


@pytest.mark.parametrize("nac_ncore", [None, 2])
def test_run_init_template_incar_merge(nacl_cell: PhonopyAtoms, nac_ncore: int | None):
    """Test of merging [vasp.incar] into [vasp.CALC_TYPE.incar].

    Tags in [vasp.incar] are added to every [vasp.CALC_TYPE.incar] unless the
    tag is given there. A tag set to {} in default_template_dict is removed:
    "sigma" etc. in [vasp.ph_bands.incar] and "ncore" etc. in
    [vasp.nac.incar]. A template value of such a tag replaces {}.

    """
    template_lines = [
        "[vasp.incar]",
        "encut = 450",
        "sigma = 0.02",
        "ncore = 4",
        "[vasp.relax.incar]",
        "encut = 600",
        "nsw = 5",
    ]
    if nac_ncore is not None:
        template_lines += ["[vasp.nac.incar]", f"ncore = {nac_ncore}"]
    toml_lines = _run_init(
        nacl_cell,
        VelphInitOptions(supercell_dimension=(2, 2, 2)),
        velph_template_fp=io.BytesIO("\n".join(template_lines).encode("utf-8")),
    )
    assert toml_lines is not None
    incars = _get_incar_dicts(tomli.loads("\n".join(toml_lines)))
    assert set(incars) == {
        "phelel",
        "phonopy",
        "phono3py",
        "selfenergy",
        "transport",
        "ph_selfenergy",
        "relax",
        "nac",
        "ph_bands",
        "el_bands.bands",
        "el_bands.dos",
    }

    for calc_type, incar in incars.items():
        if calc_type != "relax":
            assert incar["encut"] == 450
        if calc_type != "nac":
            assert incar["ncore"] == 4
        if calc_type != "ph_bands":
            assert incar["sigma"] == pytest.approx(0.02)

    # [vasp.relax.incar] in template updates that in default_template_dict.
    assert incars["relax"]["encut"] == 600
    assert incars["relax"]["nsw"] == 5
    assert incars["relax"]["isif"] == 3

    # Tags of default [vasp.CALC_TYPE.incar] are preferred to [vasp.incar].
    assert incars["el_bands.dos"]["ismear"] == -5

    for tag in ("ismear", "sigma", "ediff", "lreal", "lwave", "lcharg"):
        assert tag not in incars["ph_bands"]
    for tag in ("npar", "kpar"):
        assert tag not in incars["nac"]
    if nac_ncore is None:
        assert "ncore" not in incars["nac"]
    else:
        assert incars["nac"]["ncore"] == nac_ncore


def _get_incar_dicts(velph_dict: dict) -> dict[str, dict]:
    """Return [vasp.CALC_TYPE.incar] including [vasp.el_bands.*.incar]."""
    incars = {}
    for calc_type, calc_dict in velph_dict["vasp"].items():
        if calc_type == "el_bands":
            for subtype, subtype_dict in calc_dict.items():
                incars[f"el_bands.{subtype}"] = subtype_dict["incar"]
        else:
            incars[calc_type] = calc_dict["incar"]
    return incars


def test_run_init_template_with_vasp_calc_type_scheduler(nacl_cell: PhonopyAtoms):
    """Test of [vasp.calc_type.scheduler] settings."""
    input_cell = nacl_cell
    template_lines = ["[vasp.selfenergy.scheduler]", 'pe = "mpi* 144"']
    template_str = "\n".join(template_lines).encode("utf-8")
    velph_template_fp = io.BytesIO(template_str)
    toml_lines = _run_init(
        input_cell,
        VelphInitOptions(**{"max_num_atoms": 120, "symmetrize_cell": True}),
        velph_template_fp=velph_template_fp,
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    assert "scheduler" in velph_dict["vasp"]["selfenergy"]
    scheduler_dict = velph_dict["vasp"]["selfenergy"]["scheduler"]
    assert "pe" in scheduler_dict
    assert scheduler_dict["pe"] == "mpi* 144"


def test_run_init_template_from_file(tmp_path: pathlib.Path):
    """Test of velph-template read from a file as velph init does.

    [vasp.el_bands.dos] and [vasp.el_bands.bands] are handled only when the
    template is read from a file. The values are chosen to differ from those in
    default_template_dict.

    """
    template_lines = [
        "[init.options]",
        "supercell_dimension = [2, 2, 2]",
        "amplitude = 0.05",
        "[vasp.incar]",
        "encut = 450",
        "[vasp.selfenergy.scheduler]",
        'pe = "mpi* 144"',
        "[vasp.el_bands.dos.incar]",
        "nedos = 100",
        "[vasp.el_bands.bands.kpoints_opt]",
        "line = 21",
    ]
    template_filepath = tmp_path / "velph-template.toml"
    template_filepath.write_text("\n".join(template_lines))
    vfp = VelphFilePaths(
        cell_filepath=cwd / "POSCAR_NaCl", velph_template_filepath=template_filepath
    )
    toml_lines = run_init(VelphInitOptions(), vfp)
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    assert velph_dict["phelel"]["supercell_dimension"] == [2, 2, 2]
    assert velph_dict["phelel"]["amplitude"] == pytest.approx(0.05)
    assert velph_dict["vasp"]["phelel"]["incar"]["encut"] == pytest.approx(450)
    assert velph_dict["vasp"]["selfenergy"]["scheduler"]["pe"] == "mpi* 144"
    assert velph_dict["vasp"]["el_bands"]["dos"]["incar"]["nedos"] == 100
    assert velph_dict["vasp"]["el_bands"]["bands"]["kpoints_opt"]["line"] == 21


def test_run_init_template_file_and_bytesio_give_same_lines(
    nacl_cell: PhonopyAtoms, tmp_path: pathlib.Path
):
    """Test that a template gives the same velph.toml from a file and BytesIO.

    The other template tests use io.BytesIO. [vasp.el_bands.*] is excluded
    because it is handled only for a file.

    """
    template_str = "\n".join(
        [
            "[init.options]",
            "supercell_dimension = [2, 2, 2]",
            'cell_for_nac = "unitcell"',
            "[phelel]",
            "amplitude = 0.05",
            "[vasp.incar]",
            "encut = 450",
            "[vasp.relax]",
            'cell = "primitive"',
            "[vasp.selfenergy.scheduler]",
            'pe = "mpi* 144"',
            "[scheduler]",
            'scheduler_name = "slurm"',
        ]
    )
    template_filepath = tmp_path / "velph-template.toml"
    template_filepath.write_text(template_str)
    toml_lines_file = _run_init(
        nacl_cell,
        VelphInitOptions(),
        velph_template_fp=template_filepath,
        template_toml_filepath=template_filepath,
    )
    toml_lines_bytesio = _run_init(
        nacl_cell,
        VelphInitOptions(),
        velph_template_fp=io.BytesIO(template_str.encode("utf-8")),
    )
    assert toml_lines_file is not None
    assert toml_lines_file == toml_lines_bytesio


@pytest.mark.parametrize(
    "template_lines,cmd_init_options",
    [
        ([], {}),
        ([], {"supercell_dimension": (2, 2, 2)}),
        (["[phonopy]", "supercell_dimension = [2, 2, 2]"], {}),
        (["[phelel]", "supercell_dimension = [2, 2, 2]"], {}),
    ],
)
def test_run_init_written_vasp_calc_types(
    nacl_cell: PhonopyAtoms, template_lines: list[str], cmd_init_options: dict
):
    """Test which [vasp.CALC_TYPE] sections are written.

    [vasp.phelel], [vasp.phonopy], and [vasp.phono3py] are written when the
    corresponding [phelel], [phonopy], and [phono3py] have a supercell matrix.
    [vasp.selfenergy], [vasp.transport], [vasp.ph_selfenergy], and
    [vasp.ph_bands] are written when [phelel] has a supercell matrix. The
    others are always written. The parameters give cases both with and without
    a supercell matrix for each of [phelel] and [phonopy].

    """
    toml_lines = _run_init(
        nacl_cell,
        VelphInitOptions(**cmd_init_options),
        velph_template_fp=io.BytesIO("\n".join(template_lines).encode("utf-8")),
    )
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    vasp_dict = velph_dict["vasp"]

    def has_supercell(calc_type: str) -> bool:
        calc_dict = velph_dict.get(calc_type, {})
        return "supercell_dimension" in calc_dict or "supercell_matrix" in calc_dict

    for calc_type in ("phelel", "phonopy", "phono3py"):
        assert (calc_type in vasp_dict) == has_supercell(calc_type)
    for calc_type in ("selfenergy", "transport", "ph_selfenergy", "ph_bands"):
        assert (calc_type in vasp_dict) == has_supercell("phelel")
    for calc_type in ("relax", "nac"):
        assert calc_type in vasp_dict
    assert set(vasp_dict["el_bands"]) == {"bands", "dos"}


@pytest.mark.parametrize(
    "calc_type,template_supercell,expected,option_source",
    [
        (calc_type, *template_and_expected, option_source)
        for calc_type in ("phelel", "phonopy", "phono3py")
        for template_and_expected in (
            (
                "supercell_dimension = [2, 2, 3]",
                {"supercell_dimension": [2, 2, 3]},
            ),
            (
                "supercell_matrix = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]",
                {"supercell_matrix": [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]},
            ),
            (
                "supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 3]]",
                {"supercell_dimension": [2, 2, 3]},
            ),
        )
        for option_source in (None, "init_options", "cmd_options")
    ],
)
def test_run_init_template_supercell(
    nacl_cell: PhonopyAtoms,
    calc_type: str,
    template_supercell: str,
    expected: dict,
    option_source: str | None,
):
    """Test of supercell matrix given in [phelel], [phonopy], or [phono3py].

    The supercell matrix in the template section is used when no supercell
    option is given in [init.options] or by command-line options. A diagonal
    matrix is written as supercell_dimension. Only the section where the matrix
    is written in the template is checked.

    Preference order:
        cmd_options, [init.options] > [phelel], [phonopy], [phono3py]

    """
    template_lines = []
    cmd_init_options: dict = {}
    if option_source == "init_options":
        template_lines += ["[init.options]", "supercell_dimension = [3, 3, 3]"]
    elif option_source == "cmd_options":
        cmd_init_options["supercell_dimension"] = (3, 3, 3)
    template_lines += [f"[{calc_type}]", template_supercell]
    toml_lines = _run_init(
        nacl_cell,
        VelphInitOptions(**cmd_init_options),
        velph_template_fp=io.BytesIO("\n".join(template_lines).encode("utf-8")),
    )
    assert toml_lines is not None
    calc_dict = tomli.loads("\n".join(toml_lines))[calc_type]
    supercell = {
        key: calc_dict[key]
        for key in ("supercell_dimension", "supercell_matrix")
        if key in calc_dict
    }
    if option_source is None:
        assert supercell == expected
    else:
        assert supercell == {"supercell_dimension": [3, 3, 3]}


def _get_phelel_supercell(
    nacl_cell: PhonopyAtoms, template_lines: list[str], cmd_init_options: dict
) -> dict:
    toml_lines = _run_init(
        nacl_cell,
        VelphInitOptions(**cmd_init_options),
        velph_template_fp=io.BytesIO("\n".join(template_lines).encode("utf-8")),
    )
    assert toml_lines is not None
    calc_dict = tomli.loads("\n".join(toml_lines))["phelel"]
    return {
        key: calc_dict[key]
        for key in ("supercell_dimension", "supercell_matrix")
        if key in calc_dict
    }


@pytest.mark.parametrize(
    "init_options_lines,cmd_init_options,expected",
    [
        (
            ["max_num_atoms = 8", "symmetrize_cell = true"],
            {"supercell_dimension": (3, 3, 3)},
            {"supercell_dimension": [3, 3, 3]},
        ),
        (
            ["max_num_atoms = 8"],
            {"supercell_dimension": (3, 3, 3)},
            {"supercell_dimension": [3, 3, 3]},
        ),
        (
            ["supercell_dimension = [2, 2, 2]"],
            {"supercell_matrix": (-1, 1, 1, 1, -1, 1, 1, 1, -1)},
            {"supercell_matrix": [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]},
        ),
        (
            ["supercell_matrix = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]"],
            {"supercell_dimension": (3, 3, 3)},
            {"supercell_dimension": [3, 3, 3]},
        ),
    ],
)
def test_run_init_supercell_options_ranked_per_source(
    nacl_cell: PhonopyAtoms,
    init_options_lines: list[str],
    cmd_init_options: dict,
    expected: dict,
):
    """Test that a supercell option on the command line beats [init.options].

    max_num_atoms, supercell_dimension, and supercell_matrix are ranked as a
    group per source. When the command line gives any of them, those in
    [init.options] are not used. Then max_num_atoms in [init.options] does not
    require symmetrize_cell.

    """
    template_lines = ["[init.options]", *init_options_lines]
    supercell = _get_phelel_supercell(nacl_cell, template_lines, cmd_init_options)
    assert supercell == expected


def test_run_init_supercell_cmd_max_num_atoms_beats_init_options(
    nacl_cell: PhonopyAtoms,
):
    """Test that --max-num-atoms beats supercell_dimension in [init.options]."""
    cmd_init_options = {"max_num_atoms": 120, "symmetrize_cell": True}
    supercell_ref = _get_phelel_supercell(nacl_cell, [], cmd_init_options)
    assert supercell_ref != {"supercell_dimension": [3, 3, 3]}
    template_lines = ["[init.options]", "supercell_dimension = [3, 3, 3]"]
    supercell = _get_phelel_supercell(nacl_cell, template_lines, cmd_init_options)
    assert supercell == supercell_ref


@pytest.mark.parametrize("source", ["cmd_options", "init_options"])
@pytest.mark.parametrize(
    "keys",
    [
        ("max_num_atoms", "supercell_dimension"),
        ("max_num_atoms", "supercell_matrix"),
        ("supercell_dimension", "supercell_matrix"),
        ("max_num_atoms", "supercell_dimension", "supercell_matrix"),
    ],
)
def test_run_init_supercell_options_given_together(
    nacl_cell: PhonopyAtoms,
    capsys: pytest.CaptureFixture,
    source: str,
    keys: tuple[str, ...],
):
    """Test that two or more supercell options in one source are an error."""
    cmd_values = {
        "max_num_atoms": 120,
        "supercell_dimension": (2, 2, 2),
        "supercell_matrix": (2, 0, 0, 0, 2, 0, 0, 0, 2),
    }
    toml_values = {
        "max_num_atoms": "120",
        "supercell_dimension": "[2, 2, 2]",
        "supercell_matrix": "[[2, 0, 0], [0, 2, 0], [0, 0, 2]]",
    }
    cmd_init_options: dict = {"symmetrize_cell": True}
    template_lines = ["[init.options]"]
    for key in keys:
        if source == "cmd_options":
            cmd_init_options[key] = cmd_values[key]
        else:
            template_lines.append(f"{key} = {toml_values[key]}")
    toml_lines = _run_init(
        nacl_cell,
        VelphInitOptions(**cmd_init_options),
        velph_template_fp=io.BytesIO("\n".join(template_lines).encode("utf-8")),
    )
    assert toml_lines is None
    assert "Only one of" in capsys.readouterr().err


@pytest.mark.parametrize("cmd_supercell", [False, True])
@pytest.mark.parametrize("calc_type", ["phelel", "phonopy", "phono3py"])
def test_run_init_template_supercell_given_together(
    nacl_cell: PhonopyAtoms,
    capsys: pytest.CaptureFixture,
    calc_type: str,
    cmd_supercell: bool,
):
    """Test that both supercell keys in one template section are an error.

    This holds also when the command line gives the supercell, so that the
    template section is not used.

    """
    template_lines = [
        f"[{calc_type}]",
        "supercell_dimension = [2, 2, 2]",
        "supercell_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]",
    ]
    cmd_init_options: dict = {}
    if cmd_supercell:
        cmd_init_options["supercell_dimension"] = (2, 2, 2)
    toml_lines = _run_init(
        nacl_cell,
        VelphInitOptions(**cmd_init_options),
        velph_template_fp=io.BytesIO("\n".join(template_lines).encode("utf-8")),
    )
    assert toml_lines is None
    assert "Only one of" in capsys.readouterr().err


def _test_velph_dict_cell_choices(
    velph_dict: dict, calc_type: Literal["relax", "nac"], cell_for_calc: str | None
):
    if cell_for_calc:
        assert velph_dict["vasp"][f"{calc_type}"]["cell"] == cell_for_calc
    else:
        dcc = dataclasses.asdict(DefaultCellChoices())
        cell_choice_str = {
            CellChoice.PRIMITIVE: "primitive",
            CellChoice.UNITCELL: "unitcell",
        }
        assert (
            velph_dict["vasp"][f"{calc_type}"]["cell"]
            == cell_choice_str[dcc[calc_type]]
        )
