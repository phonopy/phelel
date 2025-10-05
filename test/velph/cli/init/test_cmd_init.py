"""Tests CLIs."""

from __future__ import annotations

import dataclasses
import io
import itertools
import pathlib
from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest
import tomli
from phono3py.phonon.grid import BZGrid, get_ir_grid_points
from phonopy.interface.calculator import read_crystal_structure
from phonopy.interface.phonopy_yaml import load_phonopy_yaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_primitive

from phelel.velph.cli.init.init import (
    _collect_init_params,
    _determine_cell_choices,
    _get_cells,
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
    get_symmetry_dataset,
)
from phelel.velph.templates import default_template_dict

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
    toml_lines = _get_toml_lines(
        velph_dict,
        vip,
        unitcell,
        primitive,
        cell_choices,
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
    toml_lines = _get_toml_lines(
        velph_dict,
        vip,
        unitcell,
        primitive,
        cell_choices,
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
        [phelel] > cmd_options > [init.options]

    Similar tests should be written for "diagonal", "plusminus", and
    "phelel_nosym".

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
    if in_phelel:
        np.testing.assert_allclose(velph_dict["phelel"]["amplitude"], 0.05)
    else:
        if in_options:
            np.testing.assert_allclose(velph_dict["phelel"]["amplitude"], 0.04)
        else:
            if in_template:
                np.testing.assert_allclose(velph_dict["phelel"]["amplitude"], 0.06)
    if not (in_phelel or in_options or in_template):
        np.testing.assert_allclose(velph_dict["phelel"]["amplitude"], 0.03)


@pytest.mark.parametrize(
    "plusminus,diagonal",
    itertools.product([True, False], repeat=2),
)
def test_run_init_plusminus_diagonal(plusminus: bool, diagonal: bool):
    """Test of plusminus and diagonal command line options.

    plusminus is True -> True
    plusminus is False -> "auto"

    """
    cell_filepath = cwd / "POSCAR_Ti"
    command_options = {
        "plusminus": plusminus,
        "diagonal": diagonal,
        "amplitude": 0.05,
        "max_num_atoms": 80,
        "symmetrize_cell": True,
    }
    vfp = VelphFilePaths(cell_filepath=cell_filepath)
    toml_lines = run_init(VelphInitOptions(**command_options), vfp)
    assert toml_lines is not None
    velph_dict = tomli.loads("\n".join(toml_lines))
    assert velph_dict["phelel"]["diagonal"] is diagonal
    if plusminus:
        assert velph_dict["phelel"]["plusminus"] is True
    else:
        assert velph_dict["phelel"]["plusminus"] == "auto"
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
    for key in velph_dict["vasp"]:
        try:
            velph_dict["vasp"][key]["incar"]["encut"]
            if encut is None:
                assert velph_dict["vasp"][key]["incar"]["encut"] == pytest.approx(
                    default_template_dict["vasp"]["incar"]["encut"]
                )
            else:
                assert velph_dict["vasp"][key]["incar"]["encut"] == pytest.approx(encut)
        except KeyError:
            print(f"Note [vasp.{key}.incar] doesn't have encut entry.")


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
