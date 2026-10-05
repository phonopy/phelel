"""Tests cli/utils.py."""

import contextlib
import copy
import io
import itertools
import pathlib
from collections.abc import Callable

import click
import numpy as np
import pytest
import tomli
from phonopy.interface.calculator import read_crystal_structure
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.atoms import PhonopyAtoms

from phelel.velph.cli.init.init import _run_init
from phelel.velph.cli.utils import (
    VelphInitOptions,
    choose_cell_in_dict,
    get_nac_params,
    get_scheduler_dict,
    kspacing_to_mesh,
    write_incar,
    write_kpoints_mesh_mode,
)
from phelel.velph.templates import default_template_dict
from phelel.velph.utils.structure import get_reduced_cell


def test_get_scheduler_dict():
    """Test get_scheduler_dict."""
    toml_dict = copy.deepcopy(default_template_dict)
    scheduler_dict = get_scheduler_dict(toml_dict, "phelel")
    assert scheduler_dict["vasp_binary"] == "vasp_std"

    phonon_dict = copy.deepcopy(toml_dict["vasp"]["phelel"])
    toml_dict["vasp"]["phelel"]["phonon"] = phonon_dict
    scheduler_dict = get_scheduler_dict(toml_dict, "phelel.phonon")
    assert scheduler_dict["vasp_binary"] == "vasp_std"

    toml_dict["vasp"]["phelel"]["scheduler"] = {"vasp_binary": "vasp_gam"}
    scheduler_dict = get_scheduler_dict(toml_dict, "phelel")
    assert scheduler_dict["vasp_binary"] == "vasp_gam"


def test_get_reduced_cell_bi2te3(
    helper_methods: Callable, bi2te3_prim_cell: PhonopyAtoms
):
    """Test of get_reduced_cell using Bi2Te3 primitive cell.

    Input cell is primitive rhombohedral.

    """
    ref_cell_str = """lattice:
- [    -2.221502746054457,     3.847755625320100,     0.000000000000000 ] # a
- [    -4.443005492108914,     0.000000000000000,     0.000000000000000 ] # b
- [    -2.221502746054457,     1.282585208440033,    10.479344379716814 ] # c
points:
- symbol: Bi # 1
  coordinates: [  0.398482502929835,  0.398482502929836,  0.804552491210493 ]
  mass: 208.980400
- symbol: Bi # 2
  coordinates: [  0.601517497070165,  0.601517497070165,  0.195447508789506 ]
  mass: 208.980400
- symbol: Te # 3
  coordinates: [  0.213380750346874,  0.213380750346874,  0.359857748959378 ]
  mass: 127.600000
- symbol: Te # 4
  coordinates: [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 127.600000
- symbol: Te # 5
  coordinates: [  0.786619249653126,  0.786619249653126,  0.640142251040622 ]
  mass: 127.600000
"""
    ref_cell = read_cell_yaml(io.StringIO(ref_cell_str))
    reduced_cell = get_reduced_cell(bi2te3_prim_cell)
    ref_lengths = np.linalg.norm(ref_cell.cell, axis=1)
    reduced_lengths = np.linalg.norm(reduced_cell.cell, axis=1)

    # Check lenghts of basis vectors.
    is_found = False
    for ref_perm in itertools.permutations(ref_lengths):
        for reduced_perm in itertools.permutations(reduced_lengths):
            if np.allclose(ref_perm, reduced_perm):
                is_found = True
                break
    assert is_found

    if np.allclose(ref_cell.cell, reduced_cell.cell):
        helper_methods.compare_positions_with_order(
            reduced_cell.scaled_positions, ref_cell.scaled_positions, ref_cell.cell
        )
    else:
        msg = (
            "Reduced cell algorithm may be sensitive to the numerical precision of "
            "computers. Therefore this failure might happen due to it. Please "
            "recondier this test."
        )
        raise AssertionError(msg)


@pytest.mark.parametrize("tag", ["kspacing", "elph_kspacing", "KSPACING"])
def test_write_incar_kspacing_is_error(tmp_path: pathlib.Path, tag: str):
    """Test that kspacing and elph_kspacing in INCAR are errors."""
    with pytest.raises(click.ClickException, match=tag):
        write_incar({"encut": 500, tag: 0.2}, tmp_path)
    assert not (tmp_path / "INCAR").exists()


@pytest.mark.parametrize(
    "kpoints_filename,incar_tag",
    [("KPOINTS", "kspacing"), ("KPOINTS_ELPH", "elph_kspacing")],
)
def test_write_kpoints_mesh_mode_always_writes(
    tmp_path: pathlib.Path, kpoints_filename: str, incar_tag: str
):
    """Test that KPOINTS is written regardless of the INCAR dict."""
    write_kpoints_mesh_mode(
        {incar_tag: 0.2},
        tmp_path,
        "vasp.relax.kpoints",
        {"mesh": [4, 4, 4]},
        kpoints_filename=kpoints_filename,
    )
    assert (tmp_path / kpoints_filename).exists()


@pytest.mark.parametrize("use_grg", [True, False])
def test_kspacing_to_mesh_follows_vasp_kspacing(use_grg: bool):
    """Test that the mesh from kspacing follows VASP KSPACING.

    VASP gives N_i = max(1, ceiling(|b_i| 2 pi / KSPACING)). For the
    conventional unit cell of NaCl (a = 5.69 Angstrom) with spacing 0.25,
    |b_i| 2 pi / KSPACING is 4.42, which gives 5. GR-grid can not be used for
    this non-primitive cell.

    """
    cell, _ = read_crystal_structure(
        pathlib.Path(__file__).parent / "init" / "POSCAR_NaCl", interface_mode="vasp"
    )
    assert cell is not None
    kpoints_dict = {"kspacing": 0.25}
    with pytest.warns() if use_grg else contextlib.nullcontext():
        kspacing_to_mesh(kpoints_dict, cell, use_grg=use_grg)
    assert kpoints_dict["mesh"] == [5, 5, 5]


def _get_nacl_velph_dict() -> dict:
    cell, _ = read_crystal_structure(
        pathlib.Path(__file__).parent / "init" / "POSCAR_NaCl", interface_mode="vasp"
    )
    assert cell is not None
    toml_lines = _run_init(cell, VelphInitOptions())
    assert toml_lines is not None
    return tomli.loads("\n".join(toml_lines))


@pytest.mark.parametrize(
    "value,num_atoms", [("unitcell", 8), ("Unitcell", 8), ("Primitive", 2)]
)
def test_choose_cell_in_dict(tmp_path: pathlib.Path, value: str, num_atoms: int):
    """Test that the cell of [vasp.relax] is chosen case-insensitively."""
    velph_dict = _get_nacl_velph_dict()
    velph_dict["vasp"]["relax"]["cell"] = value
    cell = choose_cell_in_dict(velph_dict, tmp_path / "velph.toml", "relax")
    assert cell is not None
    assert len(cell) == num_atoms


@pytest.mark.parametrize("value", ["primitiv", "primitive_cell", "unitcel"])
def test_choose_cell_in_dict_invalid(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture, value: str
):
    """Test that an invalid cell of [vasp.relax] is an error."""
    velph_dict = _get_nacl_velph_dict()
    velph_dict["vasp"]["relax"]["cell"] = value
    assert choose_cell_in_dict(velph_dict, tmp_path / "velph.toml", "relax") is None
    assert f'"{value}"' in capsys.readouterr().err


def test_get_nac_params_invalid_cell(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    """Test that an invalid cell of [vasp.nac] is an error.

    The cell is checked before vasprun.xml is read.

    """
    velph_dict = _get_nacl_velph_dict()
    velph_dict["vasp"]["nac"]["cell"] = "primitiv"
    cell = choose_cell_in_dict(velph_dict, tmp_path / "velph.toml", "relax")
    assert cell is not None
    nac_params = get_nac_params(
        velph_dict, tmp_path / "vasprun.xml", None, cell, is_symmetry=True
    )
    assert nac_params is None
    assert '"primitiv"' in capsys.readouterr().err
