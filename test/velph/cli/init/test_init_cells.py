"""Tests of the cells, symmetry and supercells made by velph init."""

from __future__ import annotations

import itertools
import pathlib
from typing import Any

import click
import numpy as np
import pytest
from phonopy.interface.calculator import read_crystal_structure
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_primitive

from phelel.velph.cli.init.cells import InitCells, make_cells, resolve_supercell
from phelel.velph.config.schema import (
    MaxNumAtoms,
    SupercellDimension,
    SupercellMatrix,
)

cwd = pathlib.Path(__file__).parent

# Primitive matrix of the face-centred lattice, whose columns are the basis
# vectors of the primitive cell in those of the conventional cell.
_F_MATRIX = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]


def _make_cells(cell: PhonopyAtoms, **options: Any) -> InitCells:
    """Return make_cells with the defaults of velph init for missing options."""
    kwargs: dict[str, Any] = {
        "tolerance": 1e-5,
        "symmetrize_cell": False,
        "find_primitive": True,
        "primitive_cell_choice": "standardized",
        "magmom": None,
        "site_mixture": None,
    }
    kwargs.update(options)
    return make_cells(cell, **kwargs)


def _rotated(cell: PhonopyAtoms, degrees: float) -> PhonopyAtoms:
    """Return cell rotated about z in Cartesian coordinates."""
    theta = np.radians(degrees)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    rotated = cell.copy()
    rotated.cell = cell.cell @ rotation.T
    return rotated


def _bct_cell(primitive: bool) -> PhonopyAtoms:
    """Return the primitive or conventional cell of BCT Sn (a = 3, c = 5)."""
    a, c = 3.0, 5.0
    if primitive:
        return PhonopyAtoms(
            cell=[
                [-a / 2, a / 2, c / 2],
                [a / 2, -a / 2, c / 2],
                [a / 2, a / 2, -c / 2],
            ],
            scaled_positions=[[0, 0, 0]],
            symbols=["Sn"],
        )
    return PhonopyAtoms(
        cell=np.diag([a, a, c]),
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        symbols=["Sn", "Sn"],
    )


@pytest.mark.parametrize(
    "symmetrize_cell,find_primitive", list(itertools.product([False, True], repeat=2))
)
def test_make_cells(
    nacl_cell: PhonopyAtoms,
    helper_methods: Any,
    symmetrize_cell: bool,
    find_primitive: bool,
):
    """Test the cells from the conventional cell of NaCl.

    The conventional cell is the standardized cell, so the unit cell is the
    input cell also with symmetrize_cell.  Without find_primitive, the
    primitive cell is the unit cell, and [symmetry] has no primitive_matrix.

    """
    cells = _make_cells(
        nacl_cell, symmetrize_cell=symmetrize_cell, find_primitive=find_primitive
    )
    helper_methods.compare_cells(cells.unitcell, nacl_cell)
    if find_primitive:
        helper_methods.compare_cells(cells.primitive, get_primitive(nacl_cell, "F"))
        assert cells.symmetry.primitive_matrix is not None
        np.testing.assert_allclose(cells.symmetry.primitive_matrix, _F_MATRIX)
    else:
        helper_methods.compare_cells(cells.primitive, nacl_cell)
        assert cells.symmetry.primitive_matrix is None
    assert cells.symmetry.spacegroup_type == "Fm-3m"
    assert cells.symmetry.uni_number is None
    assert cells.symmetry.tolerance == 1e-5


@pytest.mark.parametrize(
    "basis", [np.eye(3, dtype=int), [[1, 0, 0], [0, 1, 0], [1, 0, 1]]]
)
def test_make_cells_input_primitive_cell(
    si_prim_cell: PhonopyAtoms, helper_methods: Any, basis: Any
):
    """The primitive input cell is the primitive cell without symmetrize_cell.

    The second basis is (a1, a2, a1 + a3) of the primitive cell of Si, for which
    the primitive cell made by the transformation matrix of the symmetry
    dataset has the basis (a1, a2, a3).

    """
    cell = si_prim_cell.copy()
    cell.cell = np.array(basis) @ si_prim_cell.cell
    cell.scaled_positions = si_prim_cell.scaled_positions @ np.linalg.inv(basis)
    cells = _make_cells(cell)
    helper_methods.compare_cells_with_order(cells.unitcell, cell)
    helper_methods.compare_cells_with_order(cells.primitive, cell)
    assert cells.symmetry.primitive_matrix is None


@pytest.mark.parametrize(
    "tolerance,spacegroup_type,num_atoms", [(1e-5, "P4mm", 8), (1e-3, "Fm-3m", 2)]
)
def test_make_cells_tolerance(
    nacl_cell: PhonopyAtoms, tolerance: float, spacegroup_type: str, num_atoms: int
):
    """The symmetry is found with tolerance, which is written in [symmetry].

    An atom of the conventional cell of NaCl is displaced by 2.8e-4 Angstrom.

    """
    cell = nacl_cell.copy()
    positions = cell.scaled_positions.copy()
    positions[0] += [5e-5, 0, 0]
    cell.scaled_positions = positions
    cells = _make_cells(cell, tolerance=tolerance)
    assert cells.symmetry.spacegroup_type == spacegroup_type
    assert cells.symmetry.tolerance == tolerance
    assert len(cells.primitive) == num_atoms


@pytest.mark.parametrize("symmetrize_cell", [False, True])
def test_make_cells_orientation(nacl_cell: PhonopyAtoms, symmetrize_cell: bool):
    """The cells keep the orientation of the input cell without symmetrize_cell.

    The input cell is the conventional cell of NaCl rotated by 30 degrees about
    z.  With symmetrize_cell, the cells are the standardized cells of spglib,
    which are not rotated.

    """
    rotated = _rotated(nacl_cell, 30)
    cells = _make_cells(rotated, symmetrize_cell=symmetrize_cell)
    if symmetrize_cell:
        np.testing.assert_allclose(cells.unitcell.cell, nacl_cell.cell, atol=1e-8)
        reference = nacl_cell.cell
    else:
        np.testing.assert_allclose(cells.unitcell.cell, rotated.cell)
        reference = rotated.cell
    np.testing.assert_allclose(
        cells.primitive.cell, np.transpose(_F_MATRIX) @ reference, atol=1e-8
    )


@pytest.mark.parametrize(
    "primitive_cell_choice,lengths",
    [
        ("standardized", [10.78873291, 10.78873291, 10.78873291]),
        ("reduced", [4.44300549, 4.44300549, 10.78873291]),
    ],
)
def test_make_cells_primitive_cell_choice(
    bi2te3_prim_cell: PhonopyAtoms, primitive_cell_choice: str, lengths: list[float]
):
    """The reduced primitive cell of Bi2Te3 has shorter basis vectors."""
    cells = _make_cells(
        bi2te3_prim_cell,
        symmetrize_cell=True,
        primitive_cell_choice=primitive_cell_choice,
    )
    np.testing.assert_allclose(
        sorted(np.linalg.norm(cells.primitive.cell, axis=1)), lengths
    )


@pytest.mark.parametrize("primitive_cell_choice", ["standardized", "reduced"])
def test_make_cells_primitive_matrix(
    bi2te3_prim_cell: PhonopyAtoms, primitive_cell_choice: str
):
    """primitive_matrix of [symmetry] gives the primitive cell from the unit cell.

    It is the primitive matrix of phonopy.  The unit cell of Bi2Te3 is the
    hexagonal cell, and the matrix to the rhombohedral primitive cell is not
    symmetric, so a transposed matrix is found.

    """
    cells = _make_cells(
        bi2te3_prim_cell,
        symmetrize_cell=True,
        primitive_cell_choice=primitive_cell_choice,
    )
    assert cells.symmetry.primitive_matrix is not None
    pmat = np.array(cells.symmetry.primitive_matrix)
    assert not np.allclose(pmat, pmat.T)
    primitive = get_primitive(cells.unitcell, pmat)
    np.testing.assert_allclose(primitive.cell, cells.primitive.cell, atol=1e-8)


def test_make_cells_magmom(nacl_cell: PhonopyAtoms):
    """MAGMOM is given to both cells, and [symmetry] has the UNI number."""
    cells = _make_cells(nacl_cell, magmom="4*1 4*-1")
    assert nacl_cell.magnetic_moments is None
    assert cells.unitcell.magnetic_moments is not None
    np.testing.assert_allclose(cells.unitcell.magnetic_moments, [1] * 4 + [-1] * 4)
    assert cells.primitive.magnetic_moments is not None
    np.testing.assert_allclose(sorted(cells.primitive.magnetic_moments), [-1, 1])
    assert cells.symmetry.spacegroup_type is None
    assert cells.symmetry.uni_number is not None


def _site_mixture_cell() -> PhonopyAtoms:
    """Return a CsCl-like cell with a co-located Ge/Sn site and a Te site."""
    return PhonopyAtoms(
        symbols=["Ge", "Sn", "Te"],
        cell=np.eye(3) * 4.0,
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
    )


@pytest.mark.parametrize("symmetrize_cell", [False, True])
def test_make_cells_site_mixture(symmetrize_cell: bool):
    """The weights of site mixture are given to both cells."""
    cells = _make_cells(
        _site_mixture_cell(),
        site_mixture="0.5 0.5 1.0",
        symmetrize_cell=symmetrize_cell,
    )
    for cell in (cells.unitcell, cells.primitive):
        assert cell.symbols == ["Ge", "Sn", "Te"]
        assert cell.mixture_weights is not None
        np.testing.assert_allclose(cell.mixture_weights, [0.5, 0.5, 1.0])


@pytest.mark.parametrize(
    "options,message",
    [
        ({"site_mixture": "0.5 0.5 1.0", "magmom": "3*0"}, "--magmom"),
        ({"site_mixture": "0.5 0.5"}, "weights"),
    ],
)
def test_make_cells_site_mixture_error(options: dict[str, str], message: str):
    """site_mixture with magmom, or of a wrong length, is an error."""
    with pytest.raises(click.ClickException, match=message):
        _make_cells(_site_mixture_cell(), **options)


@pytest.mark.parametrize(
    "supercell,expected",
    [
        (
            SupercellDimension(supercell_dimension=(2, 2, 3)),
            SupercellDimension(supercell_dimension=(2, 2, 3)),
        ),
        (
            SupercellMatrix(supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 3))),
            SupercellDimension(supercell_dimension=(2, 2, 3)),
        ),
        (
            SupercellMatrix(supercell_matrix=((-1, 1, 1), (1, -1, 1), (1, 1, -1))),
            SupercellMatrix(supercell_matrix=((-1, 1, 1), (1, -1, 1), (1, 1, -1))),
        ),
    ],
)
def test_resolve_supercell(
    nacl_cell: PhonopyAtoms,
    supercell: SupercellDimension | SupercellMatrix,
    expected: SupercellDimension | SupercellMatrix,
):
    """A diagonal supercell matrix is written as supercell_dimension."""
    cells = _make_cells(nacl_cell)
    assert (
        resolve_supercell(
            "phelel", supercell, cells, symmetrize_cell=False, tolerance=1e-5
        )
        == expected
    )


def test_resolve_supercell_max_num_atoms_ti():
    """max_num_atoms 80 gives 4x4x2 of hcp Ti, as velph init of 0.14 does."""
    ti, _ = read_crystal_structure(cwd / "POSCAR_Ti", interface_mode="vasp")
    assert ti is not None
    cells = _make_cells(ti, symmetrize_cell=True)
    supercell = resolve_supercell(
        "phelel",
        MaxNumAtoms(max_num_atoms=80),
        cells,
        symmetrize_cell=True,
        tolerance=1e-5,
    )
    assert supercell == SupercellDimension(supercell_dimension=(4, 4, 2))


def test_resolve_supercell_max_num_atoms_without_symmetrize_cell(
    nacl_cell: PhonopyAtoms,
):
    """max_num_atoms without symmetrize_cell is an error naming the section."""
    cells = _make_cells(nacl_cell)
    with pytest.raises(click.ClickException, match=r"\[phonopy\]"):
        resolve_supercell(
            "phonopy",
            MaxNumAtoms(max_num_atoms=64),
            cells,
            symmetrize_cell=False,
            tolerance=1e-5,
        )


@pytest.mark.parametrize(
    "primitive,dimension,lowered",
    [
        (True, (2, 2, 1), True),
        (True, (2, 2, 2), False),
        (False, (2, 2, 1), False),
    ],
)
def test_resolve_supercell_point_group_warning(
    primitive: bool,
    dimension: tuple[int, int, int],
    lowered: bool,
    capsys: pytest.CaptureFixture[str],
):
    """A warning is shown when the supercell has a lower point group.

    diag(2, 2, 1) of the primitive cell of BCT keeps 4 of the 16 rotations,
    while diag(2, 2, 2) of it and diag(2, 2, 1) of the conventional cell keep
    16.

    """
    cells = _make_cells(_bct_cell(primitive))
    capsys.readouterr()
    resolve_supercell(
        "phelel",
        SupercellDimension(supercell_dimension=dimension),
        cells,
        symmetrize_cell=False,
        tolerance=1e-5,
    )
    out = capsys.readouterr().out
    if lowered:
        assert "Warning: the supercell of [phelel]" in out
        assert "has 4 rotations, fewer than the 16 of the unit cell" in out
        assert "[[2, 0, 0], [0, 2, 0], [0, 0, 1]]" in out
    else:
        assert "Warning" not in out
