"""Tests of the band paths made by seekpath."""

from __future__ import annotations

import numpy as np
import pytest
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture, get_primitive

from phelel.velph.utils.kpath import get_seekpath_line_path

pytest.importorskip("seekpath")

_FCC = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]


def test_get_seekpath_line_path(nacl_cell: PhonopyAtoms):
    """The band path of the fcc primitive cell of NaCl is that of seekpath."""
    primitive = get_primitive(nacl_cell, _FCC)
    line_path = get_seekpath_line_path(primitive, 51)
    assert line_path.line == 51
    assert line_path.label is not None and line_path.path is not None
    assert line_path.label[:3] == (("GAMMA", "X"), ("X", "U"), ("K", "GAMMA"))
    np.testing.assert_allclose(
        np.array(line_path.path[:3]),
        np.array(
            [
                [[0, 0, 0], [0.5, 0, 0.5]],
                [[0.5, 0, 0.5], [0.625, 0.25, 0.625]],
                [[0.375, 0.375, 0.75], [0, 0, 0]],
            ]
        ),
    )


def test_get_seekpath_line_path_in_other_basis(nacl_cell: PhonopyAtoms):
    """The points are given in the basis of the cell.

    With basis vectors M times those of the primitive cell, a point of
    fractional coordinates k becomes M k.

    """
    primitive = get_primitive(nacl_cell, _FCC)
    basis_change = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    cell = PhonopyAtoms(
        symbols=primitive.symbols,
        cell=basis_change @ primitive.cell,
        scaled_positions=(primitive.scaled_positions @ np.linalg.inv(basis_change)) % 1,
    )
    reference = get_seekpath_line_path(primitive, 51)
    line_path = get_seekpath_line_path(cell, 51)
    assert reference.path is not None and line_path.path is not None
    assert line_path.label == reference.label
    expected = np.einsum("ij,skj->ski", basis_change, np.array(reference.path))
    np.testing.assert_allclose(line_path.path, expected, atol=1e-12)


def test_get_seekpath_line_path_site_mixture():
    """A cell with atoms at the same position gets the band path of its sites."""
    cell = PhonopyAtoms(
        symbols=["Ge", "Sn", "Te"],
        cell=np.eye(3) * 4.0,
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
    )
    line_path = get_seekpath_line_path(
        apply_site_mixture(cell, [0.5, 0.5, 1.0], symprec=1e-5), 51
    )
    assert line_path.label is not None
    assert line_path.label[0] == ("GAMMA", "X")
