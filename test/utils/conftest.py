"""Fixtures of the tests of utils."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry

_LATTICES = {
    "tetragonal": np.diag([8.156585120174, 8.156585120174, 4.767746862166]),
    "cubic": np.eye(3) * 3.0,
    "hexagonal": np.array([[3.0, 0, 0], [-1.5, 1.5 * np.sqrt(3), 0], [0, 0, 5.0]]),
}


@pytest.fixture(params=list(_LATTICES))
def rotations_and_lattice(request) -> tuple[NDArray, NDArray]:
    """Return the rotations of the point group of a lattice and the lattice.

    The rotations are in the basis vectors, and the lattice has the basis
    vectors as columns.  The point groups are 4/mmm, m-3m and 6/mmm, so the
    rotations are of the orders 2 and 4, also 3, and 3 and 6, with the improper
    ones.

    """
    lattice = _LATTICES[request.param]
    cell = PhonopyAtoms(cell=lattice, scaled_positions=[[0, 0, 0]], symbols=["Si"])
    rotations = Symmetry(cell).symmetry_operations["rotations"]
    return rotations, lattice.T
