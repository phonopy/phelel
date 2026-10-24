"""Test for classes in spinor.py."""

import pathlib

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry

from phelel.utils.spinor import (
    PauliMatrices,
    SpinorRotationMatrices,
    SxSySzMatrices,
    rotate_delta_vals_in_spin_space,
)

cwd = pathlib.Path(__file__).parent

pauli = np.array(
    [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex
)


def _to_spinor(v: float, b: np.ndarray) -> list[np.ndarray]:
    """Return v + b.sigma in data order [11, 12, 21, 22]."""
    mat = v * np.eye(2) + np.einsum("i,ijk->jk", b, pauli)
    return [np.array([mat[i, j]]) for i in (0, 1) for j in (0, 1)]


def test_SpinorRotationMatrices():
    """Test of SpinorRotationMatrices."""

    def spin_swap(mat):
        ss = np.array([[0, 1], [1, 0]])
        return ss @ mat @ ss

    rots = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
        ]
    )
    L = np.array(
        [
            [8.156585120174, 0.0, 0.0],
            [0.0, 8.156585120174, 0.0],
            [0.0, 0.0, 4.767746862166],
        ]
    )

    sxsysz = SxSySzMatrices().run()
    pauli_matrices = PauliMatrices()

    np.testing.assert_array_almost_equal(spin_swap(sxsysz.Sx) * 2, pauli_matrices.Sgx)
    np.testing.assert_array_almost_equal(spin_swap(sxsysz.Sy) * 2, pauli_matrices.Sgy)
    np.testing.assert_array_almost_equal(spin_swap(sxsysz.Sz) * 2, pauli_matrices.Sgz)

    for detR in (1, -1):
        for r in rots * detR:
            sprs = SpinorRotationMatrices(r, L, sxsysz=sxsysz)
            sprs.run()
            sprp = SpinorRotationMatrices(r, L)
            sprp.run()
            np.testing.assert_array_almost_equal(spin_swap(sprs.Delta), sprp.Delta)


def test_rotate_delta_vals_in_spin_space_direction():
    """Spin is rotated by the operation, not by its inverse.

    v + b.sigma is mapped to v + (S+ b).sigma, where S is the rotation in
    Cartesian coordinates and S+ = det(S) S is its proper rotation. The 48
    rotations of the primitive cell of bcc include C3 and C4, which differ from
    their inverses, and b is along no symmetry axis, so the inverse rotation
    gives another result. Half of the rotations are improper, for which S+
    differs from S, so that b is rotated as an axial vector is also tested. The
    basis vectors are not orthogonal, so the conversion of the rotations to
    Cartesian coordinates is also tested.

    """
    lattice = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float).T * 1.4
    cell = PhonopyAtoms(cell=lattice.T, scaled_positions=[[0, 0, 0]], symbols=["Fe"])
    rotations = Symmetry(cell).symmetry_operations["rotations"]
    assert len(rotations) == 48
    b = np.array([0.3, -0.7, 1.1])
    for r in rotations:
        S = lattice @ r @ np.linalg.inv(lattice)
        vals = rotate_delta_vals_in_spin_space(_to_spinor(2.0, b), r, lattice)
        expected = _to_spinor(2.0, np.linalg.det(S) * S @ b)
        for val, ref in zip(vals, expected, strict=True):
            np.testing.assert_allclose(val, ref, atol=1e-12)
