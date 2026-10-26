"""Test for classes in spinor.py."""

import pathlib

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry

from phelel.utils.spherical_harmonics import get_n_and_rotation_order
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


def _spinor_rotation_by_sxsysz(
    r: np.ndarray, lattice: np.ndarray, sxsysz: SxSySzMatrices
) -> np.ndarray:
    """Return exp(-i alpha S.n) with S of SxSySzMatrices.

    For spin 1/2, exp(-i alpha S.n) = cos(alpha/2) - 2i sin(alpha/2) S.n.  The
    spin-up is at index 1, as SxSySzMatrices.  For the identity, alpha is 0,
    not 2 pi, which would give -1, as in SpinorRotationMatrices.

    """
    n, r_order, _ = get_n_and_rotation_order(r, lattice)
    alpha = 0.0 if r_order == 1 else 2 * np.pi / r_order
    sn = n[0] * sxsysz.Sx + n[1] * sxsysz.Sy + n[2] * sxsysz.Sz
    return np.cos(alpha / 2) * np.eye(2) - 2j * np.sin(alpha / 2) * sn


def test_SpinorRotationMatrices(rotations_and_lattice: tuple[np.ndarray, np.ndarray]):
    """SpinorRotationMatrices agrees with the rotation by SxSySzMatrices.

    SxSySzMatrices are the Pauli matrices divided by 2 with the spin-up and
    spin-down exchanged.  The rotations of the point groups of the three
    lattices are of the orders 2, 3, 4 and 6, and half of them are improper.

    """

    def spin_swap(mat):
        ss = np.array([[0, 1], [1, 0]])
        return ss @ mat @ ss

    rotations, lattice = rotations_and_lattice
    sxsysz = SxSySzMatrices().run()
    pauli_matrices = PauliMatrices()
    np.testing.assert_allclose(spin_swap(sxsysz.Sx) * 2, pauli_matrices.Sgx)
    np.testing.assert_allclose(spin_swap(sxsysz.Sy) * 2, pauli_matrices.Sgy)
    np.testing.assert_allclose(spin_swap(sxsysz.Sz) * 2, pauli_matrices.Sgz)

    orders = {get_n_and_rotation_order(r, lattice)[1] for r in rotations}
    assert orders in ({1, 2, 4}, {1, 2, 3, 4}, {1, 2, 3, 6})
    for r in rotations:
        delta = SpinorRotationMatrices(r, lattice).run().Delta
        delta_ref = spin_swap(_spinor_rotation_by_sxsysz(r, lattice, sxsysz))
        np.testing.assert_allclose(delta, delta_ref, atol=1e-10)


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
