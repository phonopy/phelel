"""Test for time reversal in magnetic symmetry operations."""

import warnings
from collections.abc import Sequence

import numpy as np
import pytest
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry

from phelel.base.Dij_qij import DDijQijFit, DeltaDijQij
from phelel.base.local_potential import (
    DeltaLocalPotential,
    LocalPotentialInterpolationNUFFT,
)
from phelel.base.magnetic_symmetry import check_magnetic_moments, get_time_reversals
from phelel.exception import MagneticMomentsError, MagneticMomentsWarning
from phelel.utils.spinor import (
    apply_time_reversal_in_spin_space,
    rotate_delta_vals_in_spin_space,
)

pauli = np.array(
    [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex
)


def _to_spinor(v: float, b: np.ndarray) -> list[np.ndarray]:
    """Return v + b.sigma in data order [11, 12, 21, 22]."""
    mat = v * np.eye(2) + np.einsum("i,ijk->jk", b, pauli)
    return [np.array([mat[i, j]]) for i in (0, 1) for j in (0, 1)]


def _bcc_fe(
    magmoms: Sequence[float] | Sequence[Sequence[float]] | None,
) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=["Fe", "Fe"],
        cell=np.eye(3) * 2.83,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        magnetic_moments=magmoms,
    )


def test_apply_time_reversal_in_spin_space():
    """Time reversal keeps the scalar part and flips the magnetic part."""
    b = np.array([0.3, -0.7, 1.1])
    vals = apply_time_reversal_in_spin_space(_to_spinor(2.0, b))
    expected = _to_spinor(2.0, -b)
    for val, ref in zip(vals, expected, strict=True):
        np.testing.assert_allclose(val, ref, atol=1e-12)

    up, down = np.array([1.0]), np.array([3.0])
    vals = apply_time_reversal_in_spin_space([up, down])
    np.testing.assert_allclose(vals, [down, up])


def test_get_time_reversals():
    """Half of the magnetic operations of FM bcc Fe (m || z) have time reversal."""
    sym = Symmetry(_bcc_fe(None))
    assert not get_time_reversals(sym).any()

    sym = Symmetry(_bcc_fe([[0, 0, 3], [0, 0, 3]]))
    time_reversals = get_time_reversals(sym)
    assert len(time_reversals) == len(sym.symmetry_operations["rotations"])
    assert time_reversals.sum() * 2 == len(time_reversals)


def test_get_time_reversals_nosym():
    """Without symmetry, no operation has time reversal even with magmoms."""
    sym = Symmetry(_bcc_fe([[0, 0, 3], [0, 0, 3]]), is_symmetry=False)
    time_reversals = get_time_reversals(sym)
    assert len(time_reversals) == 1
    assert not time_reversals.any()


@pytest.mark.parametrize(
    "magmoms,ncdij,expected",
    [
        (None, 1, "ok"),
        (None, 2, "warning"),
        (None, 4, "warning"),
        ([3, 3], 1, "ok"),
        ([[0, 0, 3], [0, 0, 3]], 1, "ok"),
        ([3, 3], 2, "ok"),
        ([0, 0], 2, "ok"),
        ([[0, 0, 3], [0, 0, 3]], 2, "error"),
        ([[0, 0, 0], [0, 0, 0]], 2, "error"),
        ([[0, 0, 3], [0, 0, 3]], 4, "ok"),
        ([[0, 0, 0], [0, 0, 0]], 4, "ok"),
        ([3, 3], 4, "error"),
        ([0, 0], 4, "error"),
    ],
)
def test_check_magnetic_moments(
    magmoms: Sequence[float] | Sequence[Sequence[float]] | None,
    ncdij: int,
    expected: str,
):
    """Collinear moments for ncdij=2 and non-collinear moments for ncdij=4.

    The number of values matters also when all the values are zero. Without
    magnetic moments, a spin-polarized calculation gives a warning.

    """
    cell = _bcc_fe(magmoms)
    if expected == "error":
        with pytest.raises(MagneticMomentsError, match=f"ncdij={ncdij}"):
            check_magnetic_moments(cell, ncdij)
    elif expected == "warning":
        with pytest.warns(MagneticMomentsWarning, match=f"ncdij={ncdij}"):
            check_magnetic_moments(cell, ncdij)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MagneticMomentsWarning)
            check_magnetic_moments(cell, ncdij)


def test_local_potential_rejects_mismatched_magmoms():
    """Collinear moments are rejected for dV of ncdij=4."""
    cell = _bcc_fe([3, 3])
    lpi = LocalPotentialInterpolationNUFFT(
        [2, 2, 2], np.eye(3, dtype="int64"), cell, Symmetry(cell), verbose=False
    )
    V = np.zeros((4, 2, 2, 2), dtype=complex)
    disp = {"number": 0, "displacement": [0.01, 0, 0]}
    with pytest.raises(MagneticMomentsError, match="ncdij=4"):
        lpi.delta_Vs = [DeltaLocalPotential(V, V, disp)]


def test_Dij_qij_rejects_mismatched_magmoms():
    """Non-collinear moments are rejected for dDij of ncdij=2."""
    cell = _bcc_fe([[0, 0, 3], [0, 0, 3]])
    Dij = np.zeros((2, 2, 1, 1), dtype=complex)
    disp = {"number": 0, "displacement": [0.01, 0, 0]}
    lm_channels = [{"channels": [{"l": 0, "m": [0]}]}] * 2
    delta_Dij_qij = DeltaDijQij(Dij, Dij, Dij, Dij, disp, lm_channels)
    with pytest.raises(MagneticMomentsError, match="ncdij=2"):
        DDijQijFit([delta_Dij_qij], cell, Symmetry(cell), verbose=False)


def test_magnetization_invariant_under_magnetic_operations():
    """Uniform B || z is invariant under every operation of the magnetic group.

    Operations that reverse B by the spin rotation must be combined with time
    reversal, otherwise symmetrization averages B out.

    """
    cell = _bcc_fe([[0, 0, 3], [0, 0, 3]])
    sym = Symmetry(cell)
    lattice = cell.cell.T
    b = np.array([0.0, 0.0, 1.0])
    spinor = _to_spinor(0.5, b)
    for r, tr in zip(
        sym.symmetry_operations["rotations"], get_time_reversals(sym), strict=True
    ):
        vals = rotate_delta_vals_in_spin_space(spinor, r, lattice)
        if tr:
            vals = apply_time_reversal_in_spin_space(vals)
        for val, ref in zip(vals, spinor, strict=True):
            np.testing.assert_allclose(val, ref, atol=1e-12)
