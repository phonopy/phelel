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
    DLocalPotential,
    DLocalPotentialFit,
)
from phelel.base.magnetic_symmetry import (
    check_magnetic_moments,
    check_primitive_translations,
    get_time_reversals,
    is_grey_group_without_time_reversal,
)
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


def test_get_time_reversals_grey_group():
    """Zero magnetic moments give a grey group, with one flag per operation.

    phonopy 4.10.0 keeps every (R, t) of the grey group with and without time
    reversal, and half of the flags are True.  Later versions keep only the
    operations without time reversal, and all flags are False.  A group that
    is not grey is not reported as a grey group without time reversal.

    """
    sym = Symmetry(_bcc_fe([0, 0]))
    n_ops = len(sym.symmetry_operations["rotations"])
    time_reversals = get_time_reversals(sym)
    assert len(time_reversals) == n_ops
    if is_grey_group_without_time_reversal(sym):
        assert len(sym.dataset.time_reversals) == 2 * n_ops
        assert not time_reversals.any()
    else:
        assert time_reversals.sum() * 2 == n_ops

    assert not is_grey_group_without_time_reversal(
        Symmetry(_bcc_fe([[0, 0, 3], [0, 0, 3]]))
    )
    assert not is_grey_group_without_time_reversal(Symmetry(_bcc_fe(None)))


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
    local_potential_fit = DLocalPotentialFit(
        [2, 2, 2], np.eye(3, dtype="int64"), cell, Symmetry(cell), verbose=False
    )
    V = np.zeros((4, 2, 2, 2), dtype=complex)
    disp = {"number": 0, "displacement": [0.01, 0, 0]}
    with pytest.raises(MagneticMomentsError, match="ncdij=4"):
        local_potential_fit.run([DeltaLocalPotential(V, V, disp)])


def test_Dij_qij_rejects_mismatched_magmoms():
    """Non-collinear moments are rejected for dDij of ncdij=2."""
    cell = _bcc_fe([[0, 0, 3], [0, 0, 3]])
    Dij = np.zeros((2, 2, 1, 1), dtype=complex)
    disp = {"number": 0, "displacement": [0.01, 0, 0]}
    lm_channels = [{"channels": [{"l": 0, "m": [0]}]}] * 2
    delta_Dij_qij = DeltaDijQij(Dij, Dij, Dij, Dij, disp, lm_channels)
    with pytest.raises(MagneticMomentsError, match="ncdij=2"):
        DDijQijFit(cell, Symmetry(cell), verbose=False).run([delta_Dij_qij])


# Supercell matrix of the cubic cell relative to the bcc primitive cell.
BCC_P2S_MATRIX = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype="int64")


@pytest.mark.parametrize(
    "magmoms,expected",
    [
        (None, "ok"),
        ([3, 3], "ok"),
        ([[0, 0, 3], [0, 0, 3]], "ok"),
        ([3, -3], "error"),
        ([3, -1], "error"),
        ([[0, 0, 3], [0, 0, -3]], "error"),
        ([[0, 0, 3], [3, 0, 0]], "error"),
    ],
)
def test_check_primitive_translations(
    magmoms: Sequence[float] | Sequence[Sequence[float]] | None, expected: str
):
    """The body-centring vector of bcc Fe has to keep the magnetic moments.

    It is an anti-translation for the antiferromagnetic moments (3, -3), which
    phonopy allows, and it is no symmetry operation for the others.

    """
    cell = _bcc_fe(magmoms)
    if expected == "error":
        with pytest.raises(ValueError, match="Magnetic moments of the supercell"):
            check_primitive_translations(cell, BCC_P2S_MATRIX, 1e-5)
    else:
        check_primitive_translations(cell, BCC_P2S_MATRIX, 1e-5)
    # The cubic cell itself is always a primitive cell.
    check_primitive_translations(cell, np.eye(3, dtype="int64"), 1e-5)


def test_check_primitive_translations_rejects_non_translation():
    """The body-centring vector of CsCl moves Cs onto Cl."""
    cell = PhonopyAtoms(
        symbols=["Cs", "Cl"],
        cell=np.eye(3) * 4.0,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    with pytest.raises(ValueError, match="not a translation of the supercell"):
        check_primitive_translations(cell, BCC_P2S_MATRIX, 1e-5)


def test_local_potential_rejects_primitive_cell_of_antiferromagnet():
    """DLocalPotential.run rejects the bcc primitive cell of antiferromagnetic Fe.

    The finite differences would give dV/du of atom 0 only, and dV/du of atom 1
    with the opposite moment would be taken wrongly as that of atom 0 moved by
    the body-centring vector.

    """
    cell = _bcc_fe([3, -3])
    V = np.zeros((2, 2, 2, 2), dtype=complex)
    disp = {"number": 0, "displacement": [0.01, 0, 0]}
    dlp = DLocalPotential(
        [2, 2, 2], BCC_P2S_MATRIX, cell, symmetry=Symmetry(cell), verbose=False
    )
    with pytest.raises(ValueError, match="Magnetic moments of the supercell"):
        dlp.run(V, [V], [disp])


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


def test_finite_differences_of_grey_group_keep_time_reversal():
    """Spin up and down of the derivatives are equal for zero magnetic moments.

    The data have different spin up and down, which only the operations with
    time reversal exchange.  If Symmetry lacks them, they are added in the
    finite-difference method, so that the derivatives are the same as with
    phonopy 4.10.0, which keeps them.  Without them, spin up and down differ
    by about the size of the derivatives.  The mesh 4x4x4 and the s and p
    channels are needed for derivatives that are not zero by the site
    symmetry of bcc.

    """
    cell = _bcc_fe([0, 0])
    rng = np.random.default_rng(0)
    disp = {"number": 0, "displacement": [0.01, 0.02, 0.03]}

    V_per = np.zeros((2, 4, 4, 4), dtype=complex)
    V_disp = rng.standard_normal((2, 4, 4, 4)).astype(complex)
    fit = DLocalPotentialFit(
        [4, 4, 4], np.eye(3, dtype="int64"), cell, Symmetry(cell), verbose=False
    )
    fit.run([DeltaLocalPotential(V_per, V_disp, disp)])
    assert np.abs(fit.dVdu).max() > 1
    np.testing.assert_allclose(fit.dVdu[0], fit.dVdu[1], atol=1e-10)

    lm_channels = [{"channels": [{"l": 0, "m": [0]}, {"l": 1, "m": [-1, 0, 1]}]}] * 2
    Dij_per = np.zeros((2, 2, 4, 4), dtype=complex)
    Dij_disp = rng.standard_normal((2, 2, 4, 4)).astype(complex)
    delta_Dij_qij = DeltaDijQij(Dij_per, Dij_disp, Dij_per, Dij_disp, disp, lm_channels)
    Dij_fit = DDijQijFit(cell, Symmetry(cell), verbose=False)
    Dij_fit.run([delta_Dij_qij])
    assert np.abs(Dij_fit.dDijdu).max() > 1
    np.testing.assert_allclose(Dij_fit.dDijdu[0], Dij_fit.dDijdu[1], atol=1e-10)
    np.testing.assert_allclose(Dij_fit.dqijdu[0], Dij_fit.dqijdu[1], atol=1e-10)
