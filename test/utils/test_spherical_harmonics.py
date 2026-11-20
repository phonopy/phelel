"""Test for classes in spherical_harmonics.py."""

import pathlib

import numpy as np
import pytest

from phelel.utils.spherical_harmonics import (
    EulerAngles,
    LxLyLzMatrices,
    SHRotationMatrices,
    SHRotationMatricesEularAngle,
    get_n_and_rotation_order,
)

cwd = pathlib.Path(__file__).parent


def test_get_n_and_rotation_order():
    """Test get_n_and_rotation_order."""
    ref_data = (
        ([0.0, 0.0, 1.0], 1, 1),
        ([0.0, 0.0, 1.0], 4, 1),
        ([0.0, 0.0, 1.0], 2, 1),
        ([0.0, 0.0, -1.0], 4, 1),
        ([1.0, 0.0, 0.0], 2, 1),
        ([0.7071067811865475, -0.7071067811865475, 0.0], 2, 1),
        ([0.0, 1.0, 0.0], 2, 1),
        ([0.7071067811865475, 0.7071067811865475, 0.0], 2, 1),
    )
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
    for r, ref in zip(rots, ref_data, strict=True):
        n, r_order, detR = get_n_and_rotation_order(r, L)
        # print(f"({n.tolist()}, {r_order}, {detR}),")
        np.testing.assert_array_almost_equal(n, ref[0])
        assert r_order == ref[1]
        assert detR == ref[2]


@pytest.mark.parametrize("r22", [1, -1])
def test_EulerAngles_beta_near_0_and_pi(r22: int):
    """sin(beta) is 0 when R[2, 2] is 1 or -1 up to one ulp.

    The rotation is the 4-fold one about z for r22=1 (beta=0) and the 2-fold
    one about x for r22=-1 (beta=pi).  R[2, 2] off from 1 or -1 by one ulp,
    as rounding errors give for the rotations of a non-orthogonal lattice, has
    to give sin(beta) = 0, not sqrt(eps).

    """
    if r22 == 1:
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype="double")
    else:
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype="double")
    R_exact = R.copy()
    R[2, 2] = np.nextafter(R[2, 2], 0)
    ea = EulerAngles(R)
    ea.run()
    assert ea.angles is not None
    assert ea.angles[3] == 0
    np.testing.assert_allclose(ea.recover_R(), R_exact, atol=1e-15)


def test_SHRotationMatrices(rotations_and_lattice: tuple[np.ndarray, np.ndarray]):
    """SHRotationMatrices agrees with the reference SHRotationMatricesEularAngle.

    The rotations of the point groups of the three lattices are of the orders 2,
    3, 4 and 6, and half of them are improper.

    """
    rotations, lattice = rotations_and_lattice
    orders = {get_n_and_rotation_order(r, lattice)[1] for r in rotations}
    assert orders in ({1, 2, 4}, {1, 2, 3, 4}, {1, 2, 3, 6})
    lxlylz = LxLyLzMatrices().run()
    for r in rotations:
        shr_ea = SHRotationMatricesEularAngle(r, lattice)
        shr_ea.run()
        shr = SHRotationMatrices(r, lattice, lxlylz)
        shr.run()
        for delta_ea, delta in zip(shr_ea.Delta, shr.Delta, strict=True):
            np.testing.assert_allclose(delta, delta_ea, atol=1e-10)


def _get_real_spherical_harmonics(ll: int, points: np.ndarray) -> np.ndarray:
    """Return the real spherical harmonics of l at the points, up to a factor.

    They are written as Cartesian polynomials, independently of phelel, in
    the order m = -l, ..., l.  The coefficients are positive, and their ratios
    are those of the normalized functions, so the functions of each l differ
    from the normalized ones by one common factor.

    """
    x, y, z = points.T
    rr = x**2 + y**2 + z**2
    if ll == 1:
        return np.stack([y, z, x], axis=1)
    if ll == 2:
        return np.stack(
            [
                x * y,
                y * z,
                (3 * z**2 - rr) / (2 * np.sqrt(3)),
                x * z,
                (x**2 - y**2) / 2,
            ],
            axis=1,
        )
    if ll == 3:
        c3 = np.sqrt(5 / 8)
        c2 = np.sqrt(15)
        c1 = np.sqrt(3 / 8)
        c0 = 1 / 2
        return np.stack(
            [
                c3 * y * (3 * x**2 - y**2),
                c2 * x * y * z,
                c1 * y * (5 * z**2 - rr),
                c0 * z * (5 * z**2 - 3 * rr),
                c1 * x * (5 * z**2 - rr),
                c2 / 2 * z * (x**2 - y**2),
                c3 * x * (x**2 - 3 * y**2),
            ],
            axis=1,
        )
    raise ValueError(f"l={ll} is not supported.")


def test_SHRotationMatrices_rotate_real_spherical_harmonics(
    rotations_and_lattice: tuple[np.ndarray, np.ndarray],
):
    """Delta of SHRotationMatrices is the matrix of the rotation S, not of S^-1.

    With S in Cartesian coordinates, the real spherical harmonics written as
    Cartesian polynomials satisfy Y_m(S^-1 r) = sum_m' Y_m'(r) Delta[m', m].
    The rotations of the orders 3, 4 and 6 tell S from S^-1, and the improper
    ones check the parity (-1)^l.

    """
    rotations, lattice = rotations_and_lattice
    points = np.random.default_rng(0).normal(size=(30, 3))
    lxlylz = LxLyLzMatrices().run()
    for r in rotations:
        r_cart = lattice @ r @ np.linalg.inv(lattice)
        shr = SHRotationMatrices(r, lattice, lxlylz)
        shr.run()
        assert shr.Delta is not None
        for ll in (1, 2, 3):
            rotated = _get_real_spherical_harmonics(
                ll, points @ np.linalg.inv(r_cart).T
            )
            np.testing.assert_allclose(
                _get_real_spherical_harmonics(ll, points) @ shr.Delta[ll],
                rotated,
                atol=1e-12,
            )
