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
