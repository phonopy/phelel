"""Test for classes in spherical_harmonics.py."""

import pathlib

import numpy as np

from phelel.utils.spherical_harmonics import (
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
    for r, ref in zip(rots, ref_data):
        n, r_order, detR = get_n_and_rotation_order(r, L)
        # print(f"({n.tolist()}, {r_order}, {detR}),")
        np.testing.assert_array_almost_equal(n, ref[0])
        assert r_order == ref[1]
        assert detR == ref[2]


def test_SHRotationMatrices():
    """Test SHRotationMatrices."""
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

    lxlylz = LxLyLzMatrices().run()

    for detR in (1, -1):
        for r in rots * detR:
            shr_ea = SHRotationMatricesEularAngle(r, L)
            shr_ea.run()
            shr = SHRotationMatrices(r, L, lxlylz)
            shr.run()

            for delta_ea, delta in zip(shr_ea.Delta, shr.Delta):
                np.testing.assert_array_almost_equal(delta_ea, delta)
