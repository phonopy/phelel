"""Test for classes in spinor.py."""

import pathlib

import numpy as np

from phelel.utils.spinor import PauliMatrices, SpinorRotationMatrices, SxSySzMatrices

cwd = pathlib.Path(__file__).parent


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
