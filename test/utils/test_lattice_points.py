"""Test for PhelelYaml class."""

import os

import numpy as np

from phelel.utils.lattice_points import get_lattice_points

current_dir = os.path.dirname(os.path.abspath(__file__))


def test_get_lattice_points():
    """Test of get_lattice_points."""
    lps_ref = [
        [0, 0, 0],
        [-1, 0, 0],
        [-2, 0, 0],
        [-3, 0, 0],
        [1, -1, 0],
        [0, -1, 0],
        [-1, -1, 0],
        [-2, -1, 0],
        [2, -2, 0],
        [1, -2, 0],
        [0, -2, 0],
        [-1, -2, 0],
        [3, -3, 0],
        [2, -3, 0],
        [1, -3, 0],
        [0, -3, 0],
        [1, 1, -1],
        [0, 1, -1],
        [-1, 1, -1],
        [-2, 1, -1],
        [2, 0, -1],
        [1, 0, -1],
        [0, 0, -1],
        [-1, 0, -1],
        [3, -1, -1],
        [2, -1, -1],
        [1, -1, -1],
        [0, -1, -1],
        [4, -2, -1],
        [3, -2, -1],
        [2, -2, -1],
        [1, -2, -1],
    ]
    smat = np.diag([2, 2, 2])
    pmat = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    p2s_matrix = np.dot(np.linalg.inv(pmat), smat)
    lps, D = get_lattice_points(p2s_matrix)
    np.testing.assert_array_equal(lps, lps_ref)
    np.testing.assert_array_equal(D, [2, 4, 4])
