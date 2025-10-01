"""Tests velph-phono3py-init."""

import numpy as np
import pytest
import tomli

from phelel.velph.cli.phono3py.init import run_init


@pytest.mark.parametrize("distance", [0.03, 0.05])
def test_phono3py_init_random_displacements(distance: float):
    """Test of plusminus and diagonal with Ti."""
    phelel_str = f"""title = "VASP el-ph settings"

[phelel]
supercell_dimension = [4, 4, 2]
amplitude = {distance}
fft_mesh = [18, 18, 28]
"""

    phono3py_str = f"""[phono3py]
supercell_dimension = [2, 2, 1]
amplitude = {distance}
"""

    unitcell_str = """
[unitcell]
lattice = [
  [     2.930720886111760,     0.000000000000000,     0.000000000000000 ], # a
  [    -1.465360443055880,     2.538078738774425,     0.000000000000000 ], # b
  [     0.000000000000000,     0.000000000000000,     4.646120482318025 ], # c
]
[[unitcell.points]]  # 1
symbol = "Ti"
coordinates = [  0.333333333333336,  0.666666666666664,  0.250000000000000 ]
magnetic_moment = [ 0.00000000, 0.00000000, 0.00000000 ]
[[unitcell.points]]  # 2
symbol = "Ti"
coordinates = [  0.666666666666664,  0.333333333333336,  0.750000000000000 ]
magnetic_moment = [ 0.00000000, 0.00000000, 0.00000000 ]"""

    for toml_dict in (
        tomli.loads(phono3py_str + unitcell_str),
        tomli.loads(phelel_str + phono3py_str + unitcell_str),
    ):
        ph3 = run_init(toml_dict, number_of_snapshots=10)
        assert ph3 is not None
        np.testing.assert_array_equal(ph3.supercell_matrix, np.diag([2, 2, 1]))

        assert len(ph3.supercell) == 8
        assert ph3.displacements.shape == (10, 8, 3)
        np.testing.assert_allclose(np.linalg.norm(ph3.displacements, axis=2), distance)
