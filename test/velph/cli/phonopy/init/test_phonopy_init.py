"""Tests velph-phonopy-init."""

import pathlib

import numpy as np
import pytest
import tomli

from phelel.velph.cli.phonopy.init import run_init


@pytest.mark.parametrize("distance", [0.03, 0.05])
def test_phonopy_init_random_displacements(distance: float):
    """Test of plusminus and diagonal with Ti."""
    phelel_str = f"""title = "VASP el-ph settings"

[phelel]
supercell_dimension = [4, 4, 2]
amplitude = {distance}
fft_mesh = [18, 18, 28]
"""

    phonopy_str = f"""[phonopy]
supercell_dimension = [2, 2, 1]
amplitude = {distance}
number_of_snapshots = 10
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
        tomli.loads(phonopy_str + unitcell_str),
        tomli.loads(phelel_str + phonopy_str + unitcell_str),
    ):
        ph = run_init(toml_dict)
        np.testing.assert_array_equal(ph.supercell_matrix, np.diag([2, 2, 1]))

        assert len(ph.supercell) == 8
        assert isinstance(ph.displacements, np.ndarray)
        assert ph.displacements.shape == (10, 8, 3)
        np.testing.assert_allclose(np.linalg.norm(ph.displacements, axis=2), distance)


_TI_UNITCELL = """
[unitcell]
lattice = [
  [ 2.930720886111760, 0.000000000000000, 0.000000000000000 ],
  [ -1.465360443055880, 2.538078738774425, 0.000000000000000 ],
  [ 0.000000000000000, 0.000000000000000, 4.646120482318025 ],
]
[[unitcell.points]]
symbol = "Ti"
coordinates = [ 0.333333333333336, 0.666666666666664, 0.25 ]
[[unitcell.points]]
symbol = "Ti"
coordinates = [ 0.666666666666664, 0.333333333333336, 0.75 ]
"""


def _get_phonopy_displacements(
    displacement_lines: list[str], tmp_path: pathlib.Path
) -> np.ndarray:
    toml_str = "\n".join(
        ["[phonopy]", "supercell_dimension = [2, 2, 1]", *displacement_lines]
    )
    ph = run_init(tomli.loads(toml_str + _TI_UNITCELL), current_directory=tmp_path)
    return np.array(ph.displacements)


def test_phonopy_init_default_displacement_settings(tmp_path: pathlib.Path):
    """Test defaults of plusminus, diagonal, and amplitude in [phonopy].

    When they are not given, the values written by velph init (plusminus=true,
    diagonal=false, amplitude=0.01 for [phonopy]) are used. These give
    displacements that differ from those with the defaults of phonopy ("auto",
    true, and 0.01).

    """
    disps = _get_phonopy_displacements([], tmp_path)
    disps_velph = _get_phonopy_displacements(
        ["plusminus = true", "diagonal = false", "amplitude = 0.01"], tmp_path
    )
    disps_phonopy = _get_phonopy_displacements(
        ['plusminus = "auto"', "diagonal = true", "amplitude = 0.01"], tmp_path
    )
    assert disps.shape == disps_velph.shape
    np.testing.assert_allclose(disps, disps_velph)
    assert disps.shape != disps_phonopy.shape or not np.allclose(disps, disps_phonopy)
