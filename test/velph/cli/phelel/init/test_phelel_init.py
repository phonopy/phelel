"""Tests CLIs."""

import itertools
from pathlib import Path

import numpy as np
import pytest
import tomli

from phelel.velph.cli.phelel.init import run_init

cwd = Path(__file__).parent


@pytest.mark.parametrize(
    "plusminus,diagonal", itertools.product([True, False], repeat=2)
)
def test_supercell_init_plusminus_diagonal(plusminus: bool, diagonal: bool):
    """Test of plusminus and diagonal with Ti."""
    if plusminus:
        pm_str = "true"
    else:
        pm_str = '"auto"'
    if diagonal:
        dg_str = "true"
    else:
        dg_str = "false"
    toml_str = f"""title = "VASP el-ph settings"

[phelel]
supercell_dimension = [4, 4, 2]
amplitude = 0.03
diagonal = {dg_str}
plusminus = {pm_str}
fft_mesh = [18, 18, 28]
[vasp.supercell.incar]
lwap = true
isym = 0
kpar = 2
ncore = 24
ismear = 0
sigma = 0.2
ediff = 1e-08
encut = 329.532
prec = "accurate"
lreal = false
lwave = false
lcharg = false
addgrid = true
lsorbit = true
[vasp.supercell.kpoints]
mesh = [6, 6, 7]

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
magnetic_moment = [ 0.00000000, 0.00000000, 0.00000000 ]
"""

    print(toml_str)
    toml_dict = tomli.loads(toml_str)
    phe = run_init(toml_dict)
    np.testing.assert_array_equal(phe.supercell_matrix, np.diag([4, 4, 2]))
    print(phe.dataset["first_atoms"])
    if plusminus and diagonal:
        disps = [
            [0.023510024693335307, 0.0, 0.018635416252897705],
            [-0.023510024693335307, 0.0, -0.018635416252897705],
        ]
        assert len(phe.dataset["first_atoms"]) == 2
    if not plusminus and diagonal:
        disps = [[0.023510024693335307, 0.0, 0.018635416252897705]]
        assert len(phe.dataset["first_atoms"]) == 1
    if plusminus and not diagonal:
        disps = [
            [0.03, 0.0, 0.0],
            [-0.03, 0.0, 0.0],
            [0.0, 0.0, 0.03],
            [0.0, 0.0, -0.03],
        ]
        assert len(phe.dataset["first_atoms"]) == 4
    if not plusminus and not diagonal:
        disps = [[0.03, 0.0, 0.0], [0.0, 0.0, 0.03]]
        assert len(phe.dataset["first_atoms"]) == 2

    for i, d in enumerate(phe.dataset["first_atoms"]):
        assert d["number"] == 0
        np.testing.assert_allclose(d["displacement"], disps[i])
