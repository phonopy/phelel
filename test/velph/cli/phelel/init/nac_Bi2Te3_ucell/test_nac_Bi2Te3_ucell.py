"""Tests CLIs."""

from pathlib import Path

import numpy as np
import tomli

from phelel.velph.cli.phelel.init import run_init

cwd = Path(__file__).parent


def test_nac_Bi2Te3_ucell(request):
    """Test of different NAC cell."""
    toml_str = """title = "VASP el-ph settings"

[phelel]
supercell_dimension = [2, 2, 1]
amplitude = 0.03
plusminus = true
fft_mesh = [80, 80, 80]

[vasp.supercell.incar]
system = "default"
ismear = 0
sigma = 0.01
prec = "accurate"
ediff = 1e-06
encut = 500
lwap = true
lreal = false
lwave = false
lcharg = false
isym = 0
[vasp.supercell.kpoints]
mesh = [1, 1, 1]

[vasp.nac]
cell = "unitcell"
[vasp.nac.incar]
system = "default"
ismear = 0
sigma = 0.01
prec = "accurate"
ediff = 1e-08
encut = 500
lreal = false
lwave = false
lcharg = false
lepsilon = true
[vasp.nac.kpoints]
mesh = [2, 2, 1]

[symmetry]
spacegroup_type = "R-3m"
tolerance = 1e-05
primitive_matrix = [
    [0.500000000000000, -0.288675134594813, -0.000000000000000],
    [0.288675134594813, 0.500000000000000, -0.081594494335131],
    [2.358616121075893, 4.085242957254472, 0.333333333333334],
]

[unitcell]
lattice = [
    [4.443005492108914, -0.000000000000000, -0.000000000000000],  # a
    [-2.221502746054457, 3.847755625320100, -0.000000000000000],  # b
    [-0.000000000000001, -0.000000000000001, 31.438033139150441],  # c
]
[[unitcell.points]]  # 1
symbol = "Bi"
coordinates = [0.666666666666664, 0.333333333333336, 0.934850830403498]
[[unitcell.points]]  # 2
symbol = "Bi"
coordinates = [0.666666666666664, 0.333333333333336, 0.731815836263166]
[[unitcell.points]]  # 3
symbol = "Bi"
coordinates = [0.333333333333336, 0.666666666666664, 0.268184163736834]
[[unitcell.points]]  # 4
symbol = "Bi"
coordinates = [0.333333333333336, 0.666666666666664, 0.065149169596502]
[[unitcell.points]]  # 5
symbol = "Bi"
coordinates = [0.000000000000000, 0.000000000000000, 0.601517497070162]
[[unitcell.points]]  # 6
symbol = "Bi"
coordinates = [0.000000000000000, 0.000000000000000, 0.398482502929838]
[[unitcell.points]]  # 7
symbol = "Te"
coordinates = [0.000000000000000, 0.000000000000000, 0.786619249653128]
[[unitcell.points]]  # 8
symbol = "Te"
coordinates = [0.000000000000000, 0.000000000000000, 0.000000000000000]
[[unitcell.points]]  # 9
symbol = "Te"
coordinates = [0.333333333333336, 0.666666666666664, 0.880047417013543]
[[unitcell.points]]  # 10
symbol = "Te"
coordinates = [0.666666666666664, 0.333333333333336, 0.119952582986457]
[[unitcell.points]]  # 11
symbol = "Te"
coordinates = [0.666666666666664, 0.333333333333336, 0.333333333333336]
[[unitcell.points]]  # 12
symbol = "Te"
coordinates = [0.000000000000000, 0.000000000000000, 0.213380750346872]
[[unitcell.points]]  # 13
symbol = "Te"
coordinates = [0.333333333333336, 0.666666666666664, 0.453285916319793]
[[unitcell.points]]  # 14
symbol = "Te"
coordinates = [0.333333333333336, 0.666666666666664, 0.666666666666664]
[[unitcell.points]]  # 15
symbol = "Te"
coordinates = [0.666666666666664, 0.333333333333336, 0.546714083680207]
[primitive_cell]
lattice = [
    [2.221502746054456, 1.282585208440033, 10.479344379716814],  # a
    [-2.221502746054457, 1.282585208440033, 10.479344379716814],  # b
    [-0.000000000000000, -2.565170416880068, 10.479344379716814],  # c
]
[[primitive_cell.points]]  # 1
symbol = "Bi"
coordinates = [0.601517497070162, 0.601517497070169, 0.601517497070162]
[[primitive_cell.points]]  # 2
symbol = "Bi"
coordinates = [0.398482502929831, 0.398482502929838, 0.398482502929831]
[[primitive_cell.points]]  # 3
symbol = "Te"
coordinates = [0.786619249653129, 0.786619249653129, 0.786619249653128]
[[primitive_cell.points]]  # 4
symbol = "Te"
coordinates = [0.000000000000000, 0.000000000000000, 0.000000000000000]
[[primitive_cell.points]]  # 5
symbol = "Te"
coordinates = [0.213380750346879, 0.213380750346872, 0.213380750346879]
"""

    toml_dict = tomli.loads(toml_str)
    phe = run_init(toml_dict, current_directory=request.path.parent)
    ref_born = [
        [11.5925019, 11.5925019, -1.15102317],
        [11.5925019, 11.5925019, -1.15102317],
        [-7.18142673, -7.18142673, 2.28221277],
        [-8.82215025, -8.82215025, -2.26237921],
        [-7.18142673, -7.18142673, 2.28221277],
    ]
    ref_dielectric = [61.0943836, 61.0943836, 48.1700721]
    np.testing.assert_array_almost_equal(
        phe.nac_params["born"], [np.diag(b) for b in ref_born]
    )
    np.testing.assert_array_almost_equal(
        phe.nac_params["dielectric"], np.diag(ref_dielectric)
    )
