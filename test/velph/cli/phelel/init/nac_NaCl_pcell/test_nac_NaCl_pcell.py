"""Tests CLIs."""

from pathlib import Path

import numpy as np
import tomli

from phelel.velph.cli.phelel.init import run_init

cwd = Path(__file__).parent


def test_nac_NaCl_pcell(request):
    """Test of different NAC cell."""
    toml_str = """title = "VASP el-ph settings"

[phelel]
supercell_dimension = [2, 2, 2]
amplitude = 0.03
plusminus = true
fft_mesh = [30, 30, 30]

[vasp.supercell.incar]
system = "default"
ismear = 0
sigma = 0.1
prec = "accurate"
ediff = 1e-06
lwap = true
lreal = false
lwave = false
lcharg = false
[vasp.supercell.kpoints]
mesh = [1, 1, 1]

[vasp.nac]
cell = "primitive"
[vasp.nac.incar]
system = "default"
ismear = 0
sigma = 0.1
prec = "accurate"
ediff = 1e-08
lreal = false
lwave = false
lcharg = false
lepsilon = true
[vasp.nac.kpoints]
mesh = [2, 2, 2]

[symmetry]
spacegroup_type = "Fm-3m"
tolerance = 1e-05
primitive_matrix = [
  [  0.000000000000000,  0.500000000000000,  0.500000000000000 ],
  [  0.500000000000000,  0.000000000000000,  0.500000000000000 ],
  [  0.500000000000000,  0.500000000000000,  0.000000000000000 ],
]

[unitcell]
lattice = [
  [     5.690301476175672,     0.000000000000000,     0.000000000000000 ], # a
  [     0.000000000000000,     5.690301476175672,     0.000000000000000 ], # b
  [     0.000000000000000,     0.000000000000000,     5.690301476175672 ], # c
]
[[unitcell.points]]  # 1
symbol = "Na"
coordinates = [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
[[unitcell.points]]  # 2
symbol = "Na"
coordinates = [  0.000000000000000,  0.500000000000000,  0.500000000000000 ]
[[unitcell.points]]  # 3
symbol = "Na"
coordinates = [  0.500000000000000,  0.000000000000000,  0.500000000000000 ]
[[unitcell.points]]  # 4
symbol = "Na"
coordinates = [  0.500000000000000,  0.500000000000000,  0.000000000000000 ]
[[unitcell.points]]  # 5
symbol = "Cl"
coordinates = [  0.500000000000000,  0.000000000000000,  0.000000000000000 ]
[[unitcell.points]]  # 6
symbol = "Cl"
coordinates = [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
[[unitcell.points]]  # 7
symbol = "Cl"
coordinates = [  0.000000000000000,  0.000000000000000,  0.500000000000000 ]
[[unitcell.points]]  # 8
symbol = "Cl"
coordinates = [  0.000000000000000,  0.500000000000000,  0.000000000000000 ]
[primitive_cell]
lattice = [
  [     0.000000000000000,     2.845150738087836,     2.845150738087836 ], # a
  [     2.845150738087836,     0.000000000000000,     2.845150738087836 ], # b
  [     2.845150738087836,     2.845150738087836,     0.000000000000000 ], # c
]
[[primitive_cell.points]]  # 1
symbol = "Na"
coordinates = [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
[[primitive_cell.points]]  # 2
symbol = "Cl"
coordinates = [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
"""

    toml_dict = tomli.loads(toml_str)
    phe = run_init(toml_dict, current_directory=request.path.parent)
    ref_born = 1.54116678
    ref_dielectric = 3.24483954
    np.testing.assert_array_almost_equal(
        phe.nac_params["born"][0], np.eye(3) * ref_born
    )
    np.testing.assert_array_almost_equal(
        phe.nac_params["dielectric"], np.eye(3) * ref_dielectric
    )
