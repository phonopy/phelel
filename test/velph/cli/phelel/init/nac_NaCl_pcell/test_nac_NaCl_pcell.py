"""Tests CLIs."""

import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np

from phelel.velph.cli.phelel.init import run_init
from phelel.velph.jobs.project import Project
from phelel.velph.jobs.project_directory import ProjectDirectory

cwd = Path(__file__).parent


def test_nac_NaCl_pcell(request, code_init_project: Callable[[dict], Project]):
    """Test of different NAC cell."""
    toml_str = """[phelel]
supercell_dimension = [2, 2, 2]
amplitude = 0.03
plusminus = true

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

    toml_dict = tomllib.loads(toml_str)
    phe = run_init(
        code_init_project(toml_dict), project_dir=ProjectDirectory(request.path.parent)
    )
    ref_born = 1.54116678
    ref_dielectric = 3.24483954
    np.testing.assert_array_almost_equal(
        phe.nac_params["born"][0], np.eye(3) * ref_born
    )
    np.testing.assert_array_almost_equal(
        phe.nac_params["dielectric"], np.eye(3) * ref_dielectric
    )
