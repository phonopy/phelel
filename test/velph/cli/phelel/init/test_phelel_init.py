"""Tests CLIs."""

import itertools
import pathlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import tomli

import phelel
from phelel.cui.phelel_script import finalize_phelel
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
elph_prepare = true
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


def _get_phelel_displacements(
    displacement_lines: list[str], tmp_path: pathlib.Path
) -> np.ndarray:
    toml_str = "\n".join(
        ["[phelel]", "supercell_dimension = [2, 2, 1]", *displacement_lines]
    )
    phe = run_init(tomli.loads(toml_str + _TI_UNITCELL), current_directory=tmp_path)
    assert phe.dataset is not None
    return np.array([a["displacement"] for a in phe.dataset["first_atoms"]])


def test_phelel_init_default_displacement_settings(tmp_path: pathlib.Path):
    """Test defaults of plusminus, diagonal, and amplitude in [phelel].

    When they are not given, the values written by velph init (plusminus=true,
    diagonal=false, amplitude=0.03) are used. These give displacements that
    differ from those with the defaults of phonopy ("auto", true, and 0.01).

    """
    disps = _get_phelel_displacements([], tmp_path)
    disps_velph = _get_phelel_displacements(
        ["plusminus = true", "diagonal = false", "amplitude = 0.03"], tmp_path
    )
    disps_phonopy = _get_phelel_displacements(
        ['plusminus = "auto"', "diagonal = true", "amplitude = 0.01"], tmp_path
    )
    assert disps.shape == disps_velph.shape
    np.testing.assert_allclose(disps, disps_velph)
    assert disps.shape != disps_phonopy.shape or not np.allclose(disps, disps_phonopy)


@pytest.mark.parametrize("split_site_mixture", [False, True])
def test_phelel_init_site_mixture(
    site_mixture_velph_toml: Callable[[bool], dict],
    tmp_path: pathlib.Path,
    split_site_mixture: bool,
):
    """Test that phelel_disp.yaml of a site-mixture cell is read back."""
    phe = run_init(
        site_mixture_velph_toml(split_site_mixture), current_directory=tmp_path
    )
    filename = tmp_path / "phelel_disp.yaml"
    finalize_phelel(
        phe, displacements_mode=True, filename=filename, sys_exit_after_finalize=False
    )
    phe2 = phelel.load(filename, log_level=0)
    if split_site_mixture:
        assert phe2.unitcell.symbols == ["Ge", "Sn", "Te"]
        assert phe2.unitcell.has_weighted_species
        assert not phe2.unitcell.has_mixtures
    else:
        assert phe2.unitcell.symbols == ["GeSn", "Te"]
        assert phe2.unitcell.has_mixtures
        assert not phe2.unitcell.has_weighted_species
    assert phe2.supercells_with_displacements is not None
    assert len(phe2.supercells_with_displacements) > 0
