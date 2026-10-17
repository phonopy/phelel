"""Tests velph-phono3py-init."""

import pathlib
import tomllib
from collections.abc import Callable, Sequence

import click
import numpy as np
import phono3py
import pytest
from phonopy import Phonopy

from phelel.velph.cli.phono3py.init import run_init
from phelel.velph.config.io import parse_velph_toml
from phelel.velph.config.schema import VelphConfig


@pytest.mark.parametrize("distance", [0.03, 0.05])
def test_phono3py_init_random_displacements(
    distance: float, code_init_config: Callable[[dict], VelphConfig]
):
    """Test of plusminus and diagonal with Ti."""
    phelel_str = f"""[phelel]
supercell_dimension = [4, 4, 2]
amplitude = {distance}
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
        tomllib.loads(phono3py_str + unitcell_str),
        tomllib.loads(phelel_str + phono3py_str + unitcell_str),
    ):
        ph3 = run_init(code_init_config(toml_dict), number_of_snapshots=10)
        assert ph3 is not None
        np.testing.assert_array_equal(ph3.supercell_matrix, np.diag([2, 2, 1]))

        assert len(ph3.supercell) == 8
        assert ph3.displacements.shape == (10, 8, 3)
        np.testing.assert_allclose(np.linalg.norm(ph3.displacements, axis=2), distance)


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


def _get_phono3py_displacements(
    displacement_lines: list[str],
    tmp_path: pathlib.Path,
    code_init_config: Callable[[dict], VelphConfig],
) -> np.ndarray:
    toml_str = "\n".join(
        ["[phono3py]", "supercell_dimension = [2, 2, 1]", *displacement_lines]
    )
    ph3 = run_init(
        code_init_config(tomllib.loads(toml_str + _TI_UNITCELL)),
        current_directory=tmp_path,
    )
    return np.array(ph3.displacements)


def test_phono3py_init_default_displacement_settings(
    tmp_path: pathlib.Path, code_init_config: Callable[[dict], VelphConfig]
):
    """Test defaults of plusminus, diagonal, and amplitude in [phono3py].

    When they are not given, the values written by velph init (plusminus=true,
    diagonal=false, amplitude=0.03) are used. These give displacements that
    differ from those with the defaults of phonopy ("auto", true, and 0.01).

    """
    disps = _get_phono3py_displacements([], tmp_path, code_init_config)
    disps_velph = _get_phono3py_displacements(
        ["plusminus = true", "diagonal = false", "amplitude = 0.03"],
        tmp_path,
        code_init_config,
    )
    disps_phonopy = _get_phono3py_displacements(
        ['plusminus = "auto"', "diagonal = true", "amplitude = 0.01"],
        tmp_path,
        code_init_config,
    )
    assert disps.shape == disps_velph.shape
    np.testing.assert_allclose(disps, disps_velph)
    assert disps.shape != disps_phonopy.shape or not np.allclose(disps, disps_phonopy)


@pytest.mark.parametrize("split_site_mixture", [False, True])
def test_phono3py_init_site_mixture_is_error(
    site_mixture_velph_toml: Callable[[bool], dict],
    tmp_path: pathlib.Path,
    split_site_mixture: bool,
):
    """Test that a site-mixture cell is an error for phono3py.

    phono3py handles neither merged mixed-species sites nor weighted species.

    """
    with pytest.raises(click.ClickException, match="site-mixture"):
        run_init(
            parse_velph_toml(site_mixture_velph_toml(split_site_mixture)),
            current_directory=tmp_path,
        )


def test_phono3py_init_number_of_snapshots_of_section(
    code_init_config: Callable[[dict], VelphConfig],
):
    """Without --rd, number_of_snapshots of [phono3py] gives random displacements."""
    toml_str = "\n".join(
        ["[phono3py]", "supercell_dimension = [2, 2, 1]", "number_of_snapshots = 7"]
    )
    config = code_init_config(tomllib.loads(toml_str + _TI_UNITCELL))
    assert run_init(config).displacements.shape == (7, 8, 3)
    assert run_init(config, number_of_snapshots=5).displacements.shape == (5, 8, 3)


@pytest.mark.parametrize(
    "phonopy_lines", [["amplitude = 0.01"], ["number_of_snapshots = 3"]]
)
def test_phono3py_init_phonon_from_phonopy(
    phonopy_lines: list[str],
    ti_velph_toml: Callable[[Sequence[str]], dict],
    code_init_config: Callable[[dict], VelphConfig],
    write_phonopy_disp_yaml: Callable[[VelphConfig, pathlib.Path], Phonopy],
    tmp_path: pathlib.Path,
):
    """The fc2 supercells are those of phonopy/phonopy_disp.yaml.

    They are kept through phono3py_disp.yaml, also with random displacements of
    fc3.  The amplitude of [phonopy] differs from that of [phono3py], with which
    phono3py would generate other fc2 supercells.  With number_of_snapshots,
    the displacements of fc2 are random.

    """
    config = code_init_config(ti_velph_toml(phonopy_lines))
    ph = write_phonopy_disp_yaml(config, tmp_path)
    ph3 = run_init(
        config,
        current_directory=tmp_path,
        number_of_snapshots=4,
        phonon_from_phonopy=True,
    )
    yaml_filename = tmp_path / "phono3py_disp.yaml"
    ph3.save(yaml_filename)
    ph3_loaded = phono3py.load(yaml_filename, produce_fc=False)
    assert ph.supercells_with_displacements is not None
    for ph3_cells in (
        ph3.phonon_supercells_with_displacements,
        ph3_loaded.phonon_supercells_with_displacements,
    ):
        assert ph3_cells is not None
        assert len(ph3_cells) == len(ph.supercells_with_displacements)
        for cell, ref_cell in zip(
            ph3_cells, ph.supercells_with_displacements, strict=True
        ):
            np.testing.assert_allclose(cell.cell, ref_cell.cell)
            np.testing.assert_allclose(cell.positions, ref_cell.positions)
    np.testing.assert_array_equal(
        ph3_loaded.phonon_supercell_matrix, ph.supercell_matrix
    )
    assert ph3_loaded.displacements.shape == (4, 2, 3)


def test_phono3py_init_phonon_from_phonopy_without_phonopy_disp_yaml(
    ti_velph_toml: Callable[[Sequence[str]], dict],
    code_init_config: Callable[[dict], VelphConfig],
    tmp_path: pathlib.Path,
):
    """Without phonopy/phonopy_disp.yaml, velph phonopy init is asked for."""
    config = code_init_config(ti_velph_toml([]))
    with pytest.raises(click.ClickException, match='"velph phonopy init" first'):
        run_init(config, current_directory=tmp_path, phonon_from_phonopy=True)
