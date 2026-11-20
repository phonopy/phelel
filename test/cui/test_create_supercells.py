"""Tests of the utilities to create supercells."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from phono3py.interface.calculator import get_default_displacement_distance

from phelel import Phelel
from phelel.cui.create_supercells import generate_phelel_supercells
from phelel.interface.phelel_yaml import PhelelYaml

cwd = pathlib.Path(__file__).parent


def test_generate_phelel_supercells_default_distance():
    """Without a distance, both kinds of supercells use the default distance.

    The default distance of the VASP interface is 0.03 angstrom, while that of
    Phonopy.generate_displacements is 0.01 angstrom, so the phonon supercells
    must be given the default distance explicitly.

    """
    filename = cwd / ".." / "interface" / "vasp" / "phelel_disp_C111.yaml"
    phe_yml = PhelelYaml().read(filename)
    phelel = Phelel(
        phe_yml.unitcell,
        supercell_matrix=phe_yml.supercell_matrix,
        primitive_matrix=phe_yml.primitive_matrix,
        phonon_supercell_matrix=np.diag([2, 2, 2]),
    )
    generate_phelel_supercells(phelel, interface_mode="vasp")

    distance = get_default_displacement_distance("vasp")
    assert phelel.dataset is not None
    assert phelel.phonon_dataset is not None
    for dataset in (phelel.dataset, phelel.phonon_dataset):
        for disp in dataset["first_atoms"]:
            assert np.linalg.norm(disp["displacement"]) == pytest.approx(distance)


def _get_phelel_C111(is_symmetry: bool = True) -> Phelel:
    """Return Phelel of the unit cell of phelel_disp_C111.yaml."""
    filename = cwd / ".." / "interface" / "vasp" / "phelel_disp_C111.yaml"
    phe_yml = PhelelYaml().read(filename)
    return Phelel(
        phe_yml.unitcell,
        supercell_matrix=phe_yml.supercell_matrix,
        primitive_matrix=phe_yml.primitive_matrix,
        is_symmetry=is_symmetry,
    )


def test_generate_phelel_supercells_random_displacements(
    capsys: pytest.CaptureFixture,
):
    """With number_of_snapshots, all atoms are displaced in each supercell.

    The same random seed gives the same displacements, and the summary of the
    eigenvalues of A(q) is printed.

    """
    datasets = []
    for _ in range(2):
        phelel = _get_phelel_C111()
        generate_phelel_supercells(
            phelel, number_of_snapshots=4, random_seed=1, log_level=1
        )
        datasets.append(phelel.dataset)
    displacements = datasets[0]["displacements"]
    assert displacements.shape == (4, len(phelel.supercell), 3)
    assert (np.linalg.norm(displacements, axis=2) > 0).all()
    np.testing.assert_array_equal(datasets[1]["displacements"], displacements)
    output = capsys.readouterr().out
    assert "Eigenvalues of U(q)^H U(q) at" in output
    assert "Warning" not in output


def test_generate_phelel_supercells_random_plusminus():
    """With is_plusminus=True, the second half has the opposite displacements."""
    phelel = _get_phelel_C111()
    generate_phelel_supercells(
        phelel, is_plusminus=True, number_of_snapshots=3, random_seed=1
    )
    assert phelel.dataset is not None
    displacements = phelel.dataset["displacements"]
    assert len(displacements) == 6
    np.testing.assert_allclose(displacements[3:], -displacements[:3])


def test_generate_phelel_supercells_warns_singular_normal_matrices(
    capsys: pytest.CaptureFixture,
):
    """One snapshot without symmetry leaves A(q) singular, which is reported."""
    phelel = _get_phelel_C111(is_symmetry=False)
    generate_phelel_supercells(
        phelel, number_of_snapshots=1, random_seed=1, log_level=1
    )
    assert "Warning: U(q)^H U(q) is singular" in capsys.readouterr().out
