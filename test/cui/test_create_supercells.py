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
