"""Pytest conftest."""

from __future__ import annotations

import pathlib

import pytest

import phelel
from phelel import Phelel
from phelel.api_phelel import PhelelDataset
from phelel.interface.vasp.file_IO import (
    read_inwap_yaml,
    read_local_potential,
    read_PAW_Dij_qij,
)

cwd = pathlib.Path(__file__).parent


@pytest.fixture(scope="session")
def phelel_empty_C111() -> Phelel:
    """Return phelel_disp.yaml instance of C111."""
    return phelel.load(cwd / "phelel_disp_C111.yaml")


@pytest.fixture(scope="session")
def phelel_empty_NaCl111() -> Phelel:
    """Return phelel_disp.yaml instance of NaCl111."""
    return phelel.load(cwd / "phelel_disp_NaCl111.yaml")


@pytest.fixture(scope="session")
def phelel_empty_CdAs2_111() -> Phelel:
    """Return phelel_disp.yaml instance of CdAs2-111."""
    return phelel.load(cwd / "phelel_disp_CdAs2.yaml")


@pytest.fixture(scope="session")
def phelel_input_filenames_C111() -> list:
    """Return diamond phelel input filenames."""
    inwap_filename = cwd / "inwap_C111.yaml"
    inwap_per = read_inwap_yaml(inwap_filename)

    locpot_filenames = []
    locpot_filenames.append(cwd / "LOCAL-POTENTIAL_C111_perfect.bin.xz")
    locpot_filenames.append(cwd / "LOCAL-POTENTIAL_C111_disp001.bin.xz")

    Dij_filenames = []
    Dij_filenames.append(cwd / "PAW-STRENGTH_C111_perfect.bin")
    Dij_filenames.append(cwd / "PAW-STRENGTH_C111_disp001.bin")

    qij_filenames = []
    qij_filenames.append(cwd / "PAW-OVERLAP_C111_perfect.bin")
    qij_filenames.append(cwd / "PAW-OVERLAP_C111_disp001.bin")

    return [inwap_per, locpot_filenames, Dij_filenames, qij_filenames]


@pytest.fixture(scope="session")
def phelel_input_C111(phelel_input_filenames_C111) -> PhelelDataset:
    """Return diamond phelel input."""
    inwap_per, locpot_filenames, Dij_filenames, qij_filenames = (
        phelel_input_filenames_C111
    )

    return _get_phelel_input(inwap_per, locpot_filenames, Dij_filenames, qij_filenames)


@pytest.fixture(scope="session")
def phelel_input_NaCl111() -> PhelelDataset:
    """Return NaCl phelel input."""
    inwap_filename = cwd / "inwap_NaCl111.yaml"
    inwap_per = read_inwap_yaml(inwap_filename)

    locpot_filenames = []
    locpot_filenames.append(cwd / "LOCAL-POTENTIAL_NaCl111_perfect.bin.xz")
    locpot_filenames.append(cwd / "LOCAL-POTENTIAL_NaCl111_disp001.bin.xz")
    locpot_filenames.append(cwd / "LOCAL-POTENTIAL_NaCl111_disp002.bin.xz")

    Dij_filenames = []
    Dij_filenames.append(cwd / "PAW-STRENGTH_NaCl111_perfect.bin")
    Dij_filenames.append(cwd / "PAW-STRENGTH_NaCl111_disp001.bin")
    Dij_filenames.append(cwd / "PAW-STRENGTH_NaCl111_disp002.bin")

    qij_filenames = []
    qij_filenames.append(cwd / "PAW-OVERLAP_NaCl111_perfect.bin")
    qij_filenames.append(cwd / "PAW-OVERLAP_NaCl111_disp001.bin")
    qij_filenames.append(cwd / "PAW-OVERLAP_NaCl111_disp002.bin")

    return _get_phelel_input(inwap_per, locpot_filenames, Dij_filenames, qij_filenames)


@pytest.fixture(scope="session")
def phelel_input_CdAs2_111() -> PhelelDataset:
    """Return CdAs2 phelel input."""
    inwap_filename = cwd / "inwap_CdAs2_111.yaml"
    inwap_per = read_inwap_yaml(inwap_filename)

    locpot_filenames = []
    Dij_filenames = []
    qij_filenames = []
    locpot_filenames.append(cwd / "LOCAL-POTENTIAL_CdAs2_111_perfect.bin.xz")
    Dij_filenames.append(cwd / "PAW-STRENGTH_CdAs2_111_perfect.bin")
    qij_filenames.append(cwd / "PAW-OVERLAP_CdAs2_111_perfect.bin")
    for i in range(1, 6):
        locpot_filenames.append(cwd / f"LOCAL-POTENTIAL_CdAs2_111_disp{i:03d}.bin.xz")
        Dij_filenames.append(cwd / f"PAW-STRENGTH_CdAs2_111_disp{i:03d}.bin")
        qij_filenames.append(cwd / f"PAW-OVERLAP_CdAs2_111_disp{i:03d}.bin")

    return _get_phelel_input(inwap_per, locpot_filenames, Dij_filenames, qij_filenames)


@pytest.fixture(scope="session")
def phelel_C111(phelel_empty_C111: Phelel, phelel_input_C111: PhelelDataset) -> Phelel:
    """Run diamond test."""
    return _get_phelel(phelel_empty_C111, phelel_input_C111)


@pytest.fixture(scope="session")
def phelel_NaCl111(
    phelel_empty_NaCl111: Phelel, phelel_input_NaCl111: PhelelDataset
) -> Phelel:
    """Run NaCl test."""
    return _get_phelel(phelel_empty_NaCl111, phelel_input_NaCl111)


@pytest.fixture(scope="session")
def phelel_CdAs2_111(
    phelel_empty_CdAs2_111: Phelel, phelel_input_CdAs2_111: PhelelDataset
) -> Phelel:
    """Run CdAs2 test."""
    return _get_phelel(phelel_empty_CdAs2_111, phelel_input_CdAs2_111)


def _get_phelel(phe: Phelel, phe_input: PhelelDataset):
    phei = phe_input
    phe.fft_mesh = [14, 14, 14]
    phe.run_derivatives(phei)
    return phe


def _get_phelel_input(inwap_per, locpot_filenames, Dij_filenames, qij_filenames):
    loc_pots = []
    for filename in locpot_filenames:
        loc_pots.append(read_local_potential(inwap_per, filename))

    Dijs = []
    for filename in Dij_filenames:
        Dijs.append(read_PAW_Dij_qij(inwap_per, filename))

    qijs = []
    for filename in qij_filenames:
        qijs.append(read_PAW_Dij_qij(inwap_per, filename))

    return PhelelDataset(
        local_potentials=loc_pots,
        Dijs=Dijs,
        qijs=qijs,
        lm_channels=inwap_per["lm_orbitals"],
    )
