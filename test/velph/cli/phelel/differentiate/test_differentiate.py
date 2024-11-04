"""Tests CLIs."""

import pathlib

import pytest

import phelel
from phelel.velph.cli.phelel.differentiate import run_derivatives

cwd = pathlib.Path(__file__).parent
cwd_called = pathlib.Path.cwd()


def test_run_derivatives():
    """Test of run_derivatives.

    This test just checks the creation of hdf5 file and go through this command.

    """
    phe = phelel.load(cwd / "C111" / "phelel_disp_C111.yaml", fft_mesh=[9, 9, 9])
    assert run_derivatives(phe, dir_name=cwd / "C111" / "phelel")


def test_run_derivatives_with_wrong_supercell_matrix():
    """Test of run_derivatives.

    This test just checks the creation of hdf5 file and go through this command.
    Supercell matrix is inconsistent. Therefore it should raise an error.

    """
    phe = phelel.load(cwd / "phelel_disp_C222.yaml", fft_mesh=[9, 9, 9])
    with pytest.raises(ValueError):
        run_derivatives(phe, dir_name=cwd / "C111" / "phelel")


def test_run_derivatives_with_wrong_phonon_supercell_matrix():
    """Test of run_derivatives.

    Phonon supercell matrix is inconsistent. Therefore it will return False.

    """
    phe = phelel.load(cwd / "phelel_disp_C111-222.yaml", fft_mesh=[9, 9, 9])
    assert not run_derivatives(phe, dir_name=cwd / "C111" / "phelel")