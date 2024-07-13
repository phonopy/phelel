"""Tests CLIs."""

import pathlib

import phelel
import pytest
from velph.cli.supercell.differentiate import run_derivatives

cwd = pathlib.Path(__file__).parent
cwd_called = pathlib.Path.cwd()


def test_run_derivatives():
    """Test of run_derivatives.

    This test just checks the creation of hdf5 file and go through this command.

    """
    phe = phelel.load(cwd / "C111" / "phelel_disp_C111.yaml", fft_mesh=[9, 9, 9])
    hdf5_filename = "phelel_params_test_run_derivatives.hdf5"
    run_derivatives(phe, hdf5_filename=hdf5_filename, dir_name=cwd / "C111/supercell")
    file_path = pathlib.Path(cwd_called / hdf5_filename)
    assert file_path.exists()
    file_path.unlink()


def test_run_derivatives_with_wrong_supercell_matrix():
    """Test of run_derivatives.

    This test just checks the creation of hdf5 file and go through this command.
    Supercell matrix is inconsistent. Therefore it should raise an error.

    """
    phe = phelel.load(cwd / "phelel_disp_C222.yaml", fft_mesh=[9, 9, 9])
    with pytest.raises(ValueError):
        run_derivatives(phe, dir_name=cwd / "C111/supercell")


def test_run_derivatives_with_wrong_phonon_supercell_matrix():
    """Test of run_derivatives.

    This test just checks the creation of hdf5 file and go through this command.
    Phonon supercell matrix is inconsistent. This results in not making the hdf5
    file.

    """
    phe = phelel.load(cwd / "phelel_disp_C111-222.yaml", fft_mesh=[9, 9, 9])
    hdf5_filename = "phelel_params_test_run_derivatives.hdf5"
    run_derivatives(phe, hdf5_filename=hdf5_filename, dir_name=cwd / "C111/supercell")

    file_path = pathlib.Path(cwd_called / hdf5_filename)
    assert not file_path.exists()
