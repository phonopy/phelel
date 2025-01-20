"""Tests CLIs."""

import pathlib

import h5py
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

    phe.save_hdf5()
    for created_filename in ["phelel_params.hdf5"]:
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()

        with h5py.File(created_filename) as f:
            try:
                assert set(f) == set(
                    (
                        "atom_indices_in_derivatives",  # This key is deprecated.
                        "Dij",
                        "FFT_mesh",
                        "dDijdu",
                        "dVdu",
                        "direct_rotations",
                        "displacements_atom_indices",
                        "displacements_vectors",
                        "dqijdu",
                        "force_constants",
                        "grid_point",
                        "lattice_point",
                        "p2s_map",
                        "primitive_lattice",
                        "primitive_masses",
                        "primitive_matrix",
                        "primitive_numbers",
                        "primitive_positions",
                        "qij",
                        "s2p_map",
                        "shortest_vector_multiplicities",
                        "shortest_vectors",
                        "spacegroup_number",
                        "supercell_lattice",
                        "supercell_masses",
                        "supercell_matrix",
                        "supercell_numbers",
                        "supercell_positions",
                        "transformation_matrix",
                        "unitcell_lattice",
                        "unitcell_masses",
                        "unitcell_numbers",
                        "unitcell_positions",
                    )
                )
            except AssertionError as e:
                file_path.unlink()
                raise AssertionError(str(e)) from e
        file_path.unlink()


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
