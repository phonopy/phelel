"""Tests for file_IO functions."""

import io
import pathlib

import h5py
import phonopy
from phonopy.interface.phonopy_yaml import read_cell_yaml

from phelel.file_IO import write_phelel_params_hdf5

cwd_called = pathlib.Path.cwd()


def test_write_phelel_params_hdf5_magnetic_spacegroup_uni_number():
    """Test magnetic_spacegroup_uni_number in write_phelel_params_hdf5."""
    unitcell_str = """unit_cell:
  lattice:
  - [     2.812696943681890,     0.000000000000000,     0.000000000000000 ] # a
  - [     0.000000000000000,     2.812696943681890,     0.000000000000000 ] # b
  - [     0.000000000000000,     0.000000000000000,     2.812696943681890 ] # c
  points:
  - symbol: Cr # 1
    coordinates: [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
    mass: 51.996100
    reduced_to: 1
    magnetic_moment: 1.00000000
  - symbol: Cr # 2
    coordinates: [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
    mass: 51.996100
    reduced_to: 1
    magnetic_moment: -1.00000000"""

    cell = read_cell_yaml(io.StringIO(unitcell_str))
    ph = phonopy.Phonopy(cell)
    write_phelel_params_hdf5(symmetry_dataset=ph.primitive_symmetry.dataset)
    for created_filename in ("phelel_params.hdf5",):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        with h5py.File(file_path, "r") as f:
            assert "magnetic_spacegroup_uni_number" in f
            assert "spacegroup_number" not in f
        file_path.unlink()


def test_write_phelel_params_hdf5_spacegroup_number():
    """Test spacegroup_number in write_phelel_params_hdf5."""
    unitcell_str = """unit_cell:
  lattice:
  - [     2.812696943681890,     0.000000000000000,     0.000000000000000 ] # a
  - [     0.000000000000000,     2.812696943681890,     0.000000000000000 ] # b
  - [     0.000000000000000,     0.000000000000000,     2.812696943681890 ] # c
  points:
  - symbol: Cr # 1
    coordinates: [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
    mass: 51.996100
    reduced_to: 1
  - symbol: Cr # 2
    coordinates: [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
    mass: 51.996100
    reduced_to: 1"""

    cell = read_cell_yaml(io.StringIO(unitcell_str))
    ph = phonopy.Phonopy(cell)
    write_phelel_params_hdf5(symmetry_dataset=ph.primitive_symmetry.dataset)
    for created_filename in ("phelel_params.hdf5",):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        with h5py.File(file_path, "r") as f:
            assert "spacegroup_number" in f
            assert "magnetic_spacegroup_uni_number" not in f
        file_path.unlink()
