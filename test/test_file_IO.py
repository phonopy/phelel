"""Tests for file_IO functions."""

import io
import pathlib

import h5py
import numpy as np
import phonopy
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture, get_primitive

from phelel.file_IO import write_phelel_params_hdf5

cwd_called = pathlib.Path.cwd()


def _site_mixture_cell():
    """Return a weighted cell: co-located Ge/Sn (0.5/0.5) + pure Te."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.0,
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
        symbols=["Ge", "Sn", "Te"],
    )
    return apply_site_mixture(cell, [0.5, 0.5, 1.0], symprec=1e-5)


def test_write_phelel_params_hdf5_mixture_weights(tmp_path):
    """Test mixture_weights datasets in write_phelel_params_hdf5.

    A weighted cell emits a per-cell ``*_mixture_weights`` dataset for every
    atom-bearing cell, with pure sites materialized to 1.0.

    """
    weighted = _site_mixture_cell()
    primitive = get_primitive(weighted, np.eye(3), symprec=1e-5)
    filename = tmp_path / "phelel_params.hdf5"
    write_phelel_params_hdf5(
        primitive=primitive,
        unitcell=weighted,
        supercell=weighted,
        phonon_supercell=weighted,
        filename=filename,
    )
    expected = [0.5, 0.5, 1.0]
    with h5py.File(filename, "r") as f:
        for key in (
            "primitive_mixture_weights",
            "unitcell_mixture_weights",
            "supercell_mixture_weights",
            "phonon_supercell_mixture_weights",
        ):
            assert key in f
            np.testing.assert_allclose(f[key][:], expected)


def test_write_phelel_params_hdf5_no_mixture_weights(tmp_path):
    """Test that ordinary cells emit no mixture_weights datasets."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.0,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        symbols=["Na", "Cl"],
    )
    primitive = get_primitive(cell, np.eye(3), symprec=1e-5)
    filename = tmp_path / "phelel_params.hdf5"
    write_phelel_params_hdf5(
        primitive=primitive,
        unitcell=cell,
        supercell=cell,
        phonon_supercell=cell,
        filename=filename,
    )
    with h5py.File(filename, "r") as f:
        assert not any("mixture_weights" in key for key in f.keys())


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
