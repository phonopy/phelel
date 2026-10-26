"""Tests for file_IO functions."""

import io
import pathlib

import h5py
import numpy as np
import phonopy
import pytest
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture, get_primitive

import phelel
from phelel.base.Dij_qij import DDijQij
from phelel.base.local_potential import DLocalPotential, get_grid_points
from phelel.file_IO import read_phelel_params_hdf5, write_phelel_params_hdf5

cwd = pathlib.Path(__file__).parent
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


def test_read_phelel_params_hdf5_without_force_constants(tmp_path):
    """A file without force constants is read, and None is returned for them.

    write_phelel_params_hdf5 does not write force constants when they are None.
    Such a file is made by removing them from a file written by save_hdf5.

    """
    dirname = cwd / "interface" / "vasp"
    phe = phelel.load(
        dirname / "phelel_disp_C111.yaml",
        dir_names=[dirname / "C111_disp-000", dirname / "C111_disp-001"],
        fft_mesh=[1, 1, 1],
        is_nac=False,
    )
    filename = tmp_path / "phelel_params.hdf5"
    phe.save_hdf5(filename=filename)
    with h5py.File(filename, "r+") as f:
        del f["force_constants"]

    dVdu, _, fft_mesh, fc = read_phelel_params_hdf5(filename)
    assert fc is None
    np.testing.assert_array_equal(fft_mesh, [1, 1, 1])
    assert dVdu.dVdu is not None


@pytest.mark.parametrize("is_equal", [True, False])
def test_read_phelel_params_hdf5_with_atom_indices_in_derivatives(
    tmp_path, is_equal: bool
):
    """Files with atom_indices_in_derivatives are read only if it is p2s_map.

    Files of older versions have this key, and save_hdf5 no longer writes it.
    The derivatives are those of the atoms of p2s_map.

    """
    dirname = cwd / "interface" / "vasp"
    phe = phelel.load(
        dirname / "phelel_disp_C111.yaml",
        dir_names=[dirname / "C111_disp-000", dirname / "C111_disp-001"],
        fft_mesh=[1, 1, 1],
        is_nac=False,
    )
    filename = tmp_path / "phelel_params.hdf5"
    phe.save_hdf5(filename=filename)
    p2s_map = phe.primitive.p2s_map
    assert len(p2s_map) > 1
    with h5py.File(filename, "r+") as f:
        assert "atom_indices_in_derivatives" not in f
        atom_indices = p2s_map if is_equal else p2s_map[::-1]
        f.create_dataset("atom_indices_in_derivatives", data=atom_indices)

    if is_equal:
        dVdu, dDijdu, _, _ = read_phelel_params_hdf5(filename)
        np.testing.assert_array_equal(dVdu.atom_indices, p2s_map)
        np.testing.assert_array_equal(dDijdu.atom_indices, p2s_map)
    else:
        with pytest.raises(ValueError, match="differs from"):
            read_phelel_params_hdf5(filename)


def _load_C111(**kwargs) -> phelel.Phelel:
    dirname = cwd / "interface" / "vasp"
    return phelel.load(
        dirname / "phelel_disp_C111.yaml",
        dir_names=[dirname / "C111_disp-000", dirname / "C111_disp-001"],
        fft_mesh=[4, 4, 4],
        is_nac=False,
        **kwargs,
    )


_DATASETS = ("dVdu", "dDijdu", "dqijdu", "Dij", "qij")


def test_phelel_params_hdf5_saved_again(tmp_path, monkeypatch):
    """phelel_params.hdf5 read by load and saved again has the same arrays.

    load reads phelel_params.hdf5 in the current directory.  The arrays have to
    be read as complex numbers, otherwise they are written with another axis.

    """
    monkeypatch.chdir(tmp_path)
    _load_C111().save_hdf5(filename="phelel_params.hdf5")
    phe = phelel.load(cwd / "interface" / "vasp" / "phelel_disp_C111.yaml")
    phe.save_hdf5(filename="saved_again.hdf5")
    with (
        h5py.File("phelel_params.hdf5") as f,
        h5py.File("saved_again.hdf5") as g,
    ):
        for key in _DATASETS:
            np.testing.assert_array_equal(g[key][:], f[key][:], err_msg=key)


def test_load_with_phelel_params_hdf5(tmp_path, monkeypatch, capsys):
    """The derivatives are computed again with phelel_params.hdf5 present.

    load reads phelel_params.hdf5 in the current directory before computing
    the derivatives from dir_names.  The results are those computed without
    the file, and with log_level=0 nothing is printed.

    """
    monkeypatch.chdir(tmp_path)
    phe_ref = _load_C111()
    phe_ref.save_hdf5(filename="phelel_params.hdf5")
    capsys.readouterr()
    phe = _load_C111()
    assert capsys.readouterr().out == ""
    np.testing.assert_array_equal(phe.dVdu.dVdu, phe_ref.dVdu.dVdu)
    for key in ("dDijdu", "dqijdu", "Dij", "qij"):
        np.testing.assert_array_equal(
            getattr(phe.dDijdu, key), getattr(phe_ref.dDijdu, key), err_msg=key
        )


def test_read_phelel_params_hdf5_of_supercell_larger_than_unit_cell(tmp_path):
    """phelel_params.hdf5 of a 2x2x2 supercell is read back as it was saved.

    "primitive_matrix" in the file is relative to the unit cell, and the matrix
    from the primitive cell to the supercell has to be made from the lattices.
    The two differ only when the supercell is larger than the unit cell.

    """
    phe = phelel.load(
        cwd / "velph" / "cli" / "phelel" / "differentiate" / "phelel_disp_C222.yaml",
        fft_mesh=[2, 2, 2],
    )
    assert not np.allclose(phe.primitive_matrix, phe.primitive.primitive_matrix)
    rng = np.random.default_rng(0)
    atom_indices = phe.primitive.p2s_map
    n_atoms, natom = len(atom_indices), len(phe.supercell)
    grid_points, lattice_points = get_grid_points(phe.fft_mesh, phe.p2s_matrix)

    def _random(shape):
        return rng.random(shape) + 1j * rng.random(shape)

    phe.dVdu = DLocalPotential.from_arrays(
        phe.fft_mesh,
        phe.p2s_matrix,
        phe.supercell,
        _random((1, n_atoms, 3, len(grid_points))),
        grid_points,
        lattice_points,
        atom_indices=atom_indices,
    )
    phe.dDijdu = DDijQij.from_arrays(
        phe.supercell,
        _random((1, n_atoms, 3, natom, 2, 2)),
        _random((1, n_atoms, 3, natom, 2, 2)),
        _random((1, n_atoms, 2, 2)),
        _random((1, n_atoms, 2, 2)),
        atom_indices=atom_indices,
    )
    filename = tmp_path / "phelel_params.hdf5"
    phe.save_hdf5(filename=filename)

    dVdu, dDijdu, _, _ = read_phelel_params_hdf5(filename)
    np.testing.assert_array_equal(dVdu.p2s_matrix, phe.p2s_matrix)
    np.testing.assert_array_equal(dVdu.dVdu, phe.dVdu.dVdu)
    np.testing.assert_array_equal(dVdu.grid_points, phe.dVdu.grid_points)
    np.testing.assert_array_equal(dVdu.lattice_points, phe.dVdu.lattice_points)
    np.testing.assert_array_equal(dDijdu.dDijdu, phe.dDijdu.dDijdu)
