"""Test for Phelel class."""

import pathlib

import h5py
import numpy as np
import pytest
from phonopy.file_IO import write_FORCE_SETS
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture

import phelel
from phelel import Phelel
from phelel.file_IO import _get_smallest_vectors, read_phelel_params_hdf5
from phelel.utils.data import cmplx2real

cwd = pathlib.Path(__file__).parent


def test_api_phelel_C111(phelel_C111: Phelel):
    """Test by diamond conv. unit cell 1x1x1."""
    filename = cwd / "phelel_params_C111.hdf5"
    _compare(filename, phelel_C111)


def test_api_phelel_NaCl111(phelel_NaCl111: Phelel):
    """Test by NaCl conv. unit cell 1x1x1."""
    filename = cwd / "phelel_params_NaCl111.hdf5"
    _compare(filename, phelel_NaCl111)


def test_api_phelel_CdAs2_111(phelel_CdAs2_111: Phelel):
    """Test by CdAs2 conv. unit cell 1x1x1 (I-centred tetragonal)."""
    filename = cwd / "phelel_params_CdAs2_111.hdf5"
    _compare(filename, phelel_CdAs2_111)


def test_read_phelel_params_hdf5(phelel_CdAs2_111: Phelel):
    """Test reading phelel_params using CdAs2."""
    filename = cwd / "phelel_params_CdAs2_111.hdf5"
    dVdu, dDijdu, _, _ = read_phelel_params_hdf5(filename=filename)
    phe_ref = phelel_CdAs2_111

    np.testing.assert_allclose(dVdu.dVdu, phe_ref.dVdu.dVdu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        dDijdu.dDijdu, cmplx2real(phe_ref.dDijdu.dDijdu), rtol=1e-5, atol=1e-5
    )


def _compare(filename: pathlib.Path, phe: Phelel):
    """Assert results.

    shortest_vectors and shortest_vector_multiplicities are included at later
    versions of Phelel. So, these are not included in old reference files.

    """
    with h5py.File(filename, "r") as f:
        dVdu_ref = f["dVdu"][:]
        dDijdu_ref = f["dDijdu"][:]

        dVdu = cmplx2real(phe.dVdu.dVdu)
        dDijdu = cmplx2real(phe.dDijdu.dDijdu)
        np.testing.assert_allclose(dVdu, dVdu_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(dDijdu, dDijdu_ref, rtol=1e-4, atol=1e-4)

        if "shortest_vectors" in f:
            shortest_vectors_ref = f["shortest_vectors"][:]
            multiplicities_ref = f["shortest_vector_multiplicities"][:]

            shortest_vectors, multiplicities = _get_smallest_vectors(phe.primitive)
            np.testing.assert_array_equal(
                shortest_vectors.shape, shortest_vectors_ref.shape
            )
            np.testing.assert_array_equal(multiplicities, multiplicities_ref)


def _get_GeSnTe_weighted_cell() -> PhonopyAtoms:
    """Return CsCl-like cell with Ge and Sn (0.5 each) co-located and Te."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.0,
        symbols=["Ge", "Sn", "Te"],
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
    )
    return apply_site_mixture(cell, [0.5, 0.5, 1.0])


def test_phelel_merge_scheme_yaml(tmp_path: pathlib.Path):
    """With the merge scheme, phelel_disp.yaml has the unmerged cells."""
    phe = Phelel(
        _get_GeSnTe_weighted_cell(),
        supercell_matrix=np.diag([2, 2, 2]),
        phonon_supercell_matrix=np.diag([2, 2, 2]),
        primitive_matrix="P",
    )
    assert phe.site_mixture_scheme == "merge"
    assert phe.unitcell.symbols == ["GeSn", "Te"]
    assert phe.unmerged_unitcell is not None
    assert phe.unmerged_unitcell.symbols == ["Ge", "Sn", "Te"]
    assert phe.phonon_supercell is not None
    assert phe.phonon_unmerged_supercell is not None
    assert len(phe.phonon_unmerged_supercell) == 24
    assert len(phe.phonon_supercell) == 16
    phe.generate_displacements()
    phe.generate_phonon_displacements()

    filename = tmp_path / "phelel_disp.yaml"
    text = str(phe.to_phelel_yaml())
    assert "site_mixture_scheme: merge" in text
    assert "mixture:" not in text
    filename.write_text(text)
    phe2 = phelel.load(filename, log_level=0)
    assert phe2.site_mixture_scheme == "merge"
    assert phe2.unitcell.symbols == ["GeSn", "Te"]
    assert phe2.phonon_dataset is not None
    assert phe2.phonon_dataset["natom"] == 16
    with pytest.raises(RuntimeError, match="merge scheme of site mixture"):
        phe2.run_derivatives(None)  # type: ignore[arg-type]


def test_phelel_load_force_sets_of_phonon_supercell(tmp_path: pathlib.Path):
    """FORCE_SETS is read for the phonon supercell."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.0,
        symbols=["Na", "Cl"],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    force_sets = tmp_path / "FORCE_SETS"
    write_FORCE_SETS(
        {
            "natom": 16,
            "first_atoms": [
                {
                    "number": 0,
                    "displacement": np.array([0.01, 0, 0]),
                    "forces": np.zeros((16, 3)),
                }
            ],
        },
        filename=force_sets,
    )
    phe = phelel.load(
        unitcell=cell,
        supercell_matrix=np.eye(3, dtype=int),
        phonon_supercell_matrix=np.diag([2, 2, 2]),
        primitive_matrix="P",
        force_sets_filename=force_sets,
    )
    assert phe.phonon_dataset is not None
    assert phe.phonon_dataset["natom"] == 16
