"""Test for Phelel class."""

import pathlib

import h5py
import numpy as np

from phelel import Phelel
from phelel.file_IO import read_phelel_params_hdf5
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
        np.testing.assert_allclose(dVdu, dVdu_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(dDijdu, dDijdu_ref, rtol=1e-5, atol=1e-5)

        if "shortest_vectors" in f:
            shortest_vectors_ref = f["shortest_vectors"][:]
            multiplicities_ref = f["shortest_vector_multiplicities"][:]

            shortest_vectors, multiplicities = phe.primitive.get_smallest_vectors()
            np.testing.assert_array_equal(
                shortest_vectors.shape, shortest_vectors_ref.shape
            )
            np.testing.assert_array_equal(multiplicities, multiplicities_ref)
