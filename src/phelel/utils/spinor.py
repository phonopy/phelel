"""Routines to handle spinors: rotation and time reversal in spin space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from phelel.utils.spherical_harmonics import get_n_and_rotation_order


class SxSySzMatrices:
    """Create matrix representation of <sigma'|Sx|sigma> and so forth.

    hbar is not multiplied.

    The matrix convention is different from that of Pauli matrices.
    m=-1/2 and 1/2 correspond to 0 and 1 indices of arrays here, respectively.

    """

    def __init__(self):
        """Init method."""
        self._Sx = None
        self._Sy = None
        self._Sz = None

    @property
    def Sx(self) -> np.ndarray | None:
        """Return <sigma'|Sx|sigma> for all sigma channel."""
        return self._Sx

    @property
    def Sy(self) -> np.ndarray | None:
        """Return <sigma'|Sy|sigma> for all sigma channel."""
        return self._Sy

    @property
    def Sz(self) -> np.ndarray | None:
        """Return <sigma'|Sz|sigma> for all sigma channel."""
        return self._Sz

    def run(self):
        """Calculate <sigma'|Sx|sigma>, and so on for all sigma channel."""
        self._Sx = self._get_Sx()
        self._Sy = self._get_Sy()
        self._Sz = self._get_Sz()
        return self

    def _get_Sx(self) -> np.ndarray:
        sx = np.zeros((2, 2), dtype=complex, order="C")
        j_s = 1.0 / 2
        for ip, _ in enumerate((-j_s, j_s)):
            for i, m_s in enumerate((-j_s, j_s)):
                if ip == i + 1:
                    sx[ip, i] += np.sqrt((j_s - m_s) * (j_s + m_s + 1))
                if ip == i - 1:
                    sx[ip, i] += np.sqrt((j_s + m_s) * (j_s - m_s + 1))
        return sx / 2

    def _get_Sy(self) -> np.ndarray:
        sy = np.zeros((2, 2), dtype=complex, order="C")
        j_s = 1.0 / 2
        for ip, _ in enumerate((-j_s, j_s)):
            for i, m_s in enumerate((-j_s, j_s)):
                if ip == i + 1:
                    sy[ip, i] -= np.sqrt((j_s - m_s) * (j_s + m_s + 1))
                if ip == i - 1:
                    sy[ip, i] += np.sqrt((j_s + m_s) * (j_s - m_s + 1))
        return 1j * sy / 2

    def _get_Sz(self) -> np.ndarray:
        sz = np.zeros((2, 2), dtype=complex, order="C")
        j_s = 1.0 / 2
        for ip, _ in enumerate((-j_s, j_s)):
            for i, m_s in enumerate((-j_s, j_s)):
                if ip == i:
                    sz[ip, i] = m_s
        return sz


@dataclass(frozen=True)
class PauliMatrices:
    """Pauli matrices.

    hbar is not multiplied.

    Matrix convention is different from SxSySzMatrices. Here, up and down spins
    are assigned to m=1/2 and -1/2, respectively. This means,

        Sgx = [[0, 1], [1, 0]] @ SxSySzMatrices.Sx @ [[0, 1], [1, 0]] * 2
        Sgy = [[0, 1], [1, 0]] @ SxSySzMatrices.Sy @ [[0, 1], [1, 0]] * 2
        Sgz = [[0, 1], [1, 0]] @ SxSySzMatrices.Sz @ [[0, 1], [1, 0]] * 2

    """

    Sgx: tuple = ((0, 1), (1, 0))
    Sgy: tuple = ((0, -1j), (1j, 0))
    Sgz: tuple = ((1, 0), (0, -1))


class SpinorRotationMatrices:
    r"""Rotation matrix for spinor, D\sigma'\sigma(n,alpha).

    D^{\sigma',\sigma(\mathbf{n}, \alpha) =
        <\sigma'|e^{-i/\hbar \mathbf{L}\cdot\mathbf{n}\alpha}|\sigma>.

    By default, the 2x2 matrix is arranged with spin-up at index-0 and spin-down at
    index-1. With `sxsysz`, it is arranged with spin-up at index-1 and spin-down at
    index-0.

    """

    def __init__(
        self,
        R: np.ndarray,
        lattice: np.ndarray,
        sxsysz: SxSySzMatrices | None = None,
    ):
        """Init method.

        Parameters
        ----------
        R : ndarray
            Rotation matrix wrt basis vectors (not Cartesian).
            shape=(3, 3).
        lattice : ndarray.
            Basis vectors in column vectors.
            shape=(3, 3)
        sxsysz : SxSySzMatrices, optional
            Generators of spin rotation. If this is not given, PauliMatrices is used.

        """
        if sxsysz is None:
            self._sxsysz = PauliMatrices()
            self._sx = np.array(self._sxsysz.Sgx) / 2
            self._sy = np.array(self._sxsysz.Sgy) / 2
            self._sz = np.array(self._sxsysz.Sgz) / 2
        else:
            self._sxsysz = sxsysz
            self._sx = self._sxsysz.Sx
            self._sy = self._sxsysz.Sy
            self._sz = self._sxsysz.Sz
        self._d = None
        self._n_vec, self._r_order, self._detR = get_n_and_rotation_order(R, lattice)
        if self._r_order == 1:
            self._alpha = 0.0
        else:
            self._alpha = 2 * np.pi / self._r_order

    @property
    def Delta(self) -> np.ndarray:
        """Return rotation matrix of spinor."""
        return self._Delta

    def run(self):
        """Rotate spinor matrix."""
        self._compute_d()
        self._Delta = np.array(self._d, dtype="complex128", order="C")
        return self

    def _compute_d(self):
        r"""Calculate <sigma'|e^iS.n\alpha|sigma>."""
        self._d = []
        n = self._n_vec
        sn = n[0] * self._sx + n[1] * self._sy + n[2] * self._sz
        w, U = np.linalg.eigh(sn)
        self._d = (U * np.exp(-1j * w * self._alpha)) @ U.T.conj()


def rotate_delta_vals_in_spin_space(
    delta_vals_rotated: list[NDArray], r: NDArray, lattice: NDArray
) -> list[NDArray]:
    r"""Take linear combination of delta vals with spin rotation matrix.

    In this method A B A^+ is computed, where A is the 2x2 rotation matrix of
    spin space and B is ``delta_vals_rotated``. What is computed is more
    precisely,

    \sum_{i_s1, i_s2} A_{i_sp, i_s1} B_{i_s1, i_s2} (A^+)_{i_s2, i_s}

    The data order of ``delta_vals_rotated`` is as follows. This method is
    performed on when ncdij=4. This 4 corresponds to 2x2 matrix in spinor
    representation. In VASP covention, the data order is defined as

    cdij=[1, 2, 3, 4] -> [11, 12, 21, 22].

    SpinorRotationMatrices.Delta is a 2x2 numpy array.

    Parameters
    ----------
    delta_vals_rotated : list[NDArray]
        Rotated delta local potential or delta-Dijqij.
    lattice : ndarray.
        Basis vectors of supercell in column vectors.
        shape=(3, 3)
    r : ndarray
        Rotation matrix wrt basis vectors (not Cartesian).
        shape=(3, 3).

    """
    srm = SpinorRotationMatrices(r, lattice).run()
    assert srm.Delta.shape == (2, 2)

    delta_vals_spinor_rotated = [np.zeros_like(delta_vals_rotated[0]) for _ in range(4)]
    for i_sp in (0, 1):
        for i_s in (0, 1):
            for i_s1 in (0, 1):
                for i_s2 in (0, 1):
                    d_d = srm.Delta[i_sp, i_s1] * srm.Delta.T.conj()[i_s2, i_s]
                    delta_vals_spinor_rotated[i_sp * 2 + i_s] += (
                        d_d * delta_vals_rotated[i_s1 * 2 + i_s2]
                    )
    return delta_vals_spinor_rotated


def apply_time_reversal_in_spin_space(delta_vals: list[NDArray]) -> list[NDArray]:
    r"""Apply time reversal to delta vals.

    Time reversal is :math:`\Theta = i\sigma_y K`. For ncdij=4, the 2x2 matrix
    in spinor representation transforms as
    :math:`B \rightarrow \sigma_y B^* \sigma_y`, i.e., with the data order
    [11, 12, 21, 22],

    [B11, B12, B21, B22] -> [B22^*, -B21^*, -B12^*, B11^*].

    This keeps the scalar part and flips the sign of the magnetic part. For
    ncdij=2, spin up and down are exchanged. For ncdij=1, only the complex
    conjugate is taken.

    Parameters
    ----------
    delta_vals : list[NDArray]
        Delta local potential or delta-Dijqij, in real space and in a real
        basis, after spatial and spin rotations.

    """
    ncdij = len(delta_vals)
    if ncdij == 4:
        return [
            delta_vals[3].conj(),
            -delta_vals[2].conj(),
            -delta_vals[1].conj(),
            delta_vals[0].conj(),
        ]
    if ncdij == 2:
        return [delta_vals[1].conj(), delta_vals[0].conj()]
    return [delta_vals[0].conj()]
