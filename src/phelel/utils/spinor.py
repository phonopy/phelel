"""Routines to handle spinor rotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

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
    def Sx(self) -> Optional[np.ndarray]:
        """Return <sigma'|Sx|sigma> for all sigma channel."""
        return self._Sx

    @property
    def Sy(self) -> Optional[np.ndarray]:
        """Return <sigma'|Sy|sigma> for all sigma channel."""
        return self._Sy

    @property
    def Sz(self) -> Optional[np.ndarray]:
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
        sxsysz: Optional[SxSySzMatrices] = None,
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
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        self._Delta = np.array(self._d, dtype=dtype, order="C")
        return self

    def _compute_d(self):
        r"""Calculate <sigma'|e^iS.n\alpha|sigma>."""
        self._d = []
        n = self._n_vec
        sn = n[0] * self._sx + n[1] * self._sy + n[2] * self._sz
        w, U = np.linalg.eigh(sn)
        self._d = (U * np.exp(-1j * w * self._alpha)) @ U.T.conj()
