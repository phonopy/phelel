"""Exact evaluation of a Fourier series of the supercell on the grid of phelel.

A function given on the FFT mesh of the supercell (N^s_1 x N^s_2 x N^s_3
points) is the Fourier series

    f(x) = sum_g f(g) exp(2 pi i g . x),

where x are the fractional coordinates in the supercell basis vectors and g
the integer coordinates of the reciprocal lattice vectors of the supercell.
The g are taken from the box -ceil(N^s_k / 2) + 1 <= g_k <= floor(N^s_k / 2),
so for even N^s_k the coefficient at the array index N^s_k / 2 is assigned to
g_k = +N^s_k / 2.  This is the box used with finufft.

The series is evaluated at the grid points of get_grid_points, the points
(N^p)^-1 m + l in the primitive cell coordinates with the mesh indices m of
fft_mesh (N^p) and the lattice points l of get_lattice_points.  With the
coordinates h = (M^s)^-T g of g in the reciprocal basis of the primitive
cell (M^s = p2s_matrix) split into the integer part K and the fractional part
q,

    f(l, m) = sum_q exp(2 pi i q . l) phi_q(m),
    phi_q(m) = exp(2 pi i q . (N^p)^-1 m)
               sum_Kbar F_q(Kbar) exp(2 pi i Kbar . (N^p)^-1 m),

where F_q(Kbar) is the sum of the coefficients with the same q and
Kbar = K mod N^p.  The sum over Kbar is a DFT of size N^p_1 x N^p_2 x N^p_3,
so no approximation is involved.

"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from phonopy.structure.cells import determinant
from phonors import snf3x3

from phelel.utils.lattice_points import get_lattice_points


def get_fourier_coefficients(values: NDArray) -> NDArray[np.cdouble]:
    """Return the Fourier coefficients of values on the FFT mesh of the supercell.

    Parameters
    ----------
    values : ndarray
        Values on the FFT mesh of the supercell.
        shape=(N^s_3, N^s_2, N^s_1)

    Returns
    -------
    ndarray
        f(g) = 1 / (N^s_1 N^s_2 N^s_3) sum_m values(m) exp(-2 pi i g . (N^s)^-1 m)
        at the array indices (g_3, g_2, g_1) mod (N^s_3, N^s_2, N^s_1).
        dtype='complex128', shape=values.shape

    """
    coefficients: NDArray[np.cdouble] = np.fft.fftn(values)
    coefficients /= values.size
    return coefficients


class FourierSeriesOnGrid:
    """Evaluate Fourier series of the supercell exactly at the grid points.

    Attributes
    ----------
    qpoints : ndarray
        q points commensurate with the supercell in the reciprocal basis of the
        primitive cell, 0 <= q_k < 1, in the order of the first axis of phi.
        With the Smith normal form U_1 M^s U_2 = diag(d), q = U_1^T diag(d)^-1 n'
        mod 1, and n' runs in the order of np.ndindex(d).
        dtype='double', shape=(D, 3)
    lattice_points : ndarray
        Lattice points of the primitive cell, one from each class modulo the
        lattice of the supercell, in the basis vectors of the primitive cell,
        the same as those of get_grid_points.  See get_lattice_points.
        dtype='int64', shape=(D, 3)

    """

    def __init__(self, fft_mesh: Sequence[int] | NDArray, p2s_matrix: NDArray):
        """Init method.

        Parameters
        ----------
        fft_mesh : array_like
            Mesh numbers N^p of the primitive cell.
            dtype='int64', shape=(3,)
        p2s_matrix : ndarray
            Supercell matrix M^s relative to the primitive cell.
            dtype='int64', shape=(3, 3)

        """
        self._fft_mesh = np.array(fft_mesh, dtype="int64")
        p2s_matrix = np.array(p2s_matrix, dtype="int64")
        self._num_cells = abs(int(determinant(p2s_matrix)))

        # D (M^s)^-1 is an integer matrix, so D h = g @ B is computed with
        # integers for row vectors g.
        self._B = np.array(
            np.rint(self._num_cells * np.linalg.inv(p2s_matrix)), dtype="int64"
        )
        # The lattice points are l = U_1^-1 n with n in the order of
        # np.ndindex(d), and the q points are labeled by n' in the same order,
        # so that exp(2 pi i q . l) = prod_k exp(2 pi i n'_k n_k / d_k).
        lattice_points, _ = get_lattice_points(p2s_matrix)
        self._lattice_points = np.array(lattice_points, dtype="int64")
        d, U1, U2 = snf3x3(p2s_matrix)
        self._snf_diag = np.array(d, dtype="int64")
        self._U2 = np.array(U2, dtype="int64")
        n_prime = np.array(list(np.ndindex(*self._snf_diag)), dtype="int64")
        Dq = ((n_prime * (self._num_cells // self._snf_diag)) @ U1) % self._num_cells
        self._qpoints = Dq / self._num_cells

        # exp(2 pi i q . (N^p)^-1 m), shape=(D, N^p_3, N^p_2, N^p_1)
        m = np.indices([int(n) for n in self._fft_mesh[::-1]]).reshape(3, -1)[::-1].T
        self._phase_qm = np.exp(
            2j * np.pi * (self._qpoints / self._fft_mesh) @ m.T
        ).reshape(-1, *self._fft_mesh[::-1])

    @property
    def qpoints(self) -> NDArray[np.double]:
        """Return q points in the order of the first axis of phi."""
        return self._qpoints

    @property
    def lattice_points(self) -> NDArray[np.int64]:
        """Return lattice points of the primitive cell modulo the supercell."""
        return self._lattice_points

    def evaluate(
        self,
        coefficients: list[NDArray],
        rotation: NDArray | None = None,
        translation: NDArray | None = None,
    ) -> list[NDArray[np.cdouble]]:
        """Return f(g^-1 x) at the grid points x for each set of coefficients.

        Parameters
        ----------
        See get_phi.

        Returns
        -------
        list of ndarray
            Values at the grid points in the order of get_grid_points.
            dtype='complex128', shape=(D * prod(fft_mesh),) each

        """
        return [
            self.sum_over_qpoints(phi)
            for phi in self.get_phi(coefficients, rotation, translation)
        ]

    def get_phi(
        self,
        coefficients: list[NDArray],
        rotation: NDArray | None = None,
        translation: NDArray | None = None,
    ) -> list[NDArray[np.cdouble]]:
        """Return phi_q(m) of f(g^-1 x) for the operation g = {rotation|translation}.

        The rotated function f(g^-1 x) has the coefficients
        exp(-2 pi i g' . w) f(W^T g') at g' = W^-T g, where W and w are rotation
        and translation.

        Parameters
        ----------
        coefficients : list of ndarray
            Fourier coefficients from get_fourier_coefficients, e.g., of the
            spin components.  All have the same shape.
            dtype='complex128', shape=(N^s_3, N^s_2, N^s_1) each
        rotation : ndarray, optional
            Rotation W in the supercell basis vectors.  Default is None, which
            means the identity.
            dtype='int64', shape=(3, 3)
        translation : ndarray, optional
            Translation w in the supercell basis vectors.  Default is None,
            which means zero.
            dtype='double', shape=(3,)

        Returns
        -------
        list of ndarray
            phi_q(m) with q in the order of qpoints.
            dtype='complex128', shape=(D, N^p_3, N^p_2, N^p_1) each

        """
        shape = coefficients[0].shape
        assert all(c.shape == shape for c in coefficients)
        g = _get_box(shape)
        if rotation is not None:
            g = g @ np.array(np.rint(np.linalg.inv(rotation)), dtype="int64")
        if translation is None:
            phase = None
        else:
            phase = np.exp(-2j * np.pi * g @ np.asarray(translation, dtype="double"))

        # h = K + q with D h = g @ B computed with integers, and q labeled by
        # n' = U_2^T g mod d.
        K = (g @ self._B) // self._num_cells
        i_q = np.ravel_multi_index(((g @ self._U2) % self._snf_diag).T, self._snf_diag)
        Kbar = K % self._fft_mesh
        n1, n2, n3 = self._fft_mesh
        index = ((i_q * n3 + Kbar[:, 2]) * n2 + Kbar[:, 1]) * n1 + Kbar[:, 0]
        size = self._num_cells * n1 * n2 * n3

        phis = []
        for c in coefficients:
            w = c.ravel() if phase is None else c.ravel() * phase
            F = np.bincount(index, weights=w.real, minlength=size) + 1j * np.bincount(
                index, weights=w.imag, minlength=size
            )
            F = F.reshape(self._num_cells, n3, n2, n1)
            phi = np.fft.ifftn(F, axes=(1, 2, 3)) * (n1 * n2 * n3)
            phis.append(phi * self._phase_qm)
        return phis

    def sum_over_qpoints(self, phi: NDArray) -> NDArray[np.cdouble]:
        """Return sum_q exp(2 pi i q . l) phi_q(m) in the order of get_grid_points.

        With q labeled by n' and l by n (see qpoints), the sum is a DFT of the
        shape diag(d) from n' to n.

        Parameters
        ----------
        phi : ndarray
            phi_q(m) from get_phi.
            dtype='complex128', shape=(D, N^p_3, N^p_2, N^p_1)

        Returns
        -------
        ndarray
            Values at the grid points, the lattice point running slowest and
            m_1 fastest.
            dtype='complex128', shape=(D * prod(fft_mesh),)

        """
        values = (
            np.fft.ifftn(phi.reshape(*self._snf_diag, -1), axes=(0, 1, 2))
            * self._num_cells
        )
        return values.ravel()


def _get_box(shape: tuple[int, ...]) -> NDArray[np.int64]:
    """Return g of the coefficients of get_fourier_coefficients.

    Parameters
    ----------
    shape : tuple of int
        (N^s_3, N^s_2, N^s_1)

    Returns
    -------
    ndarray
        (g_1, g_2, g_3) in the order of the flattened coefficient array, with
        -ceil(N^s_k / 2) + 1 <= g_k <= floor(N^s_k / 2).
        dtype='int64', shape=(prod(shape), 3)

    """
    freqs = []
    for n in shape:
        i = np.arange(n, dtype="int64")
        freqs.append(np.where(i <= n // 2, i, i - n))
    g3, g2, g1 = np.meshgrid(*freqs, indexing="ij")
    return np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=1)
