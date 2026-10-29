"""Exact evaluation of a Fourier series of the supercell on the grid of phelel.

A function given on the FFT mesh of the supercell (N^s_1 x N^s_2 x N^s_3
points) is the Fourier series

    f(x) = sum_g f(g) exp(2 pi i g . x),

where x are the fractional coordinates in the supercell basis vectors and g
the integer coordinates of the reciprocal lattice vectors of the supercell.
f(g) is periodic with the period N^s_k in g_k, so one g is chosen from each
g + N^s n (n integer vectors).  The choice does not change the values on the
FFT mesh, but it changes the values between the mesh points.  Two choices are
implemented:

- "box": -ceil(N^s_k / 2) + 1 <= g_k <= floor(N^s_k / 2), so for even N^s_k
  the coefficient at the array index N^s_k / 2 is assigned to
  g_k = +N^s_k / 2.  This is the box used with finufft.
- "shortest": the g + N^s n with the shortest reciprocal lattice vector.  When
  several have the same length, the coefficient is divided equally among them.

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
from typing import Literal

import numpy as np
import scipy.fft
from numpy.typing import NDArray
from phonopy.structure.cells import determinant, get_reduced_bases
from phonors import bz_grid_addresses, snf3x3

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

    def __init__(
        self,
        fft_mesh: Sequence[int] | NDArray,
        p2s_matrix: NDArray,
        representatives: Literal["box", "shortest"] = "box",
        supercell_lattice: NDArray | None = None,
    ):
        """Init method.

        Parameters
        ----------
        fft_mesh : array_like
            Mesh numbers N^p of the primitive cell.
            dtype='int64', shape=(3,)
        p2s_matrix : ndarray
            Supercell matrix M^s relative to the primitive cell.
            dtype='int64', shape=(3, 3)
        representatives : str, optional
            Choice of g from each g + N^s n, "box" (default) or "shortest".  See
            the module docstring.
        supercell_lattice : ndarray, optional
            Basis vectors of the supercell in row vectors, needed for
            "shortest" to compare the lengths of the reciprocal lattice
            vectors.
            dtype='double', shape=(3, 3)

        """
        self._set_representatives(representatives, supercell_lattice)
        self._fft_mesh = np.array(fft_mesh, dtype="int64")
        p2s_matrix = np.array(p2s_matrix, dtype="int64")
        self._set_integer_coordinates(p2s_matrix)
        self._set_lattice_points_and_qpoints(p2s_matrix)
        self._set_phase_qm()

    def _set_representatives(
        self,
        representatives: Literal["box", "shortest"],
        supercell_lattice: NDArray | None,
    ) -> None:
        """Set the choice of g from each g + N^s n.

        The g of "shortest" depend on the FFT mesh of the supercell, which is
        known only when the coefficients are given, so they are computed in
        get_phi and cached for each mesh.

        """
        if representatives not in ("box", "shortest"):
            raise ValueError(
                'representatives has to be "box" or "shortest", '
                f'not "{representatives}".'
            )
        if representatives == "shortest" and supercell_lattice is None:
            raise ValueError('supercell_lattice is needed for "shortest".')
        self._representatives = representatives
        self._supercell_lattice = (
            None
            if supercell_lattice is None
            else np.array(supercell_lattice, dtype="double")
        )
        self._shortest_cache: dict[
            tuple[int, ...],
            tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.double]],
        ] = {}

    def _set_integer_coordinates(self, p2s_matrix: NDArray[np.int64]) -> None:
        """Set D and B = D (M^s)^-1 to split h = (M^s)^-T g into K + q.

        D = |det M^s| is the number of primitive cells in the supercell.
        (M^s)^-1 = adj(M^s) / det(M^s) with the integer adjugate matrix, so
        B = D (M^s)^-1 is an integer matrix.  For g in row vectors,

            D h = D (M^s)^-T g = (g @ B)^T

        is an integer vector, and the integer part of h is K = floor(D h / D),
        computed by the integer floor division (g @ B) // D.  The fractional
        part q = h - K is not computed from D h; its label is given by the
        Smith normal form (see _set_lattice_points_and_qpoints).

        """
        self._num_cells = abs(int(determinant(p2s_matrix)))
        self._B = np.array(
            np.rint(self._num_cells * np.linalg.inv(p2s_matrix)), dtype="int64"
        )

    def _set_lattice_points_and_qpoints(self, p2s_matrix: NDArray[np.int64]) -> None:
        """Set the lattice points l and the q points by the Smith normal form.

        snf3x3 gives the unimodular U_1 and U_2 with U_1 M^s U_2 = diag(d), so

            (M^s)^-T = U_1^T diag(d)^-1 U_2^T,
            h = (M^s)^-T g = U_1^T diag(d)^-1 (U_2^T g).

        Adding d_k to the k-th element of U_2^T g adds the integer vector
        U_1^T e_k to h, so the fractional part q of h depends only on

            n' = U_2^T g mod d,

        and the D = d_1 d_2 d_3 labels n' give D different q because U_1^T is
        unimodular:

            q = U_1^T diag(d)^-1 n' mod 1.

        With row vectors, D q = ((n' * D / d) @ U_1) mod D, where D / d_k is an
        integer because D = d_1 d_2 d_3.  n' runs in the order of
        np.ndindex(d).

        get_lattice_points gives l = U_1^-1 n with n in the order of
        np.ndindex(d).  Then q . l = n' . diag(d)^-1 U_1 U_1^-1 n mod 1, so

            exp(2 pi i q . l) = prod_k exp(2 pi i n'_k n_k / d_k),

        and the sum over q in sum_over_qpoints is a DFT of the shape d from n'
        to n.  U_2 is kept to compute n' of each g in get_phi.

        """
        lattice_points, _ = get_lattice_points(p2s_matrix)
        self._lattice_points = np.array(lattice_points, dtype="int64")
        d, U1, U2 = snf3x3(p2s_matrix)
        self._snf_diag = np.array(d, dtype="int64")
        self._U2 = np.array(U2, dtype="int64")
        n_prime = np.array(list(np.ndindex(*self._snf_diag)), dtype="int64")
        Dq = ((n_prime * (self._num_cells // self._snf_diag)) @ U1) % self._num_cells
        self._qpoints = Dq / self._num_cells

    def _set_phase_qm(self) -> None:
        """Set the phase factors exp(2 pi i q . (N^p)^-1 m) of phi_q(m).

        m = (m_1, m_2, m_3) are the mesh indices of fft_mesh N^p.  The array
        has the shape (D, N^p_3, N^p_2, N^p_1): the first axis follows
        qpoints, and the other axes are m_3, m_2, m_1, the order of the arrays
        of the local potentials.  np.indices of (N^p_3, N^p_2, N^p_1) gives
        (m_3, m_2, m_1), which is reversed to (m_1, m_2, m_3).

        """
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
        mesh = shape[::-1]
        if self._representatives == "box":
            g = _get_reciprocal_lattice_vectors(mesh)
            source = None
            weight = None
        else:
            g, source, weight = self._get_shortest_representatives(mesh)
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
            w = c.ravel() if source is None else c.ravel()[source] * weight
            if phase is not None:
                w = w * phase
            F = np.bincount(index, weights=w.real, minlength=size) + 1j * np.bincount(
                index, weights=w.imag, minlength=size
            )
            F = F.reshape(self._num_cells, n3, n2, n1)
            phi = scipy.fft.ifftn(F, axes=(1, 2, 3), norm="forward", workers=-1)
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
        values = scipy.fft.ifftn(
            phi.reshape(*self._snf_diag, -1),
            axes=(0, 1, 2),
            norm="forward",
            workers=-1,
        )
        return values.ravel()

    def _get_shortest_representatives(
        self, mesh: tuple[int, ...]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.double]]:
        """Return get_shortest_representatives for the mesh, cached."""
        if mesh not in self._shortest_cache:
            assert self._supercell_lattice is not None
            self._shortest_cache[mesh] = get_shortest_representatives(
                mesh, self._supercell_lattice
            )
        return self._shortest_cache[mesh]


def _get_reciprocal_lattice_vectors(mesh: Sequence[int] | NDArray) -> NDArray[np.int64]:
    """Return a reciprocal lattice vector g in the box for each class g mod N^s.

    g are the reciprocal lattice vectors of the supercell in the integer
    coordinates of its reciprocal basis.  Each class g mod N^s is represented by
    the g in the box -ceil(N^s_k / 2) + 1 <= g_k <= floor(N^s_k / 2), which is
    used with representatives="box".

    Parameters
    ----------
    mesh : array_like
        FFT mesh of the supercell (N^s_1, N^s_2, N^s_3).
        dtype='int64', shape=(3,)

    Returns
    -------
    ndarray
        (g_1, g_2, g_3) of the classes.  The row

            j = (i_3 N^s_2 + i_2) N^s_1 + i_1

        is the class of g_k = i_k mod N^s_k (0 <= i_k < N^s_k), with g_k = i_k
        for i_k <= N^s_k / 2 and g_k = i_k - N^s_k otherwise.  This is the
        order of an array of the shape (N^s_3, N^s_2, N^s_1) flattened.  So g_1
        runs fastest and g_3 slowest, each in the order 0, 1, ...,
        floor(N^s_k / 2), -ceil(N^s_k / 2) + 1, ..., -1.  For mesh (3, 2, 1),
        the rows are (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (1, 1, 0),
        (-1, 1, 0).
        dtype='int64', shape=(N^s_1 N^s_2 N^s_3, 3)

    """
    freqs = []
    for n in mesh[::-1]:
        i = np.arange(n, dtype="int64")
        freqs.append(np.where(i <= n // 2, i, i - n))
    g3, g2, g1 = np.meshgrid(*freqs, indexing="ij")
    return np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=1)


def get_shortest_representatives(
    mesh: Sequence[int] | NDArray,
    supercell_lattice: NDArray,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.double]]:
    """Return the shortest g + N^s n for each class g mod N^s.

    For each reciprocal lattice vector g of _get_reciprocal_lattice_vectors, in
    the integer coordinates of the reciprocal basis of the supercell, the
    vectors g + N^s n with the shortest |G| are chosen, where G = sum_k g_k b_k
    with the reciprocal basis vectors b_k of the supercell (without the factor
    2 pi).  The coefficient is divided equally among the vectors of the same
    length.

    These are the points of the mesh in the Brillouin zone of the period
    lattice {N^s n}, including those on its surface, and they are computed by
    phonors.bz_grid_addresses with the mesh N^s.  phonors takes two squared
    lengths as equal when they differ by less than 0.01 max_i |r_i|^2 /
    (N^s_i)^2, computed from the basis vectors r_i of the period lattice that
    it is given.  For a skewed basis of the supercell, N^s_k b_k are long and
    this tolerance can be larger than the difference to the next longer
    vectors, so a Delaunay-reduced basis, the rows of P B with the rows
    N^s_k b_k of B (see _get_reduced_period_basis), is given with
    Q = (P^T)^-1.  Then the mesh and the returned g stay in the coordinates g.

    Parameters
    ----------
    mesh : array_like
        FFT mesh of the supercell (N^s_1, N^s_2, N^s_3).
        dtype='int64', shape=(3,)
    supercell_lattice : ndarray
        Basis vectors of the supercell in row vectors.
        dtype='double', shape=(3, 3)

    Returns
    -------
    g : ndarray
        Chosen reciprocal lattice vectors in the integer coordinates
        (g_1, g_2, g_3).
        dtype='int64', shape=(num, 3)
    source : ndarray
        Row j of _get_reciprocal_lattice_vectors of the class of each chosen g,
        ((g_3 mod N^s_3) N^s_2 + (g_2 mod N^s_2)) N^s_1 + (g_1 mod N^s_1).
        dtype='int64', shape=(num,)
    weight : ndarray
        1 / (number of g chosen for the same class).
        dtype='double', shape=(num,)

    """
    mesh = np.array(mesh, dtype="int64")
    period_lattice = np.diag(mesh) @ np.linalg.inv(supercell_lattice).T
    P = _get_reduced_period_basis(period_lattice)
    Q = np.array(np.rint(np.linalg.inv(P.T)), dtype="int64")
    addresses, bz_map, bzg2grg = bz_grid_addresses(
        mesh,
        Q,
        np.zeros(3, dtype="int64"),
        np.array((P @ period_lattice).T, dtype="double", order="C"),
        2,
    )
    g = np.array(addresses, dtype="int64")
    source = np.array(bzg2grg, dtype="int64")
    return g, source, 1.0 / np.diff(bz_map)[source]


def _get_reduced_period_basis(period_lattice: NDArray) -> NDArray[np.int64]:
    """Return P with P period_lattice Delaunay-reduced.

    Parameters
    ----------
    period_lattice : ndarray
        Basis vectors N^s_k b_k of the period lattice in row vectors.
        dtype='double', shape=(3, 3)

    Returns
    -------
    ndarray
        Unimodular matrix P.
        dtype='int64', shape=(3, 3)

    """
    reduced = get_reduced_bases(period_lattice, method="delaunay")
    P = np.array(np.rint(reduced @ np.linalg.inv(period_lattice)), dtype="int64")
    if abs(determinant(P)) != 1 or not np.allclose(P @ period_lattice, reduced):
        raise RuntimeError("Delaunay reduction of the period lattice failed.")
    return P
