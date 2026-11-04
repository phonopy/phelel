"""Exact evaluation of a Fourier series of the supercell on the grid of phelel.

A function given on the FFT mesh of the supercell (N^s_1 x N^s_2 x N^s_3
points) is the Fourier series

    f(x) = sum_g f(g) exp(2 pi i g . x),

where x are the fractional coordinates in the supercell basis vectors and g
the g-points, the points of the reciprocal lattice of the supercell in the
integer coordinates of its basis.  f(g) is periodic with the period N^s_k in
g_k, so the g-points at which f(g) is placed are selected from each
g + N^s t (t integer vectors), one or more.  These are called the selected
g-points.  The selection does not change the values on the FFT mesh, but it
changes the values between the mesh points.  Two rules of the selection
(g_point_selection) are implemented:

- "box": -ceil(N^s_k / 2) + 1 <= g_k <= floor(N^s_k / 2), so for even N^s_k
  the coefficient at the array index N^s_k / 2 is assigned to
  g_k = +N^s_k / 2.  This is the box used with finufft.
- "shortest": the g + N^s t with the shortest reciprocal lattice vector.  When
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

The coefficient f(g) is stored at the index of g mod N^s in the flattened
coefficient array, called the source of g.  For a symmetry operation {W|w},
the coefficient of each selected g goes to the bin (q, Kbar) of g' = W^-T g.
F_q(Kbar) is computed in two steps: the g' of all selected g are sorted by
bin, and the coefficients of g = W^T g' are summed bin by bin.  When W^-T
maps the selected g onto themselves, the g' are the selected g themselves, so
those sorted once for each mesh are used.  This holds for "shortest" when
the FFT mesh of the supercell has the symmetry of the supercell, and it is
checked for each rotation.  Otherwise the g' are sorted for the operation.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.fft
from numpy.typing import NDArray
from phonopy.structure.cells import determinant, get_reduced_bases
from phonors import (
    bin_sorted_supercell_fourier_coefficients,
    bz_grid_addresses,
    g_points_closed_under_rotation,
    snf3x3,
    sort_g_points_by_bin,
)

from phelel.base.potential_change import multiply_by_phase
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


@dataclass
class _MeshData:
    """Data that depend on the FFT mesh of the supercell.

    FourierSeriesOnGrid keeps the data of the last mesh it was given and makes
    them again when the mesh changes.

    Attributes
    ----------
    supercell_mesh : tuple of int
        FFT mesh of the supercell (N^s_1, N^s_2, N^s_3).
    g : ndarray
        Selected g-points in the order of their sources: those of
        _get_reciprocal_lattice_vectors ("box") or of get_shortest_g_points
        ("shortest").
        dtype='int64', shape=(num, 3)
    source_offsets : ndarray
        The selected g-points with the source s are
        g[source_offsets[s]:source_offsets[s + 1]].
        dtype='int64', shape=(N^s_1 N^s_2 N^s_3 + 1,)
    source_weight : ndarray
        1 / (number of the selected g-points with the source s) for each
        source s.
        dtype='double', shape=(N^s_1 N^s_2 N^s_3,)
    g_sorted : ndarray
        g sorted by bin, used for the rotations under which g are closed.
        dtype='int64', shape=(num, 3)
    bin_offsets : ndarray
        The g of the bin i_F are g_sorted[bin_offsets[i_F]:bin_offsets[i_F + 1]].
        dtype='int64', shape=(N_c * prod(fft_mesh) + 1,)
    is_closed : dict
        Whether W^-T maps g onto themselves, by rotation.tobytes(), filled
        when asked (FourierSeriesOnGrid.g_points_closed_under_rotation).

    """

    supercell_mesh: tuple[int, ...]
    g: NDArray[np.int64]
    source_offsets: NDArray[np.int64]
    source_weight: NDArray[np.double]
    g_sorted: NDArray[np.int64]
    bin_offsets: NDArray[np.int64]
    is_closed: dict[bytes, bool] = field(default_factory=dict)


class FourierSeriesOnGrid:
    """Evaluate Fourier series of the supercell exactly at the grid points.

    Attributes
    ----------
    qpoints : ndarray
        q points commensurate with the supercell in the reciprocal basis of the
        primitive cell, 0 <= q_k < 1, in the order of the first axis of phi.
        With the Smith normal form P M^s Q = diag(d), q = P^T diag(d)^-1 n_q
        mod 1, and n_q runs in the order of np.ndindex(d).
        dtype='double', shape=(N_c, 3)
    lattice_points : ndarray
        Lattice points of the primitive cell, one from each class modulo the
        lattice of the supercell, in the basis vectors of the primitive cell,
        the same as those of get_grid_points.  See get_lattice_points.
        dtype='int64', shape=(N_c, 3)

    """

    def __init__(
        self,
        fft_mesh: Sequence[int] | NDArray,
        p2s_matrix: NDArray,
        g_point_selection: Literal["box", "shortest"] = "box",
        supercell_lattice: NDArray | None = None,
        lang: Literal["Rust", "Python"] = "Rust",
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
        g_point_selection : str, optional
            Rule of the selection of g from each g + N^s t, "box" (default)
            or "shortest".  See the module docstring.
        supercell_lattice : ndarray, optional
            Basis vectors of the supercell in row vectors, needed for
            "shortest" to compare the lengths of the reciprocal lattice
            vectors.
            dtype='double', shape=(3, 3)
        lang : str, optional
            Implementation of the sums F_q(Kbar) and of the multiplication by
            the phase in get_phi, "Rust" (default) by phonors
            (sort_g_points_by_bin, bin_sorted_supercell_fourier_coefficients,
            g_points_closed_under_rotation and multiply_by_phase) or "Python" by
            their pure-Python counterparts in this module.

        """
        if lang not in ("Rust", "Python"):
            raise ValueError(f'lang has to be "Rust" or "Python", not "{lang}".')
        self._lang = lang
        self._set_g_point_selection(g_point_selection, supercell_lattice)
        self._fft_mesh = np.array(fft_mesh, dtype="int64")
        self._p2s_matrix = np.array(p2s_matrix, dtype="int64")
        self._set_integer_coordinates(self._p2s_matrix)
        self._set_lattice_points_and_qpoints(self._p2s_matrix)
        self._set_phase_qm()

    def _set_g_point_selection(
        self,
        g_point_selection: Literal["box", "shortest"],
        supercell_lattice: NDArray | None,
    ) -> None:
        """Set the rule of the selection of g from each g + N^s t.

        The selected g depend on the FFT mesh of the supercell, which is known
        only when the coefficients are given, so they are made in get_phi with
        the other data of the mesh (_MeshData, _get_mesh_data).

        """
        if g_point_selection not in ("box", "shortest"):
            raise ValueError(
                'g_point_selection has to be "box" or "shortest", '
                f'not "{g_point_selection}".'
            )
        if g_point_selection == "shortest" and supercell_lattice is None:
            raise ValueError('supercell_lattice is needed for "shortest".')
        self._g_point_selection = g_point_selection
        self._supercell_lattice = (
            None
            if supercell_lattice is None
            else np.array(supercell_lattice, dtype="double")
        )
        self._mesh_data: _MeshData | None = None

    def _set_integer_coordinates(self, p2s_matrix: NDArray[np.int64]) -> None:
        """Set N_c and N_c (M^s)^-1 to split h = (M^s)^-T g into K + q.

        N_c = |det M^s| is the number of primitive cells in the supercell.
        (M^s)^-1 = adj(M^s) / det(M^s) with the integer adjugate matrix, so
        N_c (M^s)^-1 (_nc_m_inv) is an integer matrix.  For g in row vectors,

            N_c h = N_c (M^s)^-T g = (g @ _nc_m_inv)^T

        is an integer vector, and the integer part of h is
        K = floor(N_c h / N_c), computed by the integer floor division
        (g @ _nc_m_inv) // N_c.  The fractional part q = h - K is not computed
        from N_c h; its label is given by the Smith normal form (see
        _set_lattice_points_and_qpoints).

        """
        self._num_cells = abs(int(determinant(p2s_matrix)))
        self._nc_m_inv = np.array(
            np.rint(self._num_cells * np.linalg.inv(p2s_matrix)), dtype="int64"
        )

    def _set_lattice_points_and_qpoints(self, p2s_matrix: NDArray[np.int64]) -> None:
        """Set the lattice points l and the q points by the Smith normal form.

        snf3x3 gives the unimodular P and Q with P M^s Q = diag(d), so

            (M^s)^-T = P^T diag(d)^-1 Q^T,
            h = (M^s)^-T g = P^T diag(d)^-1 (Q^T g).

        Adding d_k to the k-th element of Q^T g adds the integer vector
        P^T e_k to h, so the fractional part q of h depends only on

            n_q = Q^T g mod d,

        and the N_c = d_1 d_2 d_3 labels n_q give N_c different q because P^T
        is unimodular:

            q = P^T diag(d)^-1 n_q mod 1.

        With row vectors, N_c q = ((n_q * N_c / d) @ P) mod N_c, where N_c / d_k
        is an integer because N_c = d_1 d_2 d_3.  n_q runs in the order of
        np.ndindex(d).

        get_lattice_points gives l = P^-1 n_l with n_l in the order of
        np.ndindex(d).  Then q . l = n_q . diag(d)^-1 P P^-1 n_l mod 1, so

            exp(2 pi i q . l) = prod_k exp(2 pi i (n_q)_k (n_l)_k / d_k),

        and the sum over q in sum_over_qpoints is a DFT of the shape d from n_q
        to n_l.  Q is kept to compute n_q of each g in get_phi with
        lang="Python".  With lang="Rust", phonors.sort_g_points_by_bin computes
        n_q with the same snf3x3.

        """
        lattice_points, _ = get_lattice_points(p2s_matrix)
        self._lattice_points = np.array(lattice_points, dtype="int64")
        d, P, Q = snf3x3(p2s_matrix)
        self._snf_diag = np.array(d, dtype="int64")
        self._snf_Q = np.array(Q, dtype="int64")
        n_q = np.array(list(np.ndindex(*self._snf_diag)), dtype="int64")
        nc_q = ((n_q * (self._num_cells // self._snf_diag)) @ P) % self._num_cells
        self._qpoints = nc_q / self._num_cells

    def _set_phase_qm(self) -> None:
        """Set the phase factors exp(2 pi i q . (N^p)^-1 m) of phi_q(m).

        m = (m_1, m_2, m_3) are the mesh indices of fft_mesh N^p.  The array
        has the shape (N_c, N^p_3, N^p_2, N^p_1): the first axis follows
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
        rotation: NDArray,
        translation: NDArray,
    ) -> list[NDArray[np.cdouble]]:
        """Return f(g^-1 x) at the grid points x for each set of coefficients.

        Parameters
        ----------
        See get_phi.

        Returns
        -------
        list of ndarray
            Values at the grid points in the order of get_grid_points.
            dtype='complex128', shape=(N_c * prod(fft_mesh),) each

        """
        return [
            self.sum_over_qpoints(phi)
            for phi in self.get_phi(coefficients, rotation, translation)
        ]

    def get_phi(
        self,
        coefficients: list[NDArray],
        rotation: NDArray,
        translation: NDArray,
    ) -> list[NDArray[np.cdouble]]:
        """Return phi_q(m) of f(g^-1 x) for the operation g = {rotation|translation}.

        The rotated function f(g^-1 x) has the coefficients
        exp(-2 pi i g' . w) f(W^T g') at g' = W^-T g, where W and w are rotation
        and translation.  The sums F_q(Kbar) of the coefficients are computed
        by _bin_coefficients, the DFT over Kbar by scipy.fft, and the
        multiplication by the phase by multiply_by_phase following lang.

        Parameters
        ----------
        coefficients : list of ndarray
            Fourier coefficients from get_fourier_coefficients, e.g., of the
            spin components.  All have the same shape.
            dtype='complex128', shape=(N^s_3, N^s_2, N^s_1) each
        rotation : ndarray
            Rotation W in the supercell basis vectors, the identity for f(x)
            itself.
            dtype='int64', shape=(3, 3)
        translation : ndarray
            Translation w in the supercell basis vectors, zero for f(x) itself.
            dtype='double', shape=(3,)

        Returns
        -------
        list of ndarray
            phi_q(m) with q in the order of qpoints.
            dtype='complex128', shape=(N_c, N^p_3, N^p_2, N^p_1) each

        """
        shape = coefficients[0].shape
        assert all(c.shape == shape for c in coefficients)
        Fs = self._bin_coefficients(coefficients, rotation, translation)
        phis = []
        for F in Fs:
            phi = scipy.fft.ifftn(F, axes=(1, 2, 3), norm="forward", workers=-1)
            multiply_by_phase(phi, self._phase_qm, lang=self._lang)
            phis.append(phi)
        return phis

    def _bin_coefficients(
        self,
        coefficients: list[NDArray],
        rotation: NDArray,
        translation: NDArray,
    ) -> list[NDArray[np.cdouble]]:
        """Return F_q(Kbar), the sums of the coefficients with the same q and Kbar.

        The g' = W^-T g of the selected g sorted by bin come from
        _get_g_points_sorted_by_bin, and the coefficients of g = W^T g' are summed
        bin by bin by bin_sorted_supercell_fourier_coefficients of phonors
        (lang="Rust") or _bin_sorted_coefficients_python (lang="Python").

        Parameters
        ----------
        See get_phi.

        Returns
        -------
        list of ndarray
            F_q(Kbar) with q in the order of qpoints.
            dtype='complex128', shape=(N_c, N^p_3, N^p_2, N^p_1) each

        """
        coefficients = [
            np.ascontiguousarray(c, dtype="complex128") for c in coefficients
        ]
        rotation = np.array(rotation, dtype="int64")
        translation = np.array(translation, dtype="double")
        supercell_mesh = coefficients[0].shape[::-1]
        g_sorted, bin_offsets = self._get_g_points_sorted_by_bin(
            supercell_mesh, rotation
        )
        source_weight = self._get_mesh_data(supercell_mesh).source_weight
        Fs = self._sum_by_bin(
            coefficients, g_sorted, source_weight, bin_offsets, rotation, translation
        )
        return [F.reshape(self._num_cells, *self._fft_mesh[::-1]) for F in Fs]

    def _sum_by_bin(
        self,
        coefficients: list[NDArray[np.cdouble]],
        g_sorted: NDArray[np.int64],
        source_weight: NDArray[np.double],
        bin_offsets: NDArray[np.int64],
        rotation: NDArray,
        translation: NDArray,
    ) -> list[NDArray[np.cdouble]]:
        """Sum the coefficients by bin with phonors (lang="Rust") or in Python."""
        if self._lang == "Rust":
            return bin_sorted_supercell_fourier_coefficients(
                coefficients,
                g_sorted,
                source_weight,
                bin_offsets,
                rotation=rotation,
                translation=translation,
            )
        return _bin_sorted_coefficients_python(
            coefficients,
            g_sorted,
            source_weight,
            bin_offsets,
            rotation=rotation,
            translation=translation,
        )

    def _get_mesh_data(self, supercell_mesh: Sequence[int]) -> _MeshData:
        """Return the data of the mesh, made again when the mesh changes."""
        supercell_mesh = tuple(int(n) for n in supercell_mesh)
        if self._mesh_data is None or self._mesh_data.supercell_mesh != supercell_mesh:
            self._mesh_data = self._make_mesh_data(supercell_mesh)
        return self._mesh_data

    def _make_mesh_data(self, supercell_mesh: tuple[int, ...]) -> _MeshData:
        """Select the g-points of the mesh and sort them by bin."""
        if self._g_point_selection == "box":
            g = _get_reciprocal_lattice_vectors(supercell_mesh)
            source_offsets = np.arange(len(g) + 1, dtype="int64")
        else:
            assert self._supercell_lattice is not None
            g, source_offsets = get_shortest_g_points(
                supercell_mesh, self._supercell_lattice
            )
        g_sorted, bin_offsets = self._sort_g_points_by_bin(g, np.eye(3, dtype="int64"))
        return _MeshData(
            supercell_mesh=supercell_mesh,
            g=g,
            source_offsets=source_offsets,
            source_weight=1.0 / np.diff(source_offsets),
            g_sorted=g_sorted,
            bin_offsets=bin_offsets,
        )

    def g_points_closed_under_rotation(
        self, supercell_mesh: Sequence[int], rotation: NDArray
    ) -> bool:
        """Return whether W^-T maps the selected g onto themselves.

        g_points_closed_under_rotation of phonors (lang="Rust") or
        _g_points_closed_under_rotation_python (lang="Python") checks that W^-T g
        is one of the selected g with its source for every selected g.  Then the
        selected g sorted once are used for this rotation.  The result is
        kept in _MeshData.is_closed.

        """
        rotation = np.array(rotation, dtype="int64")
        mesh_data = self._get_mesh_data(supercell_mesh)
        key = rotation.tobytes()
        if key not in mesh_data.is_closed:
            mesh_data.is_closed[key] = self._is_closed_under(mesh_data, rotation)
        return mesh_data.is_closed[key]

    def _is_closed_under(self, mesh_data: _MeshData, rotation: NDArray) -> bool:
        """Check the closure with phonors (lang="Rust") or in Python."""
        g = mesh_data.g
        source_offsets = mesh_data.source_offsets
        supercell_mesh = np.array(mesh_data.supercell_mesh, dtype="int64")
        if self._lang == "Rust":
            return g_points_closed_under_rotation(
                g, source_offsets, supercell_mesh, rotation
            )
        return _g_points_closed_under_rotation_python(
            g, source_offsets, supercell_mesh, rotation
        )

    def _get_g_points_sorted_by_bin(
        self, supercell_mesh: Sequence[int], rotation: NDArray
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return g' = W^-T g of the selected g sorted by bin.

        When g_points_closed_under_rotation, the g' are the selected g
        themselves, and the selected g sorted once (_MeshData.g_sorted) are
        returned.  Otherwise the g' are sorted for this rotation.

        Returns
        -------
        g_sorted : ndarray
            g' in the order of (bin, index of g).
            dtype='int64', shape=(num, 3)
        bin_offsets : ndarray
            The g' of the bin i_F are g_sorted[bin_offsets[i_F]:
            bin_offsets[i_F + 1]].
            dtype='int64', shape=(N_c * prod(fft_mesh) + 1,)

        """
        mesh_data = self._get_mesh_data(supercell_mesh)
        if self.g_points_closed_under_rotation(supercell_mesh, rotation):
            return mesh_data.g_sorted, mesh_data.bin_offsets
        return self._sort_g_points_by_bin(mesh_data.g, rotation)

    def _sort_g_points_by_bin(
        self, g: NDArray[np.int64], rotation: NDArray
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Sort g' = W^-T g by bin with phonors (lang="Rust") or in Python."""
        if self._lang == "Rust":
            return self._sort_g_points_by_bin_rust(g, rotation)
        return self._sort_g_points_by_bin_python(g, rotation)

    def _sort_g_points_by_bin_rust(
        self, g: NDArray[np.int64], rotation: NDArray
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Call sort_g_points_by_bin of phonors."""
        return sort_g_points_by_bin(
            g, self._p2s_matrix, self._fft_mesh, np.array(rotation, dtype="int64")
        )

    def _sort_g_points_by_bin_python(
        self, g: NDArray[np.int64], rotation: NDArray
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return g' = W^-T g sorted by bin and the offsets of the bins.

        The pure-Python counterpart of sort_g_points_by_bin of phonors.  The g' of
        a bin are in the order of g.  See _get_g_points_sorted_by_bin for the
        returned arrays.

        """
        # g' = W^-T g in column vectors is g @ W^-1 in row vectors.
        g_rotated = g @ np.array(np.rint(np.linalg.inv(rotation)), dtype="int64")
        i_F = self._get_bin_indices(g_rotated)
        order = np.argsort(i_F, kind="stable")
        num_bins = self._num_cells * int(np.prod(self._fft_mesh))
        counts = np.bincount(i_F, minlength=num_bins)
        bin_offsets = np.concatenate([[0], np.cumsum(counts)]).astype("int64")
        return np.ascontiguousarray(g_rotated[order]), bin_offsets

    def _get_bin_indices(self, g: NDArray[np.int64]) -> NDArray[np.int64]:
        """Return the flat index i_F of the bin (q, Kbar) of each g.

        h = (M^s)^-T g = K + q with N_c h = g @ _nc_m_inv computed with integers,
        and q labeled by n_q = Q^T g mod d (see _set_integer_coordinates and
        _set_lattice_points_and_qpoints).  With the index i_q of n_q in the order
        of qpoints and Kbar = K mod N^p,

            i_F = ((i_q N^p_3 + Kbar_3) N^p_2 + Kbar_2) N^p_1 + Kbar_1,

        the flat index of the array of the shape (N_c, N^p_3, N^p_2, N^p_1).

        Parameters
        ----------
        g : ndarray
            Points g of the reciprocal lattice of the supercell in integer
            coordinates.
            dtype='int64', shape=(n, 3)

        Returns
        -------
        ndarray
            dtype='int64', shape=(n,)

        """
        K = (g @ self._nc_m_inv) // self._num_cells
        i_q = np.ravel_multi_index(
            ((g @ self._snf_Q) % self._snf_diag).T, self._snf_diag
        )
        Kbar = K % self._fft_mesh
        n1, n2, n3 = self._fft_mesh
        return ((i_q * n3 + Kbar[:, 2]) * n2 + Kbar[:, 1]) * n1 + Kbar[:, 0]

    def sum_over_qpoints(self, phi: NDArray) -> NDArray[np.cdouble]:
        """Return sum_q exp(2 pi i q . l) phi_q(m) in the order of get_grid_points.

        With q labeled by n_q and l by n_l (see qpoints), the sum is a DFT of the
        shape diag(d) from n_q to n_l.

        Parameters
        ----------
        phi : ndarray
            phi_q(m) from get_phi.
            dtype='complex128', shape=(N_c, N^p_3, N^p_2, N^p_1)

        Returns
        -------
        ndarray
            Values at the grid points, the lattice point running slowest and
            m_1 fastest.
            dtype='complex128', shape=(N_c * prod(fft_mesh),)

        """
        values = scipy.fft.ifftn(
            phi.reshape(*self._snf_diag, -1),
            axes=(0, 1, 2),
            norm="forward",
            workers=-1,
        )
        return values.ravel()


def _get_reciprocal_lattice_vectors(mesh: Sequence[int] | NDArray) -> NDArray[np.int64]:
    """Return the g-point g in the box for each source g mod N^s.

    g are the g-points, the points of the reciprocal lattice of the supercell
    in the integer coordinates of its basis.  For each source g mod N^s, the g
    in the box -ceil(N^s_k / 2) + 1 <= g_k <= floor(N^s_k / 2) is selected,
    which is used with g_point_selection="box".

    Parameters
    ----------
    mesh : array_like
        FFT mesh of the supercell (N^s_1, N^s_2, N^s_3).
        dtype='int64', shape=(3,)

    Returns
    -------
    ndarray
        (g_1, g_2, g_3) of the selected g.  The row

            j = (i_3 N^s_2 + i_2) N^s_1 + i_1

        is the g with g_k = i_k mod N^s_k (0 <= i_k < N^s_k), with g_k = i_k
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


def _bin_sorted_coefficients_python(
    coefficients: list[NDArray],
    g_sorted: NDArray[np.int64],
    source_weight: NDArray[np.double],
    bin_offsets: NDArray[np.int64],
    rotation: NDArray,
    translation: NDArray,
) -> list[NDArray[np.cdouble]]:
    """Return the sums of the coefficients over g' sorted by bin.

    The pure-Python counterpart of bin_sorted_supercell_fourier_coefficients
    of phonors.  For each g' of g_sorted and g = W^T g', the coefficient at
    the source s of g, times source_weight[s] and exp(-2 pi i g' . w), is
    added to the bin of g' given by bin_offsets.  The g' of a bin are added
    in the order of g_sorted by np.add.at.

    Parameters
    ----------
    coefficients : list of ndarray
        See FourierSeriesOnGrid.get_phi.
        dtype='complex128', shape=(N^s_3, N^s_2, N^s_1) each
    g_sorted, bin_offsets : ndarray
        See FourierSeriesOnGrid._get_g_points_sorted_by_bin.
    source_weight : ndarray
        See _MeshData.
    rotation : ndarray
        Rotation W in the supercell basis vectors.
        dtype='int64', shape=(3, 3)
    translation : ndarray
        Translation w in the supercell basis vectors.
        dtype='double', shape=(3,)

    Returns
    -------
    list of ndarray
        F_q(Kbar) flattened.
        dtype='complex128', shape=(len(bin_offsets) - 1,) each

    """
    supercell_mesh = np.array(coefficients[0].shape[::-1], dtype="int64")
    # g = W^T g' in column vectors is g' @ W in row vectors.
    g = g_sorted @ np.array(rotation, dtype="int64")
    source = _get_source(g, supercell_mesh)
    num_bins = len(bin_offsets) - 1
    bins = np.repeat(np.arange(num_bins), np.diff(bin_offsets))
    phase = np.exp(-2j * np.pi * g_sorted @ np.asarray(translation, dtype="double"))
    Fs = []
    for c in coefficients:
        w = c.ravel()[source] * source_weight[source] * phase
        F = np.zeros(num_bins, dtype="complex128")
        np.add.at(F, bins, w)
        Fs.append(F)
    return Fs


def _g_points_closed_under_rotation_python(
    g: NDArray[np.int64],
    source_offsets: NDArray[np.int64],
    supercell_mesh: NDArray[np.int64],
    rotation: NDArray,
) -> bool:
    """Return whether W^-T maps the selected g onto themselves.

    The pure-Python counterpart of g_points_closed_under_rotation of phonors:
    for every selected g, W^-T g has to be one of the selected g with the source
    of W^-T g.  source_offsets is that of _MeshData.

    """
    # g' = W^-T g in column vectors is g @ W^-1 in row vectors.
    g_rotated = g @ np.array(np.rint(np.linalg.inv(rotation)), dtype="int64")
    source = _get_source(g_rotated, supercell_mesh)
    members = {
        s: {tuple(v) for v in g[source_offsets[s] : source_offsets[s + 1]]}
        for s in np.unique(source)
    }
    return all(tuple(v) in members[s] for v, s in zip(g_rotated, source, strict=True))


def _get_source(g: NDArray[np.int64], supercell_mesh: NDArray) -> NDArray[np.int64]:
    """Return the index of g mod N^s in the flattened coefficient array.

    The source of g, ((g_3 mod N^s_3) N^s_2 + (g_2 mod N^s_2)) N^s_1
    + (g_1 mod N^s_1), for each g.

    """
    mesh = np.asarray(supercell_mesh, dtype="int64")
    r = g % mesh
    return (r[:, 2] * mesh[1] + r[:, 1]) * mesh[0] + r[:, 0]


def get_shortest_g_points(
    mesh: Sequence[int] | NDArray,
    supercell_lattice: NDArray,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the shortest g + N^s t for each source g mod N^s.

    For each reciprocal lattice vector g of _get_reciprocal_lattice_vectors, in
    the integer coordinates of the reciprocal basis of the supercell, the
    vectors g + N^s t with the shortest |G| are selected, where G = sum_k g_k b_k
    with the reciprocal basis vectors b_k of the supercell (without the factor
    2 pi).  The coefficient is divided equally among the vectors of the same
    length.

    These are the points of the mesh in the Brillouin zone of the period
    lattice {N^s t}, including those on its surface, and they are computed by
    phonors.bz_grid_addresses with the mesh N^s.  phonors takes two squared
    lengths as equal when they differ by less than 0.01 max_i |r_i|^2 /
    (N^s_i)^2, computed from the basis vectors r_i of the period lattice that
    it is given.  For a skewed basis of the supercell, N^s_k b_k are long and
    this tolerance can be larger than the difference to the next longer
    vectors, so a Delaunay-reduced basis, the rows of T B with the rows
    N^s_k b_k of B and the transformation matrix T (see _get_delaunay_tmat),
    is given with (T^T)^-1.  Then the mesh and the returned g stay in the
    coordinates g.

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
        Selected g-points in the integer coordinates (g_1, g_2, g_3), in the
        order of their sources.  The source of g is the row j of
        _get_reciprocal_lattice_vectors with the same g mod N^s,
        ((g_3 mod N^s_3) N^s_2 + (g_2 mod N^s_2)) N^s_1 + (g_1 mod N^s_1).
        dtype='int64', shape=(num, 3)
    source_offsets : ndarray
        The g with the source s are g[source_offsets[s]:source_offsets[s + 1]].
        This is bz_map of phonors.bz_grid_addresses with the dense format.
        dtype='int64', shape=(N^s_1 N^s_2 N^s_3 + 1,)

    """
    mesh = np.array(mesh, dtype="int64")
    period_lattice = np.diag(mesh) @ np.linalg.inv(supercell_lattice).T
    tmat = _get_delaunay_tmat(period_lattice)
    tmat_inv_t = np.array(np.rint(np.linalg.inv(tmat.T)), dtype="int64")
    addresses, bz_map, _ = bz_grid_addresses(
        mesh,
        tmat_inv_t,
        np.zeros(3, dtype="int64"),
        np.array((tmat @ period_lattice).T, dtype="double", order="C"),
        2,
    )
    return np.array(addresses, dtype="int64"), np.array(bz_map, dtype="int64")


def _get_delaunay_tmat(period_lattice: NDArray) -> NDArray[np.int64]:
    """Return the transformation matrix T with T period_lattice Delaunay-reduced.

    Parameters
    ----------
    period_lattice : ndarray
        Basis vectors N^s_k b_k of the period lattice in row vectors.
        dtype='double', shape=(3, 3)

    Returns
    -------
    ndarray
        Unimodular transformation matrix T.
        dtype='int64', shape=(3, 3)

    """
    reduced = get_reduced_bases(period_lattice, method="delaunay")
    tmat = np.array(np.rint(reduced @ np.linalg.inv(period_lattice)), dtype="int64")
    if abs(determinant(tmat)) != 1 or not np.allclose(tmat @ period_lattice, reduced):
        raise RuntimeError("Delaunay reduction of the period lattice failed.")
    return tmat
