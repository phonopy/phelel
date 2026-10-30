"""Calculation of lattice points."""

import numpy as np
from phonors import snf3x3


def get_lattice_points(p2s_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the lattice points of the primitive cell modulo the supercell.

    The basis vectors of the primitive cell A^p and those of the supercell
    A^s are related by A^s = A^p M (basis vectors as columns, M =
    p2s_matrix).  A lattice point is given by the integer vector l of its
    coordinates with respect to the basis vectors of the primitive cell, at
    A^p l in Cartesian coordinates.  l and l + M t (t an integer vector)
    differ by a lattice vector of the supercell, and one l is returned from
    each such class, D = |det M| of them.

    With the Smith normal form P M Q = diag(d), the l are P^-1 n_l for the
    integer vectors n_l with 0 <= (n_l)_k < d_k, in the order of
    np.ndindex(d) ((n_l)_3 runs fastest).  The l are not reduced into the
    supercell: their coordinates M^-1 l with respect to the basis vectors of
    the supercell can be outside [0, 1).

    Parameters
    ----------
    p2s_matrix : ndarray
        Integer matrix M with A^s = A^p M.
        dtype='int64', shape=(3, 3)

    Returns
    -------
    lattice_points : ndarray
        l in the basis vectors of the primitive cell.
        dtype='int64', shape=(D, 3)
    multi : ndarray
        Diagonal elements d of the Smith normal form, with prod(d) = D.
        dtype='int64', shape=(3,)

    """
    multi, P, _ = snf3x3(np.array(p2s_matrix, dtype="int64"))
    P_inv = np.rint(np.linalg.inv(P)).astype(int)
    lattice_points = np.array(
        np.dot(list(np.ndindex(tuple(multi))), P_inv.T), dtype="int_", order="C"
    )
    return lattice_points, multi


def get_p2s_matrix(primitive_matrix: np.ndarray) -> np.ndarray:
    """Return the matrix from the primitive cell to the supercell.

    Parameters
    ----------
    primitive_matrix : ndarray
        Basis vectors of the primitive cell in those of the supercell, as
        Primitive.primitive_matrix, i.e., (a_p, b_p, c_p) = (a_s, b_s, c_s)
        primitive_matrix.  Note that Phonopy.primitive_matrix is relative to
        the unit cell instead.
        shape=(3, 3)

    Returns
    -------
    ndarray
        Inverse of primitive_matrix, which has to be an integer matrix.
        dtype='int64', shape=(3, 3)

    """
    p2s_matrix = np.linalg.inv(primitive_matrix)
    p2s_matrix_int = np.rint(p2s_matrix).astype("int64")
    if not (abs(p2s_matrix_int - p2s_matrix) < 1e-5).all():
        raise ValueError(
            "The supercell is not a multiple of the primitive cell: "
            f"{p2s_matrix.tolist()}."
        )
    return p2s_matrix_int
