"""Calculation of lattice points."""

import numpy as np
from phonopy.structure.cells import SNF3x3


def get_lattice_points(p2s_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return lattice points and D (diagonal elements) of SNF."""
    snf = SNF3x3(p2s_matrix)
    P_inv = np.rint(np.linalg.inv(snf.P)).astype(int)
    multi = np.diagonal(snf.D)
    lattice_points = np.array(
        np.dot(list(np.ndindex(tuple(multi))), P_inv.T), dtype="int_", order="C"
    )
    return lattice_points, multi
