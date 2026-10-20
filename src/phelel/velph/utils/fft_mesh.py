"""FFT mesh of VASP for a plane-wave cutoff."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Bohr radius in Angstrom and Rydberg in eV, the units in which VASP computes
# the FFT mesh.
_BOHR = 0.529177249
_RYDBERG = 13.605826


def get_vasp_fft_mesh(
    encut: float, lattice: NDArray, prec: str | None = None
) -> NDArray[np.int64]:
    """Return the FFT mesh (NGX, NGY, NGZ) of VASP for ENCUT and PREC.

    lattice gives the basis vectors as rows, in Angstrom, and encut is in eV.
    Each number of the mesh is even and is factorized by 2, 3, 5 and 7.  The
    factor of the cutoff is that of VASP 6, which depends on PREC: 4 for High,
    Accurate and Single, and 3 for Normal, SingleN and when PREC is not given.

    """
    lengths = np.linalg.norm(lattice, axis=1) / _BOHR
    cutoff = np.sqrt(encut / _RYDBERG) / (2 * np.pi / lengths)
    fft_mesh = np.rint(cutoff * _cutoff_factor(prec)).astype(int)
    for i in range(3):
        while not _is_factorized_by_2357(fft_mesh[i]):
            fft_mesh[i] += 1
    return fft_mesh


def _cutoff_factor(prec: str | None) -> int:
    """Return the factor of the cutoff for PREC, in the convention of VASP 6."""
    if prec is None or prec.lower() == "singlen":
        return 3
    if prec[0].lower() in ("h", "a", "s"):
        return 4
    return 3


def _is_factorized_by_2357(n: int) -> bool:
    """Return whether n is even and is factorized only by 2, 3, 5 and 7."""
    if n % 2 != 0:
        return False
    for div in (2, 3, 5, 7):
        while n % div == 0:
            n //= div
    return n == 1
