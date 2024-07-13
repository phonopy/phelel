"""Utility functions related to data."""

from __future__ import annotations

import numpy as np


def cmplx2real(array: np.ndarray) -> np.ndarray:
    """View numpy array of complex128 by double.

    One complex value is viewed as a pair of real values, i.e.,

        a + 1j * b -> (a, b)

    """
    return array.view("double").reshape(array.shape + (2,))


def real2cmplx(array: np.ndarray) -> np.ndarray:
    """View numpy array of double by complex128.

    One complex value is viewed as a pair of real values, i.e.,

        a + 1j * b <- (a, b)

    """
    dtype = "c%d" % (np.dtype("double").itemsize * 2)
    return array.view(dtype).reshape(array.shape[:-1])
