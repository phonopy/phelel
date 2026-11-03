"""Element-wise steps of the evaluation of the change of the local potential.

The derivative of the local potential with respect to the displacement of an
atom is computed by the finite-difference method in
LocalPotentialInterpolationNUFFT._run_at_atom.  For each pair of a displacement
and a symmetry operation, the change of the potential is evaluated on the
output grid, transformed in spin space, and added to the right-hand side of the
normal equations.  The functions here are the element-wise steps of this
computation, in the order in which they are called.

Each step has two implementations with the same name, arguments and order of
the loops: one in Rust in phonors (src/potential_change.rs, in the same order
as here), and one in pure Python, which is the reference of the Rust one.  The
function without the underscore selects one of them by ``lang``.  Every
function writes into its first argument.

"""

from __future__ import annotations

from typing import Literal

import numpy as np
import phonors
from numpy.typing import NDArray


def multiply_by_phase(
    phi: NDArray[np.cdouble],
    phase: NDArray[np.cdouble],
    lang: Literal["Rust", "Python"] = "Rust",
) -> None:
    """Multiply phi by phase element by element, in place.

    This is the multiplication of phi_q(m) by exp(2 pi i q . (N^p)^-1 m) in
    FourierSeriesOnGrid.get_phi.

    Parameters
    ----------
    phi : ndarray
        Values to be multiplied, overwritten by the products.
        dtype='complex128', C-contiguous
    phase : ndarray
        Phase factors.
        dtype='complex128', shape=phi.shape, C-contiguous
    lang : str, optional
        "Rust" (default) by phonors.multiply_by_phase or "Python".

    """
    if lang == "Rust":
        _multiply_by_phase_rust(phi, phase)
    else:
        _multiply_by_phase_python(phi, phase)


def _multiply_by_phase_rust(
    phi: NDArray[np.cdouble], phase: NDArray[np.cdouble]
) -> None:
    """Multiply phi by phase by phonors.multiply_by_phase."""
    phonors.multiply_by_phase(phi, phase)


def _multiply_by_phase_python(
    phi: NDArray[np.cdouble], phase: NDArray[np.cdouble]
) -> None:
    """Multiply phi by phase by NumPy."""
    phi *= phase
