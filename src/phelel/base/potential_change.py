"""Element-wise steps of the evaluation of the change of the local potential.

The derivative of the local potential with respect to the displacement of an
atom is computed by the finite-difference method in
DLocalPotentialFit._run_at_atom.  For each pair of a displacement
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
    """Call phonors.multiply_by_phase."""
    phonors.multiply_by_phase(phi, phase)


def _multiply_by_phase_python(
    phi: NDArray[np.cdouble], phase: NDArray[np.cdouble]
) -> None:
    """Multiply phi by phase by NumPy."""
    phi *= phase


def rotate_spin_components(
    values: list[NDArray[np.cdouble]],
    spin_rotation: NDArray[np.cdouble],
    lang: Literal["Rust", "Python"] = "Rust",
) -> None:
    """Rotate the four spin components in spin space, in place.

    At each element, the values [v_0, v_1, v_2, v_3] of the spin components,
    in the order [11, 12, 21, 22] of VASP, form the 2x2 matrix
    V = [[v_0, v_1], [v_2, v_3]], which is replaced by D V D^dagger with
    D = spin_rotation:

        v'_(2 i_sp + i_s)
            = sum_(i_s1, i_s2) D[i_sp, i_s1] conj(D[i_s, i_s2]) v_(2 i_s1 + i_s2).

    Parameters
    ----------
    values : list of ndarray
        Four spin components, overwritten by the rotated ones.
        dtype='complex128', the same shape, C-contiguous
    spin_rotation : ndarray
        Rotation matrix D of spinors, e.g., SpinorRotationMatrices.Delta.
        dtype='complex128', shape=(2, 2)
    lang : str, optional
        "Rust" (default) by phonors.rotate_spin_components or "Python".

    """
    if lang == "Rust":
        _rotate_spin_components_rust(values, spin_rotation)
    else:
        _rotate_spin_components_python(values, spin_rotation)


def _rotate_spin_components_rust(
    values: list[NDArray[np.cdouble]], spin_rotation: NDArray[np.cdouble]
) -> None:
    """Call phonors.rotate_spin_components."""
    phonors.rotate_spin_components(
        values, np.asarray(spin_rotation, dtype="complex128")
    )


def _rotate_spin_components_python(
    values: list[NDArray[np.cdouble]], spin_rotation: NDArray[np.cdouble]
) -> None:
    """Rotate the spin components by NumPy.

    The factors D[i_sp, i_s1] conj(D[i_s, i_s2]) are computed first, and the
    terms are summed in the order of (i_s1, i_s2), as in the Rust
    implementation.  The loops over (i_sp, i_s, i_s1, i_s2) are outside and
    the elements are handled by NumPy, whereas in Rust the loop over the
    elements is outside.

    """
    factors = np.zeros((2, 2, 2, 2), dtype="complex128")
    for i_sp in (0, 1):
        for i_s in (0, 1):
            for i_s1 in (0, 1):
                for i_s2 in (0, 1):
                    factors[i_sp, i_s, i_s1, i_s2] = (
                        spin_rotation[i_sp, i_s1] * spin_rotation[i_s, i_s2].conj()
                    )
    original = [value.copy() for value in values]
    for i_sp in (0, 1):
        for i_s in (0, 1):
            rotated = np.zeros_like(original[0])
            for i_s1 in (0, 1):
                for i_s2 in (0, 1):
                    term = factors[i_sp, i_s, i_s1, i_s2] * original[i_s1 * 2 + i_s2]
                    rotated += term
            values[i_sp * 2 + i_s][...] = rotated


def apply_time_reversal_to_spin_components(
    values: list[NDArray[np.cdouble]],
    lang: Literal["Rust", "Python"] = "Rust",
) -> None:
    """Apply time reversal to the spin components, in place.

    Time reversal is i sigma_y K, under which the 2x2 matrix
    V = [[v_0, v_1], [v_2, v_3]] of four spin components, in the order
    [11, 12, 21, 22] of VASP, becomes sigma_y V^* sigma_y:

        [v_0, v_1, v_2, v_3] -> [conj(v_3), -conj(v_2), -conj(v_1), conj(v_0)].

    This keeps the scalar part and flips the sign of the magnetic part.  For
    two components (collinear, up and down), the two are exchanged and
    conjugated, [v_0, v_1] -> [conj(v_1), conj(v_0)], and for one component it
    is only conjugated.

    Parameters
    ----------
    values : list of ndarray
        One, two or four spin components, overwritten by the time-reversed
        ones.
        dtype='complex128', the same shape, C-contiguous
    lang : str, optional
        "Rust" (default) by phonors.apply_time_reversal_to_spin_components or
        "Python".

    """
    if lang == "Rust":
        _apply_time_reversal_to_spin_components_rust(values)
    else:
        _apply_time_reversal_to_spin_components_python(values)


def _apply_time_reversal_to_spin_components_rust(
    values: list[NDArray[np.cdouble]],
) -> None:
    """Call phonors.apply_time_reversal_to_spin_components."""
    phonors.apply_time_reversal_to_spin_components(values)


def _apply_time_reversal_to_spin_components_python(
    values: list[NDArray[np.cdouble]],
) -> None:
    """Apply time reversal by NumPy."""
    original = [value.copy() for value in values]
    if len(values) == 4:
        values[0][...] = original[3].conj()
        values[1][...] = -original[2].conj()
        values[2][...] = -original[1].conj()
        values[3][...] = original[0].conj()
    elif len(values) == 2:
        values[0][...] = original[1].conj()
        values[1][...] = original[0].conj()
    else:
        values[0][...] = original[0].conj()


def add_to_normal_equation_rhs(
    normal_equation_rhs: NDArray[np.cdouble],
    displacement: NDArray[np.double],
    values: list[NDArray[np.cdouble]],
    lang: Literal["Rust", "Python"] = "Rust",
) -> None:
    """Add the outer products of displacement and the spin components, in place.

        normal_equation_rhs[c, i, i_grid] += displacement[i] * values[c][i_grid]

    for each spin component c, each Cartesian direction i of the displacement
    and each grid point i_grid.  In DLocalPotentialFit._run_at_atom,
    displacement is a rotated displacement R u_k and values are the spin
    components of the rotated change of the potential dV_kg at the grid
    points.  One call adds one term; the calls for all pairs (k, g) give the
    right-hand side U^T dV of the normal equations.

    Parameters
    ----------
    normal_equation_rhs : ndarray
        Right-hand side of the normal equations, to which the products are
        added.  The axes are the spin components, the Cartesian directions x,
        y, z of the displacement and the grid points.
        dtype='complex128', shape=(n_components, 3, n_grid_points),
        C-contiguous
    displacement : ndarray
        Displacement in Cartesian coordinates, R u_k in _run_at_atom.
        dtype='double', shape=(3,)
    values : list of ndarray
        n_components spin components of the potential change at the grid
        points, in the order [11, 12, 21, 22] of VASP for four components.
        dtype='complex128', shape=(n_grid_points,) each, C-contiguous
    lang : str, optional
        "Rust" (default) by phonors.add_to_normal_equation_rhs or "Python".

    """
    if lang == "Rust":
        _add_to_normal_equation_rhs_rust(normal_equation_rhs, displacement, values)
    else:
        _add_to_normal_equation_rhs_python(normal_equation_rhs, displacement, values)


def _add_to_normal_equation_rhs_rust(
    normal_equation_rhs: NDArray[np.cdouble],
    displacement: NDArray[np.double],
    values: list[NDArray[np.cdouble]],
) -> None:
    """Call phonors.add_to_normal_equation_rhs."""
    phonors.add_to_normal_equation_rhs(
        normal_equation_rhs, np.ascontiguousarray(displacement, dtype="double"), values
    )


def _add_to_normal_equation_rhs_python(
    normal_equation_rhs: NDArray[np.cdouble],
    displacement: NDArray[np.double],
    values: list[NDArray[np.cdouble]],
) -> None:
    """Add the outer products by NumPy."""
    for i_component, value in enumerate(values):
        normal_equation_rhs[i_component] += np.outer(displacement, value)


def apply_inverse_normal_matrix(
    normal_equation_rhs: NDArray[np.cdouble],
    inverse_normal_matrix: NDArray[np.double],
    lang: Literal["Rust", "Python"] = "Rust",
) -> None:
    """Multiply normal_equation_rhs by pinv(U^T U) along the direction axis.

        normal_equation_rhs[c, :, i_grid]
            <- inverse_normal_matrix @ normal_equation_rhs[c, :, i_grid]

    For each spin component c and each grid point i_grid, the three values
    normal_equation_rhs[c, :, i_grid], one for each Cartesian direction of the
    displacement, form a vector, which is replaced by its product with
    inverse_normal_matrix, in place.  The sum over the columns runs in their
    order, without BLAS.  In DLocalPotentialFit._run_at_atom,
    this turns the right-hand side U^T dV of the normal equations into dV/du.

    Parameters
    ----------
    normal_equation_rhs : ndarray
        Right-hand side of the normal equations, overwritten by the products,
        which are dV/du in _run_at_atom.  The axes are the spin components,
        the Cartesian directions x, y, z of the displacement and the grid
        points.
        dtype='complex128', shape=(n_components, 3, n_grid_points),
        C-contiguous
    inverse_normal_matrix : ndarray
        Pseudo-inverse pinv(U^T U) of the matrix U^T U of the normal
        equations, where the rows of U are the rotated displacements.
        dtype='double', shape=(3, 3)
    lang : str, optional
        "Rust" (default) by phonors.apply_inverse_normal_matrix or "Python".

    """
    if lang == "Rust":
        _apply_inverse_normal_matrix_rust(normal_equation_rhs, inverse_normal_matrix)
    else:
        _apply_inverse_normal_matrix_python(normal_equation_rhs, inverse_normal_matrix)


def _apply_inverse_normal_matrix_rust(
    normal_equation_rhs: NDArray[np.cdouble], inverse_normal_matrix: NDArray[np.double]
) -> None:
    """Call phonors.apply_inverse_normal_matrix."""
    phonors.apply_inverse_normal_matrix(
        normal_equation_rhs, np.asarray(inverse_normal_matrix, dtype="double")
    )


def _apply_inverse_normal_matrix_python(
    normal_equation_rhs: NDArray[np.cdouble], inverse_normal_matrix: NDArray[np.double]
) -> None:
    """Multiply by inverse_normal_matrix with NumPy, element-wise over i_grid."""
    for i_component in range(normal_equation_rhs.shape[0]):
        original = normal_equation_rhs[i_component].copy()
        for i in range(3):
            product = np.zeros_like(original[0])
            for j in range(3):
                product += inverse_normal_matrix[i, j] * original[j]
            normal_equation_rhs[i_component, i] = product
