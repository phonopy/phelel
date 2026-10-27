"""Tests of the exact evaluation of Fourier series on the grid points."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.typing import NDArray

from phelel.base.fourier_series import FourierSeriesOnGrid, get_fourier_coefficients
from phelel.base.local_potential import get_grid_points

# (p2s_matrix, mesh shape (N^s_3, N^s_2, N^s_1) of the supercell, fft_mesh,
#  rotation, translation)
# fcc primitive cell in the conventional cubic supercell (D = 4), diag(3, 1, 1)
# (D = 3, q points with q != -q mod 1), diag(2, 2, 3) (D = 12, two diagonal
# elements of the Smith normal form larger than one), and a hexagonal six-fold
# rotation that maps the box of the coefficients outside itself.  Each mesh has
# even and odd numbers of points, so the coefficients at N^s_k / 2 are included.
_cases = {
    "fcc-identity": (
        [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
        (6, 5, 4),
        [3, 4, 5],
        None,
        None,
    ),
    "fcc-threefold": (
        [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
        (6, 5, 4),
        [3, 4, 5],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [0.25, 0.25, 0.25],
    ),
    "fcc-mirror": (
        [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
        (6, 5, 4),
        [3, 4, 5],
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        [0.25, 0.0, 0.5],
    ),
    "diag311-fourfold": (
        [[3, 0, 0], [0, 1, 0], [0, 0, 1]],
        (5, 6, 8),
        [3, 5, 5],
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        [1 / 3, 0.0, 0.5],
    ),
    "diag223-fourfold": (
        [[2, 0, 0], [0, 2, 0], [0, 0, 3]],
        (7, 6, 6),
        [4, 4, 4],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [0.5, 0.0, 1 / 3],
    ),
    "hexagonal-sixfold": (
        [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        (5, 6, 6),
        [4, 4, 3],
        [[1, -1, 0], [1, 0, 0], [0, 0, 1]],
        [0.0, 0.0, 0.5],
    ),
}


def _random_coefficients(shape: tuple[int, int, int], seed: int) -> NDArray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _direct_sum(coefs: NDArray, x: NDArray) -> NDArray:
    """Return sum_g f(g) exp(2 pi i g . x) over the box, written independently.

    coefs[g_3 mod N^s_3, g_2 mod N^s_2, g_1 mod N^s_1] is f(g), and g_k runs
    over -ceil(N^s_k / 2) + 1, ..., floor(N^s_k / 2).

    """
    n3, n2, n1 = coefs.shape
    ranges = [range(-((n + 1) // 2) + 1, n // 2 + 1) for n in (n1, n2, n3)]
    g = np.array(list(itertools.product(*ranges)), dtype="int64")
    assert len(g) == coefs.size
    f_g = coefs[g[:, 2] % n3, g[:, 1] % n2, g[:, 0] % n1]
    return np.exp(2j * np.pi * x @ g.T) @ f_g


def _assert_off_mesh_along_even_axes(shape: tuple[int, int, int], x: NDArray):
    """Assert that some x are off the mesh along each axis with even N^s_k.

    g_k = +N^s_k / 2 and -N^s_k / 2 give the same values at x with integer
    N^s_k x_k, so the assignment of the coefficients at N^s_k / 2 is checked
    only with x off the mesh.

    """
    for k, n in enumerate(shape[::-1]):
        if n % 2 == 0:
            nx = n * x[:, k]
            assert (abs(nx - np.rint(nx)) > 1e-8).any()


@pytest.mark.parametrize("case", _cases)
def test_evaluate_equals_direct_sum(case: str):
    """Values at the grid points are the Fourier series summed term by term.

    For g = {W|w}, f(g^-1 x) is evaluated at x' = W^-1 (x - w).

    """
    p2s, shape, fft_mesh, rotation, translation = _cases[case]
    p2s = np.array(p2s, dtype="int64")
    coefs = [_random_coefficients(shape, seed) for seed in (1, 2)]
    fsg = FourierSeriesOnGrid(fft_mesh, p2s)
    grid_points, lattice_points = get_grid_points(fft_mesh, p2s)
    np.testing.assert_array_equal(fsg.lattice_points, lattice_points)

    x = grid_points
    if rotation is not None:
        x = (x - translation) @ np.linalg.inv(rotation).T
    elif translation is not None:
        x = x - translation
    _assert_off_mesh_along_even_axes(shape, x)
    values = fsg.evaluate(coefs, rotation=rotation, translation=translation)
    for c, v in zip(coefs, values, strict=True):
        ref = _direct_sum(c, x)
        np.testing.assert_allclose(v, ref, rtol=0, atol=1e-10 * np.abs(ref).max())


def test_get_fourier_coefficients():
    """The Fourier series of the coefficients gives back the values on the mesh."""
    shape = (4, 5, 6)
    values = _random_coefficients(shape, 3)
    coefs = get_fourier_coefficients(values)
    m3, m2, m1 = np.indices(shape).reshape(3, -1)
    x = np.c_[m1 / shape[2], m2 / shape[1], m3 / shape[0]]
    np.testing.assert_allclose(_direct_sum(coefs, x), values.ravel(), atol=1e-12)


@pytest.mark.parametrize("case", ["fcc-identity", "diag311-fourfold"])
def test_qpoints(case: str):
    """The D q points are distinct, and D q are integer vectors in [0, D)."""
    p2s = np.array(_cases[case][0], dtype="int64")
    D = abs(round(np.linalg.det(p2s)))
    qpoints = FourierSeriesOnGrid([2, 2, 2], p2s).qpoints
    Dq = np.rint(qpoints * D).astype(int)
    assert qpoints.shape == (D, 3)
    np.testing.assert_allclose(qpoints * D, Dq, atol=1e-12)
    assert ((Dq >= 0) & (Dq < D)).all()
    assert len(np.unique(Dq, axis=0)) == D


@pytest.mark.parametrize("case", _cases)
def test_lattice_fourier_sum_equals_D_phi(case: str):
    """sum_l f(x_lm) exp(-2 pi i q . l) = D phi_q(m).

    f(x_lm) are computed by the direct sum, so that this does not depend on
    sum_over_qpoints.  With diag(3, 1, 1), q = 1/3 and 2/3 are not equal to -q
    mod 1, so the sign of q is checked.

    """
    p2s, shape, fft_mesh, rotation, translation = _cases[case]
    p2s = np.array(p2s, dtype="int64")
    coefs = _random_coefficients(shape, 4)
    fsg = FourierSeriesOnGrid(fft_mesh, p2s)
    grid_points, lattice_points = get_grid_points(fft_mesh, p2s)
    D = len(lattice_points)

    x = grid_points
    if rotation is not None:
        x = (x - translation) @ np.linalg.inv(rotation).T
    _assert_off_mesh_along_even_axes(shape, x)
    values = _direct_sum(coefs, x).reshape(D, -1)
    lattice_sum = np.exp(-2j * np.pi * fsg.qpoints @ lattice_points.T) @ values
    (phi,) = fsg.get_phi([coefs], rotation=rotation, translation=translation)
    np.testing.assert_allclose(
        lattice_sum,
        D * phi.reshape(D, -1),
        rtol=0,
        atol=1e-10 * np.abs(lattice_sum).max(),
    )
