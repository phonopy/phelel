"""Tests of the exact evaluation of Fourier series on the grid points."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.typing import NDArray

from phelel.base.fourier_series import (
    FourierSeriesOnGrid,
    _get_box,
    get_fourier_coefficients,
    get_shortest_representatives,
)
from phelel.base.local_potential import get_grid_points

# (p2s_matrix, mesh shape (N^s_3, N^s_2, N^s_1) of the supercell, fft_mesh,
#  rotation, translation, basis vectors of the supercell in rows)
# fcc primitive cell in the conventional cubic supercell (D = 4), diag(3, 1, 1)
# (D = 3, q points with q != -q mod 1), diag(2, 2, 3) (D = 12, two diagonal
# elements of the Smith normal form larger than one), and a hexagonal six-fold
# rotation that maps the box of the coefficients outside itself.  Each mesh has
# even and odd numbers of points, so the coefficients at N^s_k / 2 are included.
# The supercells are the cube of fcc, diag(3, 1, 1) and diag(2, 2, 3) of the
# simple cubic cell, and 2 x 2 x 1 of the hexagonal cell (a = 1, c = 1.6), which
# the rotations leave invariant.
_cubic = np.eye(3)
_hexagonal = np.array([[2, 0, 0], [-1, np.sqrt(3), 0], [0, 0, 1.6]])
_cases = {
    "fcc-identity": (
        [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
        (6, 5, 4),
        [3, 4, 5],
        None,
        None,
        _cubic,
    ),
    "fcc-threefold": (
        [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
        (6, 5, 4),
        [3, 4, 5],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [0.25, 0.25, 0.25],
        _cubic,
    ),
    "fcc-mirror": (
        [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
        (6, 5, 4),
        [3, 4, 5],
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        [0.25, 0.0, 0.5],
        _cubic,
    ),
    "diag311-fourfold": (
        [[3, 0, 0], [0, 1, 0], [0, 0, 1]],
        (5, 6, 8),
        [3, 5, 5],
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        [1 / 3, 0.0, 0.5],
        np.diag([3.0, 1.0, 1.0]),
    ),
    "diag223-fourfold": (
        [[2, 0, 0], [0, 2, 0], [0, 0, 3]],
        (7, 6, 6),
        [4, 4, 4],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [0.5, 0.0, 1 / 3],
        np.diag([2.0, 2.0, 3.0]),
    ),
    "hexagonal-sixfold": (
        [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        (5, 6, 6),
        [4, 4, 3],
        [[1, -1, 0], [1, 0, 0], [0, 0, 1]],
        [0.0, 0.0, 0.5],
        _hexagonal,
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


def _direct_sum_shortest(coefs: NDArray, x: NDArray, lattice: NDArray) -> NDArray:
    """Return the Fourier series with the shortest g + N^s n, written independently.

    For each g of the box, g + N^s n with n_k in -3, ..., 3 are compared by the
    length of the Cartesian reciprocal lattice vector g @ inv(lattice).T, and
    the coefficient is divided equally among the shortest ones.

    """
    n3, n2, n1 = coefs.shape
    mesh = np.array([n1, n2, n3])
    reciprocal = np.linalg.inv(lattice).T
    shifts = np.array(list(itertools.product(range(-3, 4), repeat=3))) * mesh
    ranges = [range(-((n + 1) // 2) + 1, n // 2 + 1) for n in (n1, n2, n3)]
    values = np.zeros(len(x), dtype="complex128")
    for g in itertools.product(*ranges):
        candidates = np.array(g) + shifts
        lengths = np.linalg.norm(candidates @ reciprocal, axis=1)
        shortest = candidates[lengths < lengths.min() * (1 + 1e-8) + 1e-12]
        f_g = coefs[g[2] % n3, g[1] % n2, g[0] % n1]
        values += f_g / len(shortest) * np.exp(2j * np.pi * x @ shortest.T).sum(axis=1)
    return values


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


@pytest.mark.parametrize("representatives", ["box", "shortest"])
@pytest.mark.parametrize("case", _cases)
def test_evaluate_equals_direct_sum(case: str, representatives: str):
    """Values at the grid points are the Fourier series summed term by term.

    For g = {W|w}, f(g^-1 x) is evaluated at x' = W^-1 (x - w).

    """
    p2s, shape, fft_mesh, rotation, translation, lattice = _cases[case]
    p2s = np.array(p2s, dtype="int64")
    coefs = [_random_coefficients(shape, seed) for seed in (1, 2)]
    fsg = FourierSeriesOnGrid(
        fft_mesh, p2s, representatives=representatives, supercell_lattice=lattice
    )
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
        if representatives == "box":
            ref = _direct_sum(c, x)
        else:
            ref = _direct_sum_shortest(c, x, lattice)
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
    p2s, shape, fft_mesh, rotation, translation, _ = _cases[case]
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


def _get_lattice_symmetries(lattice: NDArray, shape: tuple[int, int, int]):
    """Return W with entries in {-1, 0, 1} that keep the lattice and N^s Z^3.

    W acts on the fractional coordinates of the supercell, x -> W x, and keeps
    the metric, W^T A A^T W = A A^T with the basis vectors in the rows of A.
    g -> W^-T g keeps N^s Z^3 when diag(N^s)^-1 W^-T diag(N^s) is an integer
    matrix.

    """
    metric = lattice @ lattice.T
    mesh = np.array(shape[::-1])
    rotations = []
    for entries in itertools.product((-1, 0, 1), repeat=9):
        W = np.array(entries).reshape(3, 3)
        if not np.allclose(W.T @ metric @ W, metric):
            continue
        WinvT = np.linalg.inv(W).T
        m = np.diag(1 / mesh) @ WinvT @ np.diag(mesh)
        if np.allclose(m, np.rint(m)):
            rotations.append(W)
    return rotations


def _sorted_rows(a: NDArray) -> NDArray:
    return a[np.lexsort(a.T[::-1])]


@pytest.mark.parametrize(
    "shape,lattice,num_rotations",
    [((6, 6, 6), _cubic, 48), ((5, 6, 6), _hexagonal, 24)],
)
def test_shortest_representatives_are_invariant_under_rotations(
    shape: tuple[int, int, int], lattice: NDArray, num_rotations: int
):
    """The shortest g + N^s n with their weights are mapped to themselves.

    g -> W^-T g by the rotations W that keep the lattice and N^s Z^3, so the
    series with the shortest g is invariant under W when the coefficients are.
    The box is not invariant, because for even N^s_k it has g_k = +N^s_k / 2
    but not -N^s_k / 2, and dividing the coefficients among g of the same
    length is needed for the invariance.

    """
    rotations = _get_lattice_symmetries(lattice, shape)
    assert len(rotations) == num_rotations
    g, _, weight = get_shortest_representatives(shape, lattice)
    assert (weight < 1).any()
    ref = _sorted_rows(np.c_[g, weight])
    g_box = _get_box(shape)
    ref_box = _sorted_rows(g_box)
    is_box_invariant = []
    for W in rotations:
        WinvT = np.array(np.rint(np.linalg.inv(W).T), dtype="int64")
        np.testing.assert_array_equal(_sorted_rows(np.c_[g @ WinvT.T, weight]), ref)
        is_box_invariant.append(np.array_equal(_sorted_rows(g_box @ WinvT.T), ref_box))
    assert not all(is_box_invariant)


@pytest.mark.parametrize(
    "shape,fft_mesh,on_mesh",
    [((8, 8, 8), [4, 4, 4], True), ((6, 5, 4), [3, 4, 5], False)],
)
def test_shortest_equals_box_on_mesh(
    shape: tuple[int, int, int], fft_mesh: list[int], on_mesh: bool
):
    """The choice of g changes the values only off the FFT mesh of the supercell.

    The grid points are on the mesh when (N^p)^-1 (M^s)^-T N^s is an integer
    matrix, and then g + N^s n give the same values.

    """
    p2s = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype="int64")
    C = np.diag(1 / np.array(fft_mesh)) @ np.linalg.inv(p2s).T @ np.diag(shape[::-1])
    assert np.allclose(C, np.rint(C)) == on_mesh
    coefs = [_random_coefficients(shape, 5)]
    rotation = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    translation = [0.25, 0.25, 0.25]
    values = []
    for representatives in ("box", "shortest"):
        fsg = FourierSeriesOnGrid(
            fft_mesh, p2s, representatives=representatives, supercell_lattice=_cubic
        )
        values.append(
            fsg.evaluate(coefs, rotation=rotation, translation=translation)[0]
        )
    diff = np.abs(values[1] - values[0]).max() / np.abs(values[0]).max()
    if on_mesh:
        assert diff < 1e-12
    else:
        assert diff > 1e-3


def test_representatives_errors():
    """Unknown representatives and "shortest" without the lattice are errors."""
    p2s = np.eye(3, dtype="int64")
    with pytest.raises(ValueError, match="representatives has to be"):
        FourierSeriesOnGrid([2, 2, 2], p2s, representatives="nearest")  # type: ignore
    with pytest.raises(ValueError, match="supercell_lattice is needed"):
        FourierSeriesOnGrid([2, 2, 2], p2s, representatives="shortest")
