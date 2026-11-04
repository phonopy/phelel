"""Tests of the exact evaluation of Fourier series on the grid points."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from phonors import (
    bin_sorted_supercell_fourier_coefficients,
    g_points_closed_under_rotation,
)

import phelel.base.fourier_series as fourier_series
from phelel.base.fourier_series import (
    FourierSeriesOnGrid,
    _bin_sorted_coefficients_python,
    _g_points_closed_under_rotation_python,
    _get_reciprocal_lattice_vectors,
    get_fourier_coefficients,
    get_shortest_g_points,
)
from phelel.base.local_potential import get_grid_points

# (p2s_matrix, mesh shape (N^s_3, N^s_2, N^s_1) of the supercell, fft_mesh,
#  rotation, translation, basis vectors of the supercell in rows)
# fcc primitive cell in the conventional cubic supercell (N_c = 4), diag(3, 1, 1)
# (N_c = 3, q points with q != -q mod 1), diag(2, 2, 3) (N_c = 12, two diagonal
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
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [0.0, 0.0, 0.0],
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


@pytest.mark.parametrize("lang", ["Rust", "Python"])
@pytest.mark.parametrize("g_point_selection", ["box", "shortest"])
@pytest.mark.parametrize("case", _cases)
def test_evaluate_equals_direct_sum(case: str, g_point_selection: str, lang: str):
    """Values at the grid points are the Fourier series summed term by term.

    For g = {W|w}, f(g^-1 x) is evaluated at x' = W^-1 (x - w).

    """
    p2s, shape, fft_mesh, rotation, translation, lattice = _cases[case]
    p2s = np.array(p2s, dtype="int64")
    coefs = [_random_coefficients(shape, seed) for seed in (1, 2)]
    fsg = FourierSeriesOnGrid(
        fft_mesh,
        p2s,
        g_point_selection=g_point_selection,
        supercell_lattice=lattice,
        lang=lang,
    )
    grid_points, lattice_points = get_grid_points(fft_mesh, p2s)
    np.testing.assert_array_equal(fsg.lattice_points, lattice_points)

    x = (grid_points - translation) @ np.linalg.inv(rotation).T
    _assert_off_mesh_along_even_axes(shape, x)
    values = fsg.evaluate(coefs, rotation=rotation, translation=translation)
    for c, v in zip(coefs, values, strict=True):
        if g_point_selection == "box":
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
    """The N_c q points are distinct, and N_c q are integer vectors in [0, N_c)."""
    p2s = np.array(_cases[case][0], dtype="int64")
    num_cells = abs(round(np.linalg.det(p2s)))
    qpoints = FourierSeriesOnGrid([2, 2, 2], p2s).qpoints
    nc_q = np.rint(qpoints * num_cells).astype(int)
    assert qpoints.shape == (num_cells, 3)
    np.testing.assert_allclose(qpoints * num_cells, nc_q, atol=1e-12)
    assert ((nc_q >= 0) & (nc_q < num_cells)).all()
    assert len(np.unique(nc_q, axis=0)) == num_cells


@pytest.mark.parametrize("lang", ["Rust", "Python"])
@pytest.mark.parametrize("case", _cases)
def test_lattice_fourier_sum_equals_nc_phi(case: str, lang: str):
    """sum_l f(x_lm) exp(-2 pi i q . l) = N_c phi_q(m).

    f(x_lm) are computed by the direct sum, so that this does not depend on
    sum_over_qpoints.  With diag(3, 1, 1), q = 1/3 and 2/3 are not equal to -q
    mod 1, so the sign of q is checked.

    """
    p2s, shape, fft_mesh, rotation, translation, _ = _cases[case]
    p2s = np.array(p2s, dtype="int64")
    coefs = _random_coefficients(shape, 4)
    fsg = FourierSeriesOnGrid(fft_mesh, p2s, lang=lang)
    grid_points, lattice_points = get_grid_points(fft_mesh, p2s)
    num_cells = len(lattice_points)

    x = (grid_points - translation) @ np.linalg.inv(rotation).T
    _assert_off_mesh_along_even_axes(shape, x)
    values = _direct_sum(coefs, x).reshape(num_cells, -1)
    lattice_sum = np.exp(-2j * np.pi * fsg.qpoints @ lattice_points.T) @ values
    (phi,) = fsg.get_phi([coefs], rotation=rotation, translation=translation)
    np.testing.assert_allclose(
        lattice_sum,
        num_cells * phi.reshape(num_cells, -1),
        rtol=0,
        atol=1e-10 * np.abs(lattice_sum).max(),
    )


@pytest.mark.parametrize("g_point_selection", ["box", "shortest"])
@pytest.mark.parametrize("case", _cases)
def test_lang_rust_equals_python(case: str, g_point_selection: str):
    """F_q(Kbar) of lang="Rust" are those of lang="Python".

    Both sum the coefficients of each bin in the same order, so they are
    equal without the translation.  With the translation, the phase factors
    of Rust and NumPy may differ in the last bits.

    """
    p2s, shape, fft_mesh, rotation, translation, lattice = _cases[case]
    p2s = np.array(p2s, dtype="int64")
    coefs = [_random_coefficients(shape, seed) for seed in (5, 6)]
    fsg_rust, fsg_python = (
        FourierSeriesOnGrid(
            fft_mesh,
            p2s,
            g_point_selection=g_point_selection,
            supercell_lattice=lattice,
            lang=lang,
        )
        for lang in ("Rust", "Python")
    )
    Fs_rust = fsg_rust._bin_coefficients(coefs, rotation, np.zeros(3))
    Fs_python = fsg_python._bin_coefficients(coefs, rotation, np.zeros(3))
    for F_rust, F_python in zip(Fs_rust, Fs_python, strict=True):
        assert F_rust.shape == F_python.shape
        np.testing.assert_array_equal(F_rust, F_python)

    phis_rust = fsg_rust.get_phi(coefs, rotation=rotation, translation=translation)
    phis_python = fsg_python.get_phi(coefs, rotation=rotation, translation=translation)
    for phi_rust, phi_python in zip(phis_rust, phis_python, strict=True):
        np.testing.assert_allclose(
            phi_rust, phi_python, rtol=0, atol=1e-13 * np.abs(phi_python).max()
        )


def _get_lattice_symmetries(
    lattice: NDArray, shape: tuple[int, int, int] | None = None
) -> list[NDArray]:
    """Return W with entries in {-1, 0, 1} that keep the lattice (and N^s Z^3).

    W acts on the fractional coordinates of the supercell, x -> W x, and keeps
    the metric, W^T A A^T W = A A^T with the basis vectors in the rows of A.
    With shape, only the W are returned for which g -> W^-T g keeps N^s Z^3,
    i.e., diag(N^s)^-1 W^-T diag(N^s) is an integer matrix.

    """
    metric = lattice @ lattice.T
    rotations = []
    for entries in itertools.product((-1, 0, 1), repeat=9):
        W = np.array(entries).reshape(3, 3)
        if not np.allclose(W.T @ metric @ W, metric):
            continue
        if shape is not None:
            mesh = np.array(shape[::-1])
            WinvT = np.linalg.inv(W).T
            m = np.diag(1 / mesh) @ WinvT @ np.diag(mesh)
            if not np.allclose(m, np.rint(m)):
                continue
        rotations.append(W)
    return rotations


def _sorted_rows(a: NDArray) -> NDArray:
    return a[np.lexsort(a.T[::-1])]


def _get_shortest_g_points_with_source(
    mesh: Sequence[int], lattice: NDArray
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.double]]:
    """Return g of get_shortest_g_points with the source and weight of each g.

    The source computed from g is checked against source_offsets.

    """
    g, source_offsets = get_shortest_g_points(mesh, lattice)
    counts = np.diff(source_offsets)
    assert (counts > 0).all()
    source = np.repeat(np.arange(len(counts)), counts)
    np.testing.assert_array_equal(fourier_series._get_source(g, mesh), source)
    return g, source, 1.0 / counts[source]


@pytest.mark.parametrize(
    "shape,lattice,num_rotations",
    [((6, 6, 6), _cubic, 48), ((5, 6, 6), _hexagonal, 24)],
)
def test_shortest_g_points_are_invariant_under_rotations(
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
    g, _, weight = _get_shortest_g_points_with_source(shape[::-1], lattice)
    assert (weight < 1).any()
    ref = _sorted_rows(np.c_[g, weight])
    g_box = _get_reciprocal_lattice_vectors(shape[::-1])
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
    """The selection of g changes the values only off the FFT mesh of the supercell.

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
    for g_point_selection in ("box", "shortest"):
        fsg = FourierSeriesOnGrid(
            fft_mesh, p2s, g_point_selection=g_point_selection, supercell_lattice=_cubic
        )
        values.append(
            fsg.evaluate(coefs, rotation=rotation, translation=translation)[0]
        )
    diff = np.abs(values[1] - values[0]).max() / np.abs(values[0]).max()
    if on_mesh:
        assert diff < 1e-12
    else:
        assert diff > 1e-3


# Bases of the orthorhombic and hexagonal lattices multiplied by unimodular
# matrices, for which the Delaunay reduction changes the basis of the period
# lattice.
_skewed_orthorhombic = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]]) @ np.diag(
    [1.0, 1.3, 1.7]
)
_skewed_hexagonal = np.array([[2, 1, 0], [1, 1, 0], [0, 1, 1]]) @ _hexagonal
# A basis found by a random search, for which skipping the move of g into the
# centered cell of the reduced basis chooses a longer g.
_random_skewed = np.array(
    [
        [2.532413380062898, -2.573145484891704, 2.923962746453382],
        [3.5074710879688458, -0.49063708996215505, 0.9912863771012315],
        [4.243828401469066, -2.4855201785036973, 2.8482581228465507],
    ]
)
# Product of shears that makes a basis strongly skewed.
_shears = (
    np.array([[1, 3, 0], [0, 1, 0], [0, 0, 1]])
    @ np.array([[1, 0, 0], [0, 1, 4], [0, 0, 1]])
    @ np.array([[1, 0, 0], [0, 1, 0], [3, 2, 1]])
)


def _get_shortest_g_points_by_bound(
    shape: tuple[int, int, int], lattice: NDArray, tolerance: float = 1e-5
) -> tuple[NDArray, NDArray, NDArray]:
    """Return the shortest g + N^s n searched without reducing the basis.

    A candidate y = g + N^s n not longer than the longest |G| of the box has
    |y_k| <= bound = (longest |G|) / sqrt(lambda_min) with the smallest
    eigenvalue lambda_min of the metric, so |n_k| <= (bound + N^s_k / 2) / N^s_k.
    The number of n grows with the skewness of the basis vectors.

    """
    g_box = _get_reciprocal_lattice_vectors(shape[::-1])
    mesh = np.array(shape[::-1], dtype="int64")
    metric = np.linalg.inv(lattice @ lattice.T)

    def get_lengths(g: NDArray) -> NDArray:
        return np.sqrt(np.einsum("ij,jk,ik->i", g, metric, g))

    shortest = get_lengths(g_box)
    bound = shortest.max() * (1 + tolerance) / np.sqrt(np.linalg.eigvalsh(metric).min())
    n_max = ((bound + mesh / 2) // mesh).astype("int64")
    shifts = np.array(list(itertools.product(*[range(-n, n + 1) for n in n_max])))
    shifts = shifts * mesh
    for shift in shifts:
        shortest = np.minimum(shortest, get_lengths(g_box + shift))
    selected_g = []
    selected_source = []
    for shift in shifts:
        g = g_box + shift
        is_shortest = get_lengths(g) <= shortest * (1 + tolerance)
        selected_g.append(g[is_shortest])
        selected_source.append(np.nonzero(is_shortest)[0])
    g = np.concatenate(selected_g)
    source = np.concatenate(selected_source)
    multiplicity = np.bincount(source, minlength=len(g_box))
    return g, source, 1.0 / multiplicity[source]


@pytest.mark.parametrize(
    "shape,lattice",
    [
        ((6, 6, 6), _cubic),
        ((5, 6, 8), np.diag([3.0, 1.0, 1.0])),
        ((5, 6, 6), _hexagonal),
        ((5, 6, 4), _skewed_orthorhombic),
        ((6, 5, 6), _skewed_hexagonal),
    ],
)
def test_shortest_g_points_equal_search_without_reduction(
    shape: tuple[int, int, int], lattice: NDArray
):
    """The shortest g + N^s n equal those searched without the reduced basis.

    The coefficients divided among g of the same length are included.

    """
    g, source, weight = _get_shortest_g_points_with_source(shape[::-1], lattice)
    g_ref, source_ref, weight_ref = _get_shortest_g_points_by_bound(shape, lattice)
    assert (weight < 1).any()
    np.testing.assert_array_equal(
        _sorted_rows(np.c_[source, g, weight]),
        _sorted_rows(np.c_[source_ref, g_ref, weight_ref]),
    )


@pytest.mark.parametrize("lattice", [_cubic, _hexagonal])
def test_shortest_g_points_do_not_depend_on_basis(lattice: NDArray):
    """The shortest G with their weights are the same for a skewed basis.

    With the mesh (n, n, n), the period lattice {N^s n} of g is kept when the
    basis vectors A of the supercell are replaced by U A with a unimodular U,
    so the shortest G = g (U A)^-T and their weights are the same.  U is
    _shears, which makes the basis strongly skewed.

    """
    shape = (6, 6, 6)
    G_weight = []
    for A in (lattice, _shears @ lattice):
        g, _, weight = _get_shortest_g_points_with_source(shape[::-1], A)
        G = g @ np.linalg.inv(A).T
        G_weight.append(_sorted_rows(np.round(np.c_[G, weight], 8)))
    assert (G_weight[0][:, 3] < 1).any()
    np.testing.assert_allclose(G_weight[1], G_weight[0], atol=1e-8)


@pytest.mark.parametrize(
    "shape,lattice",
    [
        ((6, 4, 5), _random_skewed),
        ((6, 5, 4), _shears @ _cubic),
        ((5, 6, 6), _shears @ _hexagonal),
    ],
)
def test_shortest_g_points_are_closest(shape: tuple[int, int, int], lattice: NDArray):
    """No selected g becomes shorter by a Voronoi-relevant vector of the period lattice.

    The Voronoi-relevant vectors of a lattice with a Delaunay-reduced basis
    w_1, w_2, w_3 have coefficients in {-1, 0, 1} (Conway and Sloane, Proc. R.
    Soc. Lond. A 436, 55 (1992)), and y is the shortest of y + {N^s n} when no
    relevant vector v makes |G(y - v)| shorter than |G(y)|.  This holds
    independently of the range of the search.  The bases are strongly skewed,
    for which the search without the reduction is too slow to compare with.

    """
    mesh = np.array(shape[::-1])
    period_lattice = np.diag(mesh) @ np.linalg.inv(lattice).T
    W = fourier_series._get_delaunay_tmat(period_lattice) @ np.diag(mesh)
    v = np.array(list(itertools.product((-1, 0, 1), repeat=3))) @ W
    metric = np.linalg.inv(lattice @ lattice.T)
    g, _ = get_shortest_g_points(shape[::-1], lattice)
    lengths = np.sqrt(np.einsum("ij,jk,ik->i", g, metric, g))
    y = g[:, None, :] - v[None, :, :]
    others = np.sqrt(np.einsum("abj,jk,abk->ab", y, metric, y))
    assert (others.min(axis=1) >= lengths * (1 - 1e-8)).all()


@pytest.mark.parametrize("is_reduced", [True, False])
def test_shortest_g_points_path_superbase(
    is_reduced: bool, monkeypatch: pytest.MonkeyPatch
):
    """All the closest points are selected with the superbase of the path type.

    The period lattice {N^s n} is the cubic lattice with the basis
    w_1 = (-1, 1, 0), w_2 = (1, 0, -1), w_3 = (0, 0, 1) of N^s_k b_k, which with
    w_0 = (0, -1, 0) is an obtuse superbase of the path type.  G of
    g = (N/2, 0, N/2) is (w_1 + w_3) / 2 = (-1/2, 1/2, 1/2), the center of a cube
    of the lattice, so the shortest G are the eight (+-1/2, +-1/2, +-1/2).  One
    of them is G - w_1 - w_2 - 2 w_3, with the coefficient 2 outside
    {-1, 0, 1}.  The search is made in this basis when the reduction, which
    gives an orthogonal basis, is replaced by the identity.

    """
    if not is_reduced:
        monkeypatch.setattr(
            fourier_series,
            "_get_delaunay_tmat",
            lambda _: np.eye(3, dtype="int64"),
        )
    n = 4
    shape = (n, n, n)
    path = np.array([[-1.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    reciprocal = path / n
    lattice = np.linalg.inv(reciprocal).T
    g, source, weight = _get_shortest_g_points_with_source(shape[::-1], lattice)
    i = np.nonzero(
        (_get_reciprocal_lattice_vectors(shape[::-1]) == [n // 2, 0, n // 2]).all(
            axis=1
        )
    )[0][0]
    np.testing.assert_array_equal(weight[source == i], 1 / 8)
    G = g[source == i] @ reciprocal
    expected = np.array(list(itertools.product([-0.5, 0.5], repeat=3)))
    np.testing.assert_allclose(_sorted_rows(G), expected, atol=1e-12)


def test_g_point_selection_errors():
    """Unknown g_point_selection and "shortest" without the lattice are errors."""
    p2s = np.eye(3, dtype="int64")
    with pytest.raises(ValueError, match="g_point_selection has to be"):
        FourierSeriesOnGrid([2, 2, 2], p2s, g_point_selection="nearest")  # type: ignore
    with pytest.raises(ValueError, match="supercell_lattice is needed"):
        FourierSeriesOnGrid([2, 2, 2], p2s, g_point_selection="shortest")


def test_lang_errors():
    """Unknown lang is an error."""
    p2s = np.eye(3, dtype="int64")
    with pytest.raises(ValueError, match="lang has to be"):
        FourierSeriesOnGrid([2, 2, 2], p2s, lang="C")  # type: ignore


def _bin_by_definition(
    fsg: FourierSeriesOnGrid,
    coefficients: list[NDArray],
    rotation: NDArray,
    translation: NDArray,
) -> list[NDArray]:
    """Return F_q(Kbar) summed term by term in the order of the selected g.

    The coefficient of each selected g, times its weight and
    exp(-2 pi i g' . w), is added to the bin of g' = W^-T g by np.add.at in
    the order of the selected g, without sorting.

    """
    supercell_mesh = coefficients[0].shape[::-1]
    if fsg._g_point_selection == "box":
        g = _get_reciprocal_lattice_vectors(supercell_mesh)
        source = np.arange(len(g))
        weight = np.ones(len(g))
    else:
        g, source, weight = _get_shortest_g_points_with_source(
            supercell_mesh, fsg._supercell_lattice
        )
    g = g @ np.array(np.rint(np.linalg.inv(rotation)), dtype="int64")
    i_F = fsg._get_bin_indices(g)
    phase = np.exp(-2j * np.pi * g @ np.asarray(translation, dtype="double"))
    Fs = []
    for c in coefficients:
        w = c.ravel()[source] * weight * phase
        F = np.zeros(fsg._num_cells * np.prod(fsg._fft_mesh), dtype="complex128")
        np.add.at(F, i_F, w)
        Fs.append(F.reshape(fsg._num_cells, *fsg._fft_mesh[::-1]))
    return Fs


def _make_fsg(case: str, g_point_selection: str, lang: str = "Rust"):
    p2s, shape, fft_mesh, rotation, translation, lattice = _cases[case]
    fsg = FourierSeriesOnGrid(
        fft_mesh,
        np.array(p2s, dtype="int64"),
        g_point_selection=g_point_selection,
        supercell_lattice=lattice,
        lang=lang,
    )
    return fsg, shape, rotation, translation, lattice


@pytest.mark.parametrize("g_point_selection", ["box", "shortest"])
@pytest.mark.parametrize("case", _cases)
def test_sort_g_points_by_bin_rust_equals_python(case: str, g_point_selection: str):
    """sort_g_points_by_bin of phonors equals _sort_g_points_by_bin_python.

    The g' of each bin are those whose i_F is the bin, in the order of the
    selected g.

    """
    fsg, shape, rotation, _, _ = _make_fsg(case, g_point_selection)
    g = fsg._get_mesh_data(shape[::-1]).g
    for W in (np.eye(3, dtype="int64"), rotation):
        g_sorted, bin_offsets = fsg._sort_g_points_by_bin_rust(g, W)
        g_sorted_python, bin_offsets_python = fsg._sort_g_points_by_bin_python(g, W)
        np.testing.assert_array_equal(g_sorted, g_sorted_python)
        np.testing.assert_array_equal(bin_offsets, bin_offsets_python)
        assert len(bin_offsets) == len(fsg.qpoints) * np.prod(fsg._fft_mesh) + 1
        bins = np.repeat(np.arange(len(bin_offsets) - 1), np.diff(bin_offsets))
        np.testing.assert_array_equal(fsg._get_bin_indices(g_sorted), bins)


@pytest.mark.parametrize("g_point_selection", ["box", "shortest"])
@pytest.mark.parametrize("case", _cases)
def test_bin_sorted_rust_equals_python(case: str, g_point_selection: str):
    """bin_sorted_supercell_fourier_coefficients equals its Python counterpart.

    Both add the g' of each bin in the order of g_sorted, so they are equal
    without the translation.  With the translation, the phase factors of Rust
    and NumPy may differ in the last bits.  All rotations of the lattice are
    used with the g' sorted for each of them.

    """
    fsg, shape, _, translation, lattice = _make_fsg(case, g_point_selection)
    coefs = [_random_coefficients(shape, seed) for seed in (7, 8)]
    mesh_data = fsg._get_mesh_data(shape[::-1])
    source_weight = mesh_data.source_weight
    g = mesh_data.g
    t_case = np.array(translation if np.any(translation) else [0.1, 0.2, 0.3])
    zero = np.zeros(3)
    for W in _get_lattice_symmetries(lattice):
        g_sorted, bin_offsets = fsg._sort_g_points_by_bin_rust(g, W)
        for t in (zero, t_case):
            Fs_rust = bin_sorted_supercell_fourier_coefficients(
                coefs, g_sorted, source_weight, bin_offsets, W, t
            )
            Fs_python = _bin_sorted_coefficients_python(
                coefs, g_sorted, source_weight, bin_offsets, W, t
            )
            for F_rust, F_python in zip(Fs_rust, Fs_python, strict=True):
                if t is zero:
                    np.testing.assert_array_equal(F_rust, F_python)
                else:
                    np.testing.assert_allclose(
                        F_rust, F_python, rtol=0, atol=1e-13 * np.abs(F_python).max()
                    )


@pytest.mark.parametrize("g_point_selection", ["box", "shortest"])
@pytest.mark.parametrize("case", _cases)
def test_g_points_closed_rust_equals_python(case: str, g_point_selection: str):
    """g_points_closed_under_rotation of phonors equals its Python counterpart."""
    fsg, shape, _, _, lattice = _make_fsg(case, g_point_selection)
    supercell_mesh = np.array(shape[::-1], dtype="int64")
    mesh_data = fsg._get_mesh_data(shape[::-1])
    g, source_offsets = mesh_data.g, mesh_data.source_offsets
    for W in _get_lattice_symmetries(lattice):
        assert g_points_closed_under_rotation(
            g, source_offsets, supercell_mesh, W
        ) == _g_points_closed_under_rotation_python(
            g, source_offsets, supercell_mesh, W
        )


def test_g_points_closed_under_rotation_examples():
    """Closed and not closed selected g.

    The shortest g of a mesh with the symmetry of the lattice are closed under
    all its rotations.  For the mesh (4, 4, 6) of a cubic lattice and the
    rotation that exchanges the first and third axes, (0, 0, 3) is one of the
    shortest g and (3, 0, 0) is not, because (-1, 0, 0) is shorter.  The box of
    an even mesh has +N^s_k / 2 but not -N^s_k / 2.

    """
    fsg = FourierSeriesOnGrid(
        [4, 4, 3],
        np.diag([2, 2, 1]),
        g_point_selection="shortest",
        supercell_lattice=_hexagonal,
    )
    rotations = _get_lattice_symmetries(_hexagonal)
    assert len(rotations) == 24
    assert all(fsg.g_points_closed_under_rotation((6, 6, 5), W) for W in rotations)
    assert not all(fsg.g_points_closed_under_rotation((6, 5, 5), W) for W in rotations)

    fsg = FourierSeriesOnGrid(
        [2, 2, 2], np.eye(3, dtype="int64"), "shortest", supercell_lattice=_cubic
    )
    g = fsg._get_mesh_data((4, 4, 6)).g
    g_points = {tuple(point) for point in g}
    assert (0, 0, 3) in g_points and (3, 0, 0) not in g_points
    swap13 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    assert not fsg.g_points_closed_under_rotation((4, 4, 6), swap13)
    assert fsg.g_points_closed_under_rotation((4, 4, 6), np.eye(3, dtype="int64"))

    fsg_box = FourierSeriesOnGrid([2, 2, 2], np.eye(3, dtype="int64"))
    minus = -np.eye(3, dtype="int64")
    assert not fsg_box.g_points_closed_under_rotation((4, 4, 4), minus)
    assert fsg_box.g_points_closed_under_rotation((5, 5, 5), minus)


@pytest.mark.parametrize("lang", ["Rust", "Python"])
@pytest.mark.parametrize("g_point_selection", ["box", "shortest"])
@pytest.mark.parametrize("case", _cases)
def test_bin_coefficients_equal_definition(
    case: str, g_point_selection: str, lang: str
):
    """F_q(Kbar) equal the term-by-term sum for all rotations of the lattice.

    For a rotation under which the selected g are not closed, the g' are sorted
    for it and added in the order of the selected g, as in the definition, so
    the results are equal without the translation.  For the other rotations,
    the selected g sorted once are used, and the terms are added in another
    order.

    """
    fsg, shape, _, translation, lattice = _make_fsg(case, g_point_selection, lang)
    coefs = [_random_coefficients(shape, seed) for seed in (13, 14)]
    t_case = np.array(translation if np.any(translation) else [0.1, 0.2, 0.3])
    zero = np.zeros(3)
    for W in _get_lattice_symmetries(lattice):
        is_closed = fsg.g_points_closed_under_rotation(shape[::-1], W)
        for t in (zero, t_case):
            Fs = fsg._bin_coefficients(coefs, W, t)
            refs = _bin_by_definition(fsg, coefs, W, t)
            for F, ref in zip(Fs, refs, strict=True):
                if t is zero and not is_closed:
                    np.testing.assert_array_equal(F, ref)
                else:
                    np.testing.assert_allclose(
                        F, ref, rtol=0, atol=1e-13 * np.abs(ref).max()
                    )


@pytest.mark.parametrize("lang", ["Rust", "Python"])
@pytest.mark.parametrize("shape", [(5, 6, 6), (5, 5, 6)])
def test_evaluate_equals_direct_sum_for_all_rotations(
    shape: tuple[int, int, int], lang: str
):
    """Values at the grid points are the direct sum for all hexagonal rotations.

    The mesh (6, 6, 5) has the symmetry of the hexagonal lattice, so the
    shortest g sorted once are used for all 24 rotations.  For (6, 5, 5), the
    rotations that exchange the first two axes are not, and the g' are sorted
    for them.  The hexagonal basis is not orthogonal, so W^T and W^-1 differ.

    """
    p2s = np.diag([2, 2, 1]).astype("int64")
    fft_mesh = [4, 4, 3]
    translation = np.array([0.0, 0.0, 0.5])
    fsg = FourierSeriesOnGrid(
        fft_mesh,
        p2s,
        g_point_selection="shortest",
        supercell_lattice=_hexagonal,
        lang=lang,
    )
    coefs = [_random_coefficients(shape, 15)]
    grid_points, _ = get_grid_points(fft_mesh, p2s)
    rotations = _get_lattice_symmetries(_hexagonal)
    num_closed = sum(
        fsg.g_points_closed_under_rotation(shape[::-1], W) for W in rotations
    )
    assert num_closed == (24 if shape[1] == shape[2] else 8)
    for W in rotations:
        x = (grid_points - translation) @ np.linalg.inv(W).T
        values = fsg.evaluate(coefs, rotation=W, translation=translation)
        ref = _direct_sum_shortest(coefs[0], x, _hexagonal)
        np.testing.assert_allclose(
            values[0], ref, rtol=0, atol=1e-10 * np.abs(ref).max()
        )


def test_mesh_data_are_made_once_for_each_mesh():
    """The data of a mesh are made once and made again when the mesh changes.

    For the rotations under which the selected g are closed, g_sorted of the
    data of the mesh are returned; for the others, the g' are sorted for the
    rotation.

    """
    fsg = FourierSeriesOnGrid(
        [4, 4, 3], np.diag([2, 2, 1]), "shortest", supercell_lattice=_hexagonal
    )
    rotations = _get_lattice_symmetries(_hexagonal)
    made = []
    for shape in [(5, 6, 6), (5, 5, 6)]:
        coefs = [_random_coefficients(shape, 16)]
        mesh_data = fsg._get_mesh_data(shape[::-1])
        for W in rotations:
            fsg._bin_coefficients(coefs, W, np.zeros(3))
            assert fsg._get_mesh_data(shape[::-1]) is mesh_data
            g_sorted, _ = fsg._get_g_points_sorted_by_bin(shape[::-1], W)
            assert (g_sorted is mesh_data.g_sorted) == (
                fsg.g_points_closed_under_rotation(shape[::-1], W)
            )
        assert len(mesh_data.is_closed) == len(rotations)
        made.append(mesh_data)
    assert made[0] is not made[1]
    assert made[0].supercell_mesh == (6, 6, 5)
    assert made[1].supercell_mesh == (6, 5, 5)
