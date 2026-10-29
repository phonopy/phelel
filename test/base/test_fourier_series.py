"""Tests of the exact evaluation of Fourier series on the grid points."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.typing import NDArray

import phelel.base.fourier_series as fourier_series
from phelel.base.fourier_series import (
    FourierSeriesOnGrid,
    _get_reciprocal_lattice_vectors,
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


@pytest.mark.parametrize("lang", ["Rust", "Python"])
@pytest.mark.parametrize("representatives", ["box", "shortest"])
@pytest.mark.parametrize("case", _cases)
def test_evaluate_equals_direct_sum(case: str, representatives: str, lang: str):
    """Values at the grid points are the Fourier series summed term by term.

    For g = {W|w}, f(g^-1 x) is evaluated at x' = W^-1 (x - w).

    """
    p2s, shape, fft_mesh, rotation, translation, lattice = _cases[case]
    p2s = np.array(p2s, dtype="int64")
    coefs = [_random_coefficients(shape, seed) for seed in (1, 2)]
    fsg = FourierSeriesOnGrid(
        fft_mesh,
        p2s,
        representatives=representatives,
        supercell_lattice=lattice,
        lang=lang,
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


@pytest.mark.parametrize("lang", ["Rust", "Python"])
@pytest.mark.parametrize("case", _cases)
def test_lattice_fourier_sum_equals_D_phi(case: str, lang: str):
    """sum_l f(x_lm) exp(-2 pi i q . l) = D phi_q(m).

    f(x_lm) are computed by the direct sum, so that this does not depend on
    sum_over_qpoints.  With diag(3, 1, 1), q = 1/3 and 2/3 are not equal to -q
    mod 1, so the sign of q is checked.

    """
    p2s, shape, fft_mesh, rotation, translation, _ = _cases[case]
    p2s = np.array(p2s, dtype="int64")
    coefs = _random_coefficients(shape, 4)
    fsg = FourierSeriesOnGrid(fft_mesh, p2s, lang=lang)
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


@pytest.mark.parametrize("representatives", ["box", "shortest"])
@pytest.mark.parametrize("case", _cases)
def test_lang_rust_equals_python(case: str, representatives: str):
    """F_q(Kbar) of lang="Rust" are those of the reference lang="Python".

    Both sum the coefficients of each bin in the order of the rows of g, so
    they are equal without the translation.  With the translation, the phase
    factors of Rust and NumPy may differ in the last bits.

    """
    p2s, shape, fft_mesh, rotation, translation, lattice = _cases[case]
    p2s = np.array(p2s, dtype="int64")
    coefs = [_random_coefficients(shape, seed) for seed in (5, 6)]
    fsg_rust, fsg_python = (
        FourierSeriesOnGrid(
            fft_mesh,
            p2s,
            representatives=representatives,
            supercell_lattice=lattice,
            lang=lang,
        )
        for lang in ("Rust", "Python")
    )
    Fs_rust = fsg_rust._bin_coefficients_rust(coefs, rotation, None)
    Fs_python = fsg_rust._bin_coefficients_python(coefs, rotation, None)
    for F_rust, F_python in zip(Fs_rust, Fs_python, strict=True):
        assert F_rust.shape == F_python.shape
        np.testing.assert_array_equal(F_rust, F_python)

    phis_rust = fsg_rust.get_phi(coefs, rotation=rotation, translation=translation)
    phis_python = fsg_python.get_phi(coefs, rotation=rotation, translation=translation)
    for phi_rust, phi_python in zip(phis_rust, phis_python, strict=True):
        np.testing.assert_allclose(
            phi_rust, phi_python, rtol=0, atol=1e-13 * np.abs(phi_python).max()
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
    g, _, weight = get_shortest_representatives(shape[::-1], lattice)
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


def _get_shortest_representatives_by_bound(
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
    chosen_g = []
    chosen_source = []
    for shift in shifts:
        g = g_box + shift
        is_shortest = get_lengths(g) <= shortest * (1 + tolerance)
        chosen_g.append(g[is_shortest])
        chosen_source.append(np.nonzero(is_shortest)[0])
    g = np.concatenate(chosen_g)
    source = np.concatenate(chosen_source)
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
def test_shortest_representatives_equal_search_without_reduction(
    shape: tuple[int, int, int], lattice: NDArray
):
    """The shortest g + N^s n equal those searched without the reduced basis.

    The coefficients divided among g of the same length are included.

    """
    g, source, weight = get_shortest_representatives(shape[::-1], lattice)
    g_ref, source_ref, weight_ref = _get_shortest_representatives_by_bound(
        shape, lattice
    )
    assert (weight < 1).any()
    np.testing.assert_array_equal(
        _sorted_rows(np.c_[source, g, weight]),
        _sorted_rows(np.c_[source_ref, g_ref, weight_ref]),
    )


@pytest.mark.parametrize("lattice", [_cubic, _hexagonal])
def test_shortest_representatives_do_not_depend_on_basis(lattice: NDArray):
    """The shortest G with their weights are the same for a skewed basis.

    With the mesh (n, n, n), the period lattice {N^s n} of g is kept when the
    basis vectors A of the supercell are replaced by U A with a unimodular U,
    so the shortest G = g (U A)^-T and their weights are the same.  U is
    _shears, which makes the basis strongly skewed.

    """
    shape = (6, 6, 6)
    G_weight = []
    for A in (lattice, _shears @ lattice):
        g, _, weight = get_shortest_representatives(shape[::-1], A)
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
def test_shortest_representatives_are_closest(
    shape: tuple[int, int, int], lattice: NDArray
):
    """No chosen g becomes shorter by a Voronoi-relevant vector of the period lattice.

    The Voronoi-relevant vectors of a lattice with a Delaunay-reduced basis
    w_1, w_2, w_3 have coefficients in {-1, 0, 1} (Conway and Sloane, Proc. R.
    Soc. Lond. A 436, 55 (1992)), and y is the shortest of y + {N^s n} when no
    relevant vector v makes |G(y - v)| shorter than |G(y)|.  This holds
    independently of the range of the search.  The bases are strongly skewed,
    for which the search without the reduction is too slow to compare with.

    """
    mesh = np.array(shape[::-1])
    period_lattice = np.diag(mesh) @ np.linalg.inv(lattice).T
    W = fourier_series._get_reduced_period_basis(period_lattice) @ np.diag(mesh)
    v = np.array(list(itertools.product((-1, 0, 1), repeat=3))) @ W
    metric = np.linalg.inv(lattice @ lattice.T)
    g, _, _ = get_shortest_representatives(shape[::-1], lattice)
    lengths = np.sqrt(np.einsum("ij,jk,ik->i", g, metric, g))
    y = g[:, None, :] - v[None, :, :]
    others = np.sqrt(np.einsum("abj,jk,abk->ab", y, metric, y))
    assert (others.min(axis=1) >= lengths * (1 - 1e-8)).all()


@pytest.mark.parametrize("is_reduced", [True, False])
def test_shortest_representatives_path_superbase(
    is_reduced: bool, monkeypatch: pytest.MonkeyPatch
):
    """All the closest points are chosen with the superbase of the path type.

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
            "_get_reduced_period_basis",
            lambda _: np.eye(3, dtype="int64"),
        )
    n = 4
    shape = (n, n, n)
    path = np.array([[-1.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    reciprocal = path / n
    lattice = np.linalg.inv(reciprocal).T
    g, source, weight = get_shortest_representatives(shape[::-1], lattice)
    i = np.nonzero(
        (_get_reciprocal_lattice_vectors(shape[::-1]) == [n // 2, 0, n // 2]).all(
            axis=1
        )
    )[0][0]
    np.testing.assert_array_equal(weight[source == i], 1 / 8)
    G = g[source == i] @ reciprocal
    expected = np.array(list(itertools.product([-0.5, 0.5], repeat=3)))
    np.testing.assert_allclose(_sorted_rows(G), expected, atol=1e-12)


def test_representatives_errors():
    """Unknown representatives and "shortest" without the lattice are errors."""
    p2s = np.eye(3, dtype="int64")
    with pytest.raises(ValueError, match="representatives has to be"):
        FourierSeriesOnGrid([2, 2, 2], p2s, representatives="nearest")  # type: ignore
    with pytest.raises(ValueError, match="supercell_lattice is needed"):
        FourierSeriesOnGrid([2, 2, 2], p2s, representatives="shortest")


def test_lang_errors():
    """Unknown lang is an error."""
    p2s = np.eye(3, dtype="int64")
    with pytest.raises(ValueError, match="lang has to be"):
        FourierSeriesOnGrid([2, 2, 2], p2s, lang="C")  # type: ignore
