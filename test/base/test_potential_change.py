"""Tests of the element-wise steps in phelel.base.potential_change.

Each step has a Rust implementation in phonors and a pure-Python one.  The
tests apply both to copies of the same random input and compare the results.

"""

from __future__ import annotations

import numpy as np
import phonors
import pytest
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry

from phelel.base.local_potential import (
    DeltaLocalPotential,
    LocalPotentialInterpolationNUFFT,
)
from phelel.base.potential_change import (
    _add_to_normal_equation_rhs_python,
    _add_to_normal_equation_rhs_rust,
    _apply_inverse_normal_matrix_python,
    _apply_inverse_normal_matrix_rust,
    _apply_time_reversal_to_spin_components_python,
    _apply_time_reversal_to_spin_components_rust,
    _multiply_by_phase_python,
    _multiply_by_phase_rust,
    _rotate_spin_components_python,
    _rotate_spin_components_rust,
)
from phelel.utils.spinor import SpinorRotationMatrices


def _random_complex(shape: tuple[int, ...], seed: int) -> NDArray[np.cdouble]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_multiply_by_phase():
    """Rust and Python give the product, and phase is not changed.

    The shape is that of phi_q(m) in get_phi, (D, N^p_3, N^p_2, N^p_1), with
    more elements than one chunk of the Rust kernel (4096).

    """
    phi = _random_complex((4, 9, 10, 11), seed=1)
    phase = np.exp(2j * np.pi * np.random.default_rng(2).random(phi.shape))
    phase_original = phase.copy()
    phi_rust = phi.copy()
    phi_python = phi.copy()

    _multiply_by_phase_rust(phi_rust, phase)
    _multiply_by_phase_python(phi_python, phase)

    np.testing.assert_allclose(phi_python, phi * phase, rtol=0, atol=0)
    np.testing.assert_allclose(phi_rust, phi_python, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(phase, phase_original)


def test_multiply_by_phase_rust_rejects_wrong_input():
    """Different shapes and non-contiguous arrays raise ValueError."""
    phi = _random_complex((4, 6), seed=1)
    phase = _random_complex((4, 6), seed=2)
    with pytest.raises(ValueError, match="same shape"):
        _multiply_by_phase_rust(phi, phase.ravel())
    with pytest.raises(ValueError, match="phi must be C-contiguous"):
        _multiply_by_phase_rust(phi[:, ::2], phase[:, ::2].copy())
    with pytest.raises(ValueError, match="phase must be C-contiguous"):
        _multiply_by_phase_rust(phi[:, ::2].copy(), phase[:, ::2])


def test_multiply_by_phase_rust_rejects_the_same_array_twice():
    """Passing the same array as phi and phase stops before computing.

    rust-numpy refuses to borrow the array mutably while it is borrowed, and
    the extraction of the arguments panics.

    """
    phi = _random_complex((4, 6), seed=1)
    phi_original = phi.copy()
    with pytest.raises(BaseException, match="AlreadyBorrowed"):
        phonors.multiply_by_phase(phi, phi)
    np.testing.assert_array_equal(phi, phi_original)


def _spin_rotation_of_threefold_axis() -> NDArray[np.cdouble]:
    """Return D of the three-fold rotation about [111] of a cubic lattice.

    D is not diagonal and has complex elements.

    """
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    spin_rotation = SpinorRotationMatrices(rotation, np.eye(3)).run().Delta
    assert np.abs(spin_rotation - np.diag(np.diag(spin_rotation))).max() > 0.1
    assert np.abs(spin_rotation.imag).max() > 0.1
    return spin_rotation


def test_rotate_spin_components():
    """Rust and Python give D V D^dagger at every element.

    The reference is computed with 2x2 matrix products by np.einsum,
    independently of the two implementations.  The number of elements is larger
    than one chunk of the Rust kernel (4096).

    """
    spin_rotation = _spin_rotation_of_threefold_axis()
    values = [_random_complex((3, 40, 50), seed=seed) for seed in range(4)]
    matrices = np.array(values).reshape(2, 2, -1)
    expected = np.einsum(
        "ij,jkn,lk->iln", spin_rotation, matrices, spin_rotation.conj()
    ).reshape(4, *values[0].shape)
    values_rust = [value.copy() for value in values]
    values_python = [value.copy() for value in values]

    _rotate_spin_components_rust(values_rust, spin_rotation)
    _rotate_spin_components_python(values_python, spin_rotation)

    atol = 1e-14 * np.abs(expected).max()
    np.testing.assert_allclose(values_python, expected, rtol=0, atol=atol)
    np.testing.assert_allclose(values_rust, values_python, rtol=0, atol=atol)


def test_rotate_spin_components_rust_rejects_wrong_input():
    """Wrong numbers, shapes and layouts raise ValueError."""
    spin_rotation = _spin_rotation_of_threefold_axis()
    values = [_random_complex((4, 6), seed=seed) for seed in range(4)]
    with pytest.raises(ValueError, match="four spin components"):
        _rotate_spin_components_rust(values[:2], spin_rotation)
    with pytest.raises(ValueError, match="same shape"):
        _rotate_spin_components_rust(
            [*values[:3], values[3][:, :3].copy()], spin_rotation
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        _rotate_spin_components_rust([value[:, ::2] for value in values], spin_rotation)
    with pytest.raises(ValueError, match="shape \\(2, 2\\)"):
        _rotate_spin_components_rust(values, np.eye(3, dtype="complex128"))


@pytest.mark.parametrize("num_components", [1, 2, 4])
def test_apply_time_reversal_to_spin_components(num_components: int):
    """Rust and Python give the formulas of the docstring.

    The number of elements is larger than one chunk of the Rust kernel (4096).

    """
    values = [_random_complex((5, 1000), seed=seed) for seed in range(num_components)]
    if num_components == 4:
        expected = [
            values[3].conj(),
            -values[2].conj(),
            -values[1].conj(),
            values[0].conj(),
        ]
    elif num_components == 2:
        expected = [values[1].conj(), values[0].conj()]
    else:
        expected = [values[0].conj()]
    values_rust = [value.copy() for value in values]
    values_python = [value.copy() for value in values]

    _apply_time_reversal_to_spin_components_rust(values_rust)
    _apply_time_reversal_to_spin_components_python(values_python)

    np.testing.assert_array_equal(values_python, expected)
    np.testing.assert_array_equal(values_rust, expected)


def test_apply_time_reversal_to_spin_components_rust_rejects_wrong_input():
    """Wrong numbers, shapes and layouts raise ValueError."""
    values = [_random_complex((4, 6), seed=seed) for seed in range(4)]
    with pytest.raises(ValueError, match="one, two or four spin components"):
        _apply_time_reversal_to_spin_components_rust(values[:3])
    with pytest.raises(ValueError, match="same shape"):
        _apply_time_reversal_to_spin_components_rust(
            [values[0], values[1][:, :3].copy()]
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        _apply_time_reversal_to_spin_components_rust(
            [value[:, ::2] for value in values]
        )


def test_add_to_normal_equation_rhs():
    """Rust and Python add displacement[i] * values[c] to normal_equation_rhs[c, i].

    Two additions to a nonzero normal_equation_rhs, with two spin components,
    are compared with np.einsum.  The number of elements is larger than one
    chunk of the Rust kernel (4096).

    """
    rng = np.random.default_rng(10)
    num_grid_points = 5000
    start = _random_complex((2, 3, num_grid_points), seed=11)
    additions = [
        (
            rng.standard_normal(3),
            [_random_complex((num_grid_points,), seed=s) for s in (12, 13)],
        ),
        (
            rng.standard_normal(3),
            [_random_complex((num_grid_points,), seed=s) for s in (14, 15)],
        ),
    ]
    expected = start + sum(
        np.einsum("i,cn->cin", displacement, np.array(values))
        for displacement, values in additions
    )
    normal_equation_rhs_rust = start.copy()
    normal_equation_rhs_python = start.copy()
    for displacement, values in additions:
        _add_to_normal_equation_rhs_rust(normal_equation_rhs_rust, displacement, values)
        _add_to_normal_equation_rhs_python(
            normal_equation_rhs_python, displacement, values
        )

    atol = 1e-14 * np.abs(expected).max()
    np.testing.assert_allclose(normal_equation_rhs_python, expected, rtol=0, atol=atol)
    np.testing.assert_allclose(
        normal_equation_rhs_rust, normal_equation_rhs_python, rtol=0, atol=atol
    )


def test_add_to_normal_equation_rhs_rust_rejects_wrong_input():
    """Inconsistent shapes and non-contiguous arrays raise ValueError."""
    normal_equation_rhs = _random_complex((2, 3, 8), seed=1)
    displacement = np.ones(3)
    values = [_random_complex((8,), seed=seed) for seed in (2, 3)]
    with pytest.raises(ValueError, match="same number of elements"):
        _add_to_normal_equation_rhs_rust(
            normal_equation_rhs, displacement, [values[0], values[1][:5].copy()]
        )
    with pytest.raises(ValueError, match="displacement must have shape"):
        _add_to_normal_equation_rhs_rust(normal_equation_rhs, np.ones(2), values)
    with pytest.raises(ValueError, match="normal_equation_rhs must have shape"):
        _add_to_normal_equation_rhs_rust(normal_equation_rhs, displacement, values[:1])
    with pytest.raises(ValueError, match="spin components must be C-contiguous"):
        _add_to_normal_equation_rhs_rust(
            normal_equation_rhs,
            displacement,
            [_random_complex((16,), seed=4)[::2]] * 2,
        )
    with pytest.raises(ValueError, match="normal_equation_rhs must be C-contiguous"):
        _add_to_normal_equation_rhs_rust(
            _random_complex((2, 3, 16), seed=5)[:, :, ::2], displacement, values
        )


def test_apply_inverse_normal_matrix():
    """Rust and Python multiply normal_equation_rhs[c, :, x] by the matrix.

    The reference is computed with np.einsum.  The number of elements is
    larger than one chunk of the Rust kernel (4096).

    """
    matrix = np.random.default_rng(20).standard_normal((3, 3))
    normal_equation_rhs = _random_complex((4, 3, 5000), seed=21)
    expected = np.einsum("ij,cjn->cin", matrix, normal_equation_rhs)
    normal_equation_rhs_rust = normal_equation_rhs.copy()
    normal_equation_rhs_python = normal_equation_rhs.copy()

    _apply_inverse_normal_matrix_rust(normal_equation_rhs_rust, matrix)
    _apply_inverse_normal_matrix_python(normal_equation_rhs_python, matrix)

    atol = 1e-14 * np.abs(expected).max()
    np.testing.assert_allclose(normal_equation_rhs_python, expected, rtol=0, atol=atol)
    np.testing.assert_allclose(
        normal_equation_rhs_rust, normal_equation_rhs_python, rtol=0, atol=atol
    )


def test_apply_inverse_normal_matrix_rust_rejects_wrong_input():
    """Wrong shapes and a non-contiguous normal_equation_rhs raise ValueError."""
    matrix = np.eye(3)
    with pytest.raises(ValueError, match="shape \\(n_components, 3, n_grid_points\\)"):
        _apply_inverse_normal_matrix_rust(_random_complex((2, 2, 8), seed=1), matrix)
    with pytest.raises(ValueError, match="shape \\(3, 3\\)"):
        _apply_inverse_normal_matrix_rust(_random_complex((2, 3, 8), seed=1), np.eye(2))
    with pytest.raises(ValueError, match="C-contiguous"):
        _apply_inverse_normal_matrix_rust(
            _random_complex((2, 3, 8), seed=1)[:, :, ::2], matrix
        )


@pytest.mark.parametrize("rank", [3, 2])
def test_normal_equations_give_pinv_of_displacements(rank: int):
    """pinv(U^T U) U^T dV equals pinv(U) dV for the displacements of a site.

    U has more rows than columns, as the rotated displacements of
    get_displacements_with_rotations.  With rank 2, all displacements lie in
    a plane, U^T U is singular, and both give the minimum-norm solution.  The
    sum of the outer products and the multiplication by pinv(U^T U) are those
    of _run_at_atom.

    """
    rng = np.random.default_rng(30)
    disps = rng.standard_normal((8, 3))
    if rank == 2:
        disps[:, 2] = disps[:, 0] - 0.5 * disps[:, 1]
    assert np.linalg.matrix_rank(disps) == rank
    values = [
        [_random_complex((50,), seed=100 + 2 * k + c) for c in range(2)]
        for k in range(8)
    ]
    expected = np.einsum("wk,kcn->cwn", np.linalg.pinv(disps), np.array(values))
    for add, multiply in (
        (_add_to_normal_equation_rhs_rust, _apply_inverse_normal_matrix_rust),
        (_add_to_normal_equation_rhs_python, _apply_inverse_normal_matrix_python),
    ):
        normal_equation_rhs = np.zeros((2, 3, 50), dtype="complex128")
        for disp, value in zip(disps, values, strict=True):
            add(normal_equation_rhs, disp, value)
        multiply(
            normal_equation_rhs, np.linalg.pinv(np.einsum("ki,kj->ij", disps, disps))
        )
        np.testing.assert_allclose(
            normal_equation_rhs, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
        )


@pytest.mark.parametrize(
    "magnetic_moments,ncdij",
    [([[0, 0, 3], [0, 0, 3]], 4), ([3, -3], 2), (None, 1)],
)
def test_local_potential_interpolation_rust_and_python_agree(
    magnetic_moments: list | None, ncdij: int
):
    """dV/du with lang="Rust" and lang="Python" agree.

    bcc Fe in the conventional cell, with the one-atom primitive cell
    (N_c = 2), random potentials, and a displacement off the symmetry axes.
    With non-collinear moments along z (ncdij=4), the operations include
    rotations whose D is not diagonal and operations combined with time
    reversal.  With collinear antiferromagnetic moments (ncdij=2), the second
    atom is reached only by operations combined with time reversal.

    """
    cell = PhonopyAtoms(
        symbols=["Fe", "Fe"],
        cell=np.eye(3) * 2.83,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        magnetic_moments=magnetic_moments,
    )
    p2s_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype="int64")
    V_perfect = _random_complex((ncdij, 6, 6, 6), seed=1)
    V_displaced = V_perfect + 0.1 * _random_complex((ncdij, 6, 6, 6), seed=2)
    displacement = {"number": 0, "displacement": [0.01, 0.02, 0.005]}

    dVdus = {}
    for lang in ("Rust", "Python"):
        lpi = LocalPotentialInterpolationNUFFT(
            [4, 4, 4], p2s_matrix, cell, Symmetry(cell), verbose=False, lang=lang
        )
        lpi.run([DeltaLocalPotential(V_perfect, V_displaced, displacement)])
        dVdus[lang] = lpi.dVdu

    assert dVdus["Rust"].shape == (ncdij, 2, 3, 2 * 4**3)
    scale = np.abs(dVdus["Python"]).max()
    np.testing.assert_allclose(
        dVdus["Rust"], dVdus["Python"], rtol=0, atol=1e-13 * scale
    )
