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


def test_local_potential_interpolation_rust_and_python_agree():
    """dV/du with lang="Rust" and lang="Python" agree for ncdij=4.

    bcc Fe with non-collinear moments along z in the conventional cell, with
    the one-atom primitive cell (N_c = 2), random potentials of four spin
    components, and a displacement off the symmetry axes.  The operations
    include rotations whose D is not diagonal and operations combined with
    time reversal.

    """
    cell = PhonopyAtoms(
        symbols=["Fe", "Fe"],
        cell=np.eye(3) * 2.83,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        magnetic_moments=[[0, 0, 3], [0, 0, 3]],
    )
    p2s_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype="int64")
    V_perfect = _random_complex((4, 6, 6, 6), seed=1)
    V_displaced = V_perfect + 0.1 * _random_complex((4, 6, 6, 6), seed=2)
    displacement = {"number": 0, "displacement": [0.01, 0.02, 0.005]}

    dVdus = {}
    for lang in ("Rust", "Python"):
        lpi = LocalPotentialInterpolationNUFFT(
            [4, 4, 4], p2s_matrix, cell, Symmetry(cell), verbose=False, lang=lang
        )
        lpi.run([DeltaLocalPotential(V_perfect, V_displaced, displacement)])
        dVdus[lang] = lpi.dVdu

    assert dVdus["Rust"].shape == (4, 2, 3, 2 * 4**3)
    scale = np.abs(dVdus["Python"]).max()
    np.testing.assert_allclose(
        dVdus["Rust"], dVdus["Python"], rtol=0, atol=1e-13 * scale
    )
