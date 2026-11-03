"""Tests of the element-wise steps in phelel.base.potential_change.

Each step has a Rust implementation in phonors and a pure-Python one.  The
tests apply both to copies of the same random input and compare the results.

"""

from __future__ import annotations

import numpy as np
import phonors
import pytest
from numpy.typing import NDArray

from phelel.base.potential_change import (
    _multiply_by_phase_python,
    _multiply_by_phase_rust,
)


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
