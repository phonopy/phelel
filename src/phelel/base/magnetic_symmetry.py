"""Magnetic moments and time reversal in magnetic symmetry operations."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry
from spglib import SpglibMagneticDataset

from phelel.exception import MagneticMomentsError, MagneticMomentsWarning


def get_time_reversals(symmetry: Symmetry) -> NDArray[np.bool_]:
    """Return time-reversal flags of symmetry operations.

    With magnetic moments, phonopy's ``Symmetry`` holds the operations of the
    magnetic space group, and some of them are combined with time reversal.
    Without magnetic moments, no operation has time reversal.

    """
    n_ops = len(symmetry.symmetry_operations["rotations"])
    dataset = symmetry.dataset
    if not isinstance(dataset, SpglibMagneticDataset):
        return np.zeros(n_ops, dtype=bool)
    time_reversals = np.array(dataset.time_reversals, dtype=bool)
    assert len(time_reversals) == n_ops
    return time_reversals


def check_magnetic_moments(cell: PhonopyAtoms, ncdij: int) -> None:
    """Check that the magnetic moments of the cell fit ncdij.

    The magnetic space group found from the magnetic moments has to describe
    the spin of the calculation: collinear moments, shape=(natom,), for ncdij=2
    and non-collinear moments, shape=(natom, 3), for ncdij=4. Otherwise the
    symmetrization gives wrong dV and dDij without notice, and
    MagneticMomentsError is raised. The number of values matters also when
    all the values are zero.

    Without magnetic moments for ncdij=2 or 4, MagneticMomentsWarning is
    issued. The symmetry operations then act on the atomic positions only, but
    they are applied also to the spin parts of dV and dDij. When the magnetic
    moments are not zero in the VASP calculation, an operation that maps the
    atomic positions but not the magnetic moments may be used. The default
    MAGMOM of VASP is not zero. Nothing is checked for ncdij=1.

    """
    if ncdij == 1:
        return
    magmoms = cell.magnetic_moments
    if magmoms is None:
        if ncdij == 2:
            kind, n_values = "collinear spin-polarized (ncdij=2)", "one value"
        else:
            kind, n_values = "non-collinear (ncdij=4)", "three values"
        warnings.warn(
            "No magnetic moments are given, but the VASP calculation is "
            f"{kind}. The symmetry operations act on the atomic positions "
            "only, and they are applied also to the spin parts of the "
            "derivatives. When the magnetic moments are not zero in the VASP "
            "calculation, an operation that maps the atomic positions but not "
            "the magnetic moments may be used, and the derivatives are wrong. "
            "The default MAGMOM of VASP is not zero, so the VASP calculation "
            "may have started from magnetic moments. The magnetic moments are "
            "included in the symmetry operations by phelel-init --magmom with "
            f"{n_values} per atom, and zeros are given when the magnetic "
            "moments are zero. The supercells have to be generated again with "
            "--magmom.",
            MagneticMomentsWarning,
            stacklevel=1,
        )
        return
    if ncdij == 2 and magmoms.ndim != 1:
        raise MagneticMomentsError(
            "The magnetic moments have three values per atom, but the VASP "
            "calculation is collinear (ncdij=2). For a collinear calculation, "
            "give one value per atom by phelel-init --magmom, and generate the "
            "supercells again."
        )
    if ncdij == 4 and magmoms.ndim != 2:
        raise MagneticMomentsError(
            "The magnetic moments have one value per atom, but the VASP "
            "calculation is non-collinear (ncdij=4). For a non-collinear "
            "calculation, give three values per atom by phelel-init --magmom, "
            "and generate the supercells again."
        )


def apply_time_reversal_in_spin_space(delta_vals: list[NDArray]) -> list[NDArray]:
    r"""Apply time reversal to delta vals.

    Time reversal is :math:`\Theta = i\sigma_y K`. For ncdij=4, the 2x2 matrix
    in spinor representation transforms as
    :math:`B \rightarrow \sigma_y B^* \sigma_y`, i.e., with the data order
    [11, 12, 21, 22],

    [B11, B12, B21, B22] -> [B22^*, -B21^*, -B12^*, B11^*].

    This keeps the scalar part and flips the sign of the magnetic part. For
    ncdij=2, spin up and down are exchanged. For ncdij=1, only the complex
    conjugate is taken.

    Parameters
    ----------
    delta_vals : list[NDArray]
        Delta local potential or delta-Dijqij, in real space and in a real
        basis, after spatial and spin rotations.

    """
    ncdij = len(delta_vals)
    if ncdij == 4:
        return [
            delta_vals[3].conj(),
            -delta_vals[2].conj(),
            -delta_vals[1].conj(),
            delta_vals[0].conj(),
        ]
    if ncdij == 2:
        return [delta_vals[1].conj(), delta_vals[0].conj()]
    return [delta_vals[0].conj()]
