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


def check_primitive_translations(
    supercell: PhonopyAtoms, p2s_matrix: NDArray, tolerance: float
) -> None:
    """Check that the lattice vectors of the primitive cell are translations.

    phelel gives the derivatives of the atoms of the primitive cell only, and
    those of the other atoms of the supercell are obtained by the lattice
    vectors of the primitive cell.  Every lattice vector of the primitive cell
    therefore has to be a translation of the supercell without time reversal:
    it has to move every atom onto an atom of the same species with the same
    magnetic moment.  A vector that moves the atoms onto atoms with the
    opposite moments, an anti-translation, is not allowed.  phonopy builds the
    primitive cell without looking at the magnetic moments (until version
    4.10.0) or allows anti-translations (later versions), so the primitive
    cell of the crystal structure of an antiferromagnet passes phonopy.

    The three basis vectors of the primitive cell are checked, which is enough,
    because the lattice vectors are their sums and the translations by the
    lattice vectors of the supercell keep the supercell.

    Parameters
    ----------
    supercell : PhonopyAtoms
        Perfect supercell.
    p2s_matrix : ndarray
        Supercell matrix relative to the primitive cell.
        dtype='int64', shape=(3, 3)
    tolerance : float
        Tolerance of the distances in Cartesian coordinates and of the norms of
        the differences of the magnetic moments.

    Raises
    ------
    ValueError
        When a basis vector of the primitive cell does not move an atom onto
        an atom of the same species, or onto an atom with the same magnetic
        moment.

    """
    positions = supercell.scaled_positions
    # species_ids also distinguish the mixed-species sites of the merge scheme.
    species = supercell.species_ids
    magmoms = supercell.magnetic_moments
    moments = None
    if magmoms is not None:
        moments = np.array(magmoms, dtype="double").reshape(len(positions), -1)
    # The basis vectors of the primitive cell are the columns of
    # (p2s_matrix)^-1 in the fractional coordinates of the supercell.
    for translation in np.linalg.inv(p2s_matrix).T:
        differences = positions[None, :, :] - (positions[:, None, :] + translation)
        differences -= np.rint(differences)
        distances = np.linalg.norm(differences @ supercell.cell, axis=2)
        matches = (distances < tolerance) & (species[None, :] == species[:, None])
        if not matches.any(axis=1).all():
            raise ValueError(
                "A lattice vector of the primitive cell is not a translation of "
                "the supercell. The primitive cell has to be that of the supercell "
                "and p2s_matrix."
            )
        if moments is None:
            continue
        moment_differences = np.linalg.norm(
            moments[None, :, :] - moments[:, None, :], axis=2
        )
        if not (matches & (moment_differences <= tolerance)).any(axis=1).all():
            raise ValueError(
                "Magnetic moments of the supercell differ between atoms related by "
                "a lattice vector of the primitive cell, so the lattice vectors of "
                "the primitive cell are not all translations of the magnetic "
                "structure. Use a primitive cell of the magnetic structure, e.g., "
                'primitive_matrix="auto".'
            )


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
