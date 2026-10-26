"""Site-symmetry operations used to symmetrize derivatives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from phonopy.harmonic.displacement import FirstAtomDisplacement
from phonopy.structure.symmetry import Symmetry
from phonopy.utils import similarity_transformation

if TYPE_CHECKING:
    from phelel.base.Dij_qij import DeltaDijQij
    from phelel.base.local_potential import DeltaLocalPotential


def collect_site_symmetry_operations(
    disp_atom: int, symmetry: Symmetry
) -> tuple[NDArray, NDArray]:
    """Collect site symmetry operations.

    Collect symmetry operations from a diplaced atom to symmetrically
    equivalent atoms. When the symmetrically equivalent atom is the
    displaced atom, those operations are the site-symmetry
    operations. In the other cases, those operations are bound for
    each symmetrically equivalent atom.

    Returns
    -------
    sitesym_sets : ndarray
        Sets of indices of symmetry operations. Each set contains symmetry
        operations that send the displaced atom to specific one of equivalent
        atoms.
        dtype='int64'
        shape=(equiv_atoms, site_syms)
    equiv_atoms : ndarray
        Indices of symmetrically equivalent atoms
        dtype='int64'
        shape=(equiv_atoms,)

    """
    perms = symmetry.atomic_permutations[:, disp_atom]

    # This gives something like [0, 0, 0, 0, 4, 4, 4, 4, ...].
    map_to_indep_atoms = symmetry.get_map_atoms()
    # This gives something like [0, 1, 2, 3] for disp_atom=0.
    equiv_atoms = np.array(
        [
            i
            for i, x in enumerate(map_to_indep_atoms)
            if map_to_indep_atoms[disp_atom] == x
        ],
        dtype="int64",
    )
    ops_sets = []
    for i_atom in equiv_atoms:
        isyms = np.where(perms == i_atom)[0]
        if len(isyms) > 0:
            ops_sets.append(isyms)

    return np.array(ops_sets, dtype="int64", order="C"), equiv_atoms


def select_equivalent_atoms(
    disp_atom: int, symmetry: Symmetry, atom_indices: Sequence[int] | NDArray | None
) -> tuple[NDArray, NDArray]:
    """Return the atoms equivalent to a displaced atom and their operations.

    Parameters
    ----------
    disp_atom : int
        Index of the displaced atom in the supercell.
    symmetry : Symmetry
        Symmetry of the supercell.
    atom_indices : Sequence[int] | NDArray | None
        Atoms among which the equivalent atoms are selected.  None means all
        atoms of the supercell.

    Returns
    -------
    atoms : ndarray
        Atoms of atom_indices equivalent to disp_atom by symmetry, in the order
        of atom_indices.
        dtype='int64', shape=(n_atoms,)
    sitesym_sets : ndarray
        Indices of the symmetry operations that send disp_atom to each of the
        atoms.  See collect_site_symmetry_operations.
        dtype='int64', shape=(n_atoms, n_site_syms)

    """
    sitesym_sets, equiv_atoms = collect_site_symmetry_operations(disp_atom, symmetry)
    if atom_indices is None:
        atom_indices = np.arange(len(symmetry.get_map_atoms()))
    atoms = np.array([i for i in atom_indices if i in equiv_atoms], dtype="int64")
    # The rows of sitesym_sets follow equiv_atoms.  They are picked in the order
    # of the selected atoms.
    i_equiv_atom = {int(atom): i for i, atom in enumerate(equiv_atoms)}
    return atoms, sitesym_sets[[i_equiv_atom[int(atom)] for atom in atoms]]


def check_displaced_atoms(
    atom_indices: NDArray,
    displacements: Sequence[FirstAtomDisplacement],
    symmetry: Symmetry,
) -> None:
    """Raise ValueError for atoms not equivalent to any displaced atom.

    The derivatives of an atom are obtained from the displacements of the atoms
    equivalent to it by symmetry.  Without them, the derivatives would be left
    zero.

    """
    map_atoms = symmetry.get_map_atoms()
    displaced = {map_atoms[d["number"]] for d in displacements}
    missing = [int(i) + 1 for i in atom_indices if map_atoms[i] not in displaced]
    if missing:
        raise ValueError(
            "No atom equivalent to atoms "
            f"{', '.join(str(i) for i in missing)} of the supercell (counted "
            "from 1) is displaced, so their derivatives are not obtained."
        )


def get_displacements_with_rotations(
    rotations: NDArray,
    lattice: NDArray,
    delta_vals: list[DeltaLocalPotential] | list[DeltaDijQij],
) -> NDArray:
    """Rotate displacements by site-symmetry actively.

    Displacements are stored in the following order:

        disps = []
        for calc_disp_dir in disps_of_one_atoms:
            for r in sitesym:
                disps.append(dot(r, calc_disp_dir))

    Returns
    -------
    disps : ndarray
        shape=(calc_disp_dir * sitesyms, 3)

    """
    disps = np.zeros((len(rotations) * len(delta_vals), 3), dtype="double", order="C")
    count = 0
    for delta_V in delta_vals:
        disp = delta_V.displacement["displacement"]
        for r in rotations:
            r_cart = similarity_transformation(lattice, r)
            disps[count] = np.dot(r_cart, disp)
            count += 1
    return disps
