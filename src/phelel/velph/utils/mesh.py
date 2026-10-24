"""Utility functions for k-point meshes."""

from __future__ import annotations

import warnings

import numpy as np
from phonopy.phonon.grid import GridMatrix
from phonopy.structure.atoms import PhonopyAtoms
from spglib import SpglibDataset, SpglibMagneticDataset

from phelel.velph.utils.structure import get_symmetry_dataset


def kspacing_to_mesh(
    kspacing: float,
    cell: PhonopyAtoms,
    use_grg: bool = False,
    tolerance: float = 1e-5,
) -> list[int] | list[list[int]]:
    """Return the k-point mesh of cell for kspacing.

    The mesh numbers follow the VASP INCAR tag KSPACING,
    max(1, ceiling(|b_i| 2 pi / kspacing)), and respect the point group of
    cell found with tolerance.  The mesh is three integers, or a grid matrix
    of a generalized regular grid (GR-grid) whose rows are the generating
    vectors in the reciprocal basis vectors.

    use_grg is used only when cell has a single lattice point, which is when
    phonopy can make a GR-grid.  For the other cells, the regular mesh is
    returned without the warning of phonopy.  Whether cell has a single
    lattice point is decided by GridMatrix of phonopy, also from the operations
    of a magnetic space group.  When the regular mesh breaks the symmetry of a
    cell with a single lattice point, as a primitive cell of a body-centred
    lattice does, the GR-grid is returned even with use_grg=False.
    RuntimeError of phonopy is raised when no mesh respects the symmetry.

    """
    sym_dataset = get_symmetry_dataset(cell, tolerance=tolerance)
    try:
        gm = _get_grid_matrix(kspacing, cell, sym_dataset, use_grg)
    except RuntimeError as e:
        if "Grid symmetry is broken." not in str(e) or use_grg:
            raise
        gm = _get_grid_matrix(kspacing, cell, sym_dataset, True)
    if gm.grid_matrix is None:
        return gm.D_diag.tolist()
    return gm.grid_matrix.tolist()


def _get_grid_matrix(
    kspacing: float,
    cell: PhonopyAtoms,
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
    use_grg: bool,
) -> GridMatrix:
    with warnings.catch_warnings():
        # GridMatrix warns when it cannot make a GR-grid for a cell of several
        # lattice points, and makes the regular mesh.
        warnings.filterwarnings(
            "ignore", message="Non primitive cell input", category=RuntimeWarning
        )
        return GridMatrix(
            2 * np.pi / kspacing,
            lattice=cell.cell,
            symmetry_dataset=sym_dataset,
            use_grg=use_grg,
            rounding="ceiling",
        )
