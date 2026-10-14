"""Utility functions for k-point meshes."""

from __future__ import annotations

import numpy as np
from phonopy.phonon.grid import GridMatrix
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import is_primitive_cell
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
    phonopy can make a GR-grid.  When the regular mesh breaks the symmetry of
    a cell with a single lattice point, as a primitive cell of a body-centred
    lattice does, the GR-grid is returned even with use_grg=False.
    RuntimeError of phonopy is raised when no mesh respects the symmetry.

    """
    sym_dataset = get_symmetry_dataset(cell, tolerance=tolerance)
    is_primitive = bool(is_primitive_cell(sym_dataset.rotations))
    try:
        gm = _get_grid_matrix(kspacing, cell, sym_dataset, use_grg and is_primitive)
    except RuntimeError as e:
        if "Grid symmetry is broken." not in str(e) or not is_primitive:
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
    return GridMatrix(
        2 * np.pi / kspacing,
        lattice=cell.cell,
        symmetry_dataset=sym_dataset,
        use_grg=use_grg,
        rounding="ceiling",
    )
