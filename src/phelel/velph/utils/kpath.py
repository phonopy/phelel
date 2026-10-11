"""Band paths made by seekpath.

seekpath is an optional dependency.  It is used only when a line block of
velph.toml, such as [vasp.el_bands.kpoints_opt], does not give "path".

"""

from __future__ import annotations

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import merge_weighted_species

from phelel.velph.config.schema import FloatVector3, LinePath


def get_seekpath_line_path(cell: PhonopyAtoms, line: int) -> LinePath:
    """Return the band path that seekpath gives, in the basis of cell.

    seekpath rotates the cell to its standard orientation and may choose other
    basis vectors of the same lattice.  The coordinates of the points it gives
    are converted to the basis of cell.  Atoms at the same position, as in a
    cell with site mixture, are merged before the cell is given to seekpath.

    ModuleNotFoundError is raised when seekpath is not installed.

    """
    import seekpath

    if cell.has_weighted_species:
        cell, _ = merge_weighted_species(cell)
    band_path = seekpath.get_path(cell.totuple())
    # Rows of lattice_sp are P times the rows of the lattice of cell, both in
    # the standard orientation of seekpath.  The fractional coordinates of a
    # point are k = inv(P) k_sp.
    lattice_sp = np.array(band_path["primitive_lattice"])
    rotation = np.array(band_path["rotation_matrix"])
    matrix = lattice_sp @ np.linalg.inv(cell.cell @ rotation.T)
    basis_change = np.rint(matrix).astype(int)
    if not np.allclose(matrix, basis_change, atol=1e-5) or (
        abs(round(np.linalg.det(basis_change))) != 1
    ):
        raise RuntimeError("The lattice of seekpath differs from that of the cell.")
    inv_basis_change = np.rint(np.linalg.inv(basis_change)).astype(int)
    coords: dict[str, FloatVector3] = {}
    for label, k_sp in band_path["point_coords"].items():
        k = inv_basis_change @ np.array(k_sp)
        coords[label] = (float(k[0]), float(k[1]), float(k[2]))
    return LinePath(
        line=line,
        path=tuple((coords[start], coords[end]) for start, end in band_path["path"]),
        label=tuple((start, end) for start, end in band_path["path"]),
    )
