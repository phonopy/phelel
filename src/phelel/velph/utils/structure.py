"""Utility functions for crystal structure."""

from __future__ import annotations

import numpy as np
import spglib
from numpy.typing import NDArray
from phonopy.interface.vasp import sort_positions_by_symbols
from phonopy.structure.atoms import PhonopyAtoms, get_atomic_data
from phonopy.structure.cells import (
    get_primitive,
    get_primitive_matrix_by_centring,
    get_reduced_bases,
)
from spglib import SpglibDataset, SpglibMagneticDataset


def generate_standardized_cells(
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
    tolerance: float = 1e-5,
) -> tuple[PhonopyAtoms, PhonopyAtoms, NDArray]:
    """Return standardized unit cell and primitive cell."""
    convcell = _get_standardized_unitcell(sym_dataset)
    pmat = _get_primitive_matrix_from_dataset(sym_dataset)
    if (np.abs(pmat - np.eye(3)) < 1e-8).all():
        primitive = convcell
    else:
        primitive = get_primitive(convcell, primitive_matrix=pmat, symprec=tolerance)

    return convcell, primitive, pmat


def get_primitive_cell(
    cell: PhonopyAtoms,
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
    tolerance: float = 1e-5,
) -> tuple[PhonopyAtoms, NDArray]:
    """Return primitive cell and transformation matrix.

    This primitive cell is generated from the input cell without
    rigid rotation in contrast to `_get_standardized_unitcell`.

    """
    tmat = sym_dataset.transformation_matrix
    pmat = _get_primitive_matrix_from_dataset(sym_dataset)
    total_tmat = np.array(np.linalg.inv(tmat) @ pmat, dtype="double", order="C")

    return (
        get_primitive(cell, primitive_matrix=total_tmat, symprec=tolerance),
        total_tmat,
    )


def get_reduced_cell(
    cell: PhonopyAtoms, method="niggli", tolerance: float = 1e-5
) -> PhonopyAtoms:
    """Return a reduced cell of input cell."""
    reduced_lattice = get_reduced_bases(cell.cell, method=method, tolerance=tolerance)
    assert reduced_lattice is not None
    reduced_positions = cell.scaled_positions @ (
        cell.cell @ np.linalg.inv(reduced_lattice)
    )
    reduced_positions = reduced_positions - np.rint(reduced_positions)
    reduced_positions[:, :] += np.where(reduced_positions < 0, 1, 0)
    reduced_cell = cell.copy()
    reduced_cell.cell = reduced_lattice
    reduced_cell.scaled_positions = reduced_positions
    return reduced_cell


def get_symmetry_dataset(
    cell: PhonopyAtoms, tolerance: float = 1e-5
) -> SpglibDataset | SpglibMagneticDataset:
    """Return spglib symmetry dataset."""
    if cell.magnetic_moments is None:
        dataset = spglib.get_symmetry_dataset(cell.totuple(), symprec=tolerance)
    else:
        dataset = spglib.get_magnetic_symmetry_dataset(
            cell.totuple(), symprec=tolerance
        )

    assert dataset is not None

    return dataset


def _get_primitive_matrix_from_dataset(
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
) -> NDArray:
    """Return primitive matrix from symmetry dataset."""
    if isinstance(sym_dataset, SpglibDataset):
        centring = sym_dataset.international[0]
    else:
        spg_type = spglib.get_spacegroup_type(sym_dataset.hall_number)
        assert spg_type is not None
        centring = spg_type.international_short[0]
    return get_primitive_matrix_by_centring(centring)


def _get_standardized_unitcell(
    dataset: SpglibDataset | SpglibMagneticDataset,
) -> PhonopyAtoms:
    """Return conventional unit cell.

    This conventional unit cell can include rigid rotation with respect to
    input unit cell for which symmetry was analized.

    Parameters
    ----------
    cell : PhonopyAtoms
        Input cell.
    dataset : SpgliDataset
        Symmetry dataset of spglib.

    Returns
    -------
    PhonopyAtoms
        Convetional unit cell.

    """
    std_positions = dataset.std_positions
    std_types = dataset.std_types
    _, _, _, perm = sort_positions_by_symbols(std_types, std_positions)
    atom_data = get_atomic_data().atom_data
    if isinstance(dataset, SpglibDataset):
        return PhonopyAtoms(
            cell=dataset.std_lattice,
            scaled_positions=std_positions[perm],
            symbols=[atom_data[n][1] for n in std_types[perm]],
        )
    else:
        return PhonopyAtoms(
            cell=dataset.std_lattice,
            scaled_positions=std_positions[perm],
            symbols=[atom_data[n][1] for n in std_types[perm]],
            magnetic_moments=dataset.std_tensors[perm],
        )
