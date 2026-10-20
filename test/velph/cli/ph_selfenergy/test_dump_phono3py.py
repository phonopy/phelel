"""Tests of velph ph_selfenergy dump_phono3py."""

from __future__ import annotations

import io

import numpy as np
import pytest
import spglib
from numpy.typing import NDArray
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.phonon.grid import BZGrid, get_ir_grid_points

from phelel.velph.cli.ph_selfenergy.dump_phono3py import _get_bz_grid_and_id_map

# Primitive cell of rutile-type TiO2 in I4_1/amd, a body-centred tetragonal
# lattice, whose k-point mesh of velph init is a generalized regular grid.
_TIO2 = """primitive_cell:
  lattice:
  - [    -1.888070425000000,     1.888070425000000,     4.790243149999999 ]
  - [     1.888070425000000,    -1.888070424999999,     4.790243149999999 ]
  - [     1.888070425000000,     1.888070424999999,    -4.790243149999999 ]
  points:
  - symbol: O
    coordinates: [  0.457386310000000,  0.957386310000000,  0.500000000000000 ]
  - symbol: O
    coordinates: [  0.707386310000000,  0.707386310000000,  1.000000000000000 ]
  - symbol: O
    coordinates: [  0.042613690000000,  0.542613690000000,  0.500000000000000 ]
  - symbol: O
    coordinates: [  0.292613690000000,  0.292613690000000,  1.000000000000000 ]
  - symbol: Ti
    coordinates: [  0.500000000000000,  0.500000000000000,  1.000000000000000 ]
  - symbol: Ti
    coordinates: [  0.250000000000000,  0.750000000000000,  0.500000000000000 ]
"""


def _ir_kpoints_of_vasp(cell: tuple, mesh: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Return irreducible k-points as VASP reports them, and their order.

    The k-points are the irreducible grid points of the mesh in fractional
    coordinates, with weights that sum to one, in an order shuffled by a fixed
    seed.  The order is returned as the index of the k-point of each
    irreducible grid point.

    """
    sym_dataset = spglib.get_symmetry_dataset(cell)
    bz_grid = BZGrid(mesh, lattice=cell[0], symmetry_dataset=sym_dataset)
    ir_grid_points, ir_weights, _ = get_ir_grid_points(bz_grid)
    d = bz_grid.D_diag
    addresses = np.array(
        [
            ir_grid_points % d[0],
            (ir_grid_points // d[0]) % d[1],
            ir_grid_points // (d[0] * d[1]),
        ]
    ).T
    kpoints = addresses @ bz_grid.QDinv.T
    perm = np.random.default_rng(7).permutation(len(ir_grid_points))
    weights = ir_weights / ir_weights.sum()
    return kpoints[perm], weights[perm], np.argsort(perm)


@pytest.mark.parametrize(
    "mesh,n_ir",
    [
        # Generalized regular grid, for which VASP gives 50 irreducible
        # k-points (see test_write_kpoints_mesh_mode_grg_tio2_prim).
        ([[0, 8, 8], [8, 0, 8], [3, 3, 0]], 50),
        ([[4, 0, 0], [0, 4, 0], [0, 0, 4]], None),
    ],
)
def test_id_map(mesh: list[list[int]], n_ir: int | None):
    """id_map puts the irreducible k-points of VASP in the order of phono3py.

    The mesh is given by its generating vectors, as in vaspout.h5.

    """
    cell = read_cell_yaml(io.StringIO(_TIO2), cell_type="primitive").totuple()
    kpoints, weights, expected = _ir_kpoints_of_vasp(cell, np.array(mesh))
    if n_ir is not None:
        assert len(kpoints) == n_ir
    k_gen_vecs = np.linalg.inv(np.array(mesh, dtype=float)).T
    id_map, bz_grid = _get_bz_grid_and_id_map(
        cell[0], cell[1], cell[2], k_gen_vecs, kpoints, weights
    )
    np.testing.assert_array_equal(id_map, expected)
    assert np.prod(bz_grid.D_diag) == round(abs(np.linalg.det(mesh)))
