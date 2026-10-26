"""Tests of velph/utils/mesh.py."""

from __future__ import annotations

import numpy as np
import pytest
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_supercell

from phelel.velph.utils.mesh import kspacing_to_mesh


@pytest.mark.parametrize("use_grg", [True, False])
def test_kspacing_to_mesh_fcc_prim(si_prim_cell: PhonopyAtoms, use_grg: bool):
    """Test the GR-grid and the regular mesh of a primitive cell of FCC."""
    mesh = kspacing_to_mesh(0.5, si_prim_cell, use_grg)
    if use_grg:
        assert mesh == [[-3, 3, 3], [3, -3, 3], [3, 3, -3]]
    else:
        assert mesh == [4, 4, 4]


@pytest.mark.parametrize("use_grg", [True, False])
def test_kspacing_to_mesh_bct_prim(tio2_prim_cell: PhonopyAtoms, use_grg: bool):
    """Test that the GR-grid is returned for a primitive cell of BCT.

    The regular mesh breaks the symmetry of the primitive cell of a
    body-centred lattice, so the GR-grid is returned even with use_grg=False.

    """
    mesh = kspacing_to_mesh(0.5, tio2_prim_cell, use_grg)
    assert mesh == [[0, 4, 4], [4, 0, 4], [2, 2, 0]]


@pytest.mark.parametrize("use_grg", [True, False])
def test_kspacing_to_mesh_hexagonal(aln_cell: PhonopyAtoms, use_grg: bool):
    """Test that the GR-grid of a hexagonal cell is the regular mesh."""
    assert kspacing_to_mesh(0.5, aln_cell, use_grg) == [5, 5, 3]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("use_grg", [True, False])
def test_kspacing_to_mesh_follows_vasp_kspacing(nacl_cell: PhonopyAtoms, use_grg: bool):
    """Test that the mesh follows VASP KSPACING for a cell of several lattice points.

    VASP gives N_i = max(1, ceiling(|b_i| 2 pi / KSPACING)).  For the
    conventional unit cell of NaCl (a = 5.69 Angstrom) with spacing 0.25,
    |b_i| 2 pi / KSPACING is 4.42, which gives 5.  The GR-grid cannot be used
    for this cell, and use_grg=True is ignored without the warning of phonopy.

    """
    assert kspacing_to_mesh(0.25, nacl_cell, use_grg) == [5, 5, 5]


def test_kspacing_to_mesh_tolerance(si_prim_cell: PhonopyAtoms):
    """Test that the symmetry of the cell is found with tolerance.

    With the first basis vector of the primitive cell of Si longer by 1e-4,
    the GR-grid of FCC is found only with the larger tolerance.

    """
    cell = si_prim_cell.copy()
    lattice = cell.cell.copy()
    lattice[0] *= 1.0001
    cell.cell = lattice
    fcc_grg = [[-4, 4, 4], [4, -4, 4], [4, 4, -4]]
    assert kspacing_to_mesh(0.3, cell, True, tolerance=1e-2) == fcc_grg
    assert kspacing_to_mesh(0.3, cell, True) != fcc_grg


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("use_grg", [True, False])
def test_kspacing_to_mesh_broken_symmetry(tio2_prim_cell: PhonopyAtoms, use_grg: bool):
    """Test that RuntimeError is raised when no mesh respects the symmetry.

    The 2x2x2 supercell of the primitive cell of BCT has several lattice
    points, so the GR-grid cannot be used, and the regular mesh breaks the
    symmetry.  The GR-grid is not tried, so phonopy gives no warning.

    """
    supercell = get_supercell(tio2_prim_cell, np.diag([2, 2, 2]))
    with pytest.raises(RuntimeError, match="Grid symmetry is broken."):
        kspacing_to_mesh(0.5, supercell, use_grg)


@pytest.mark.parametrize(
    "magmoms",
    [[0.0, 0.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [1.0, 1.0]],
)
def test_kspacing_to_mesh_fcc_prim_with_magmoms(
    si_prim_cell: PhonopyAtoms, magmoms: list
):
    """Test the GR-grid of a primitive cell of FCC with magnetic moments.

    With zero magnetic moments, the magnetic space group is a grey group, in
    which every operation appears twice, without and with time reversal.  The
    identity combined with time reversal is not a lattice translation, and the
    cell is still a primitive cell.

    """
    cell = si_prim_cell.copy()
    cell.magnetic_moments = magmoms
    assert kspacing_to_mesh(0.5, cell, True) == [[-3, 3, 3], [3, -3, 3], [3, 3, -3]]


def test_kspacing_to_mesh_bct_prim_with_zero_magmoms(tio2_prim_cell: PhonopyAtoms):
    """Test the GR-grid of a primitive cell of BCT with zero magnetic moments.

    The regular mesh breaks the symmetry of this cell, so the GR-grid is
    returned with use_grg=False, as without magnetic moments.

    """
    cell = tio2_prim_cell.copy()
    cell.magnetic_moments = [0.0] * len(cell)
    assert kspacing_to_mesh(0.5, cell, False) == [[0, 4, 4], [4, 0, 4], [2, 2, 0]]


@pytest.mark.filterwarnings("error")
def test_kspacing_to_mesh_nacl_with_zero_magmoms(nacl_cell: PhonopyAtoms):
    """Test that a cell of several lattice points is not taken as primitive.

    The pure translations of the conventional unit cell of NaCl appear without
    time reversal also with zero magnetic moments, so the GR-grid is not used.

    """
    cell = nacl_cell.copy()
    cell.magnetic_moments = [0.0] * len(cell)
    assert kspacing_to_mesh(0.25, cell, True) == [5, 5, 5]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("magmoms", [[1.0, -1.0], [[0, 0, 1.0], [0, 0, -1.0]]])
def test_kspacing_to_mesh_type4_magnetic_space_group(magmoms: list):
    """Test that the GR-grid is not used for a type-4 magnetic space group.

    In the conventional cell of bcc with antiparallel moments, the
    translation to the body centre is combined with time reversal.  As in
    GridMatrix of phonopy, the cell is not taken as a primitive cell, so the
    regular mesh is returned with use_grg=True without the warning of phonopy.

    """
    cell = PhonopyAtoms(
        cell=np.eye(3) * 2.87,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        symbols=["Cr", "Cr"],
        magnetic_moments=magmoms,
    )
    assert kspacing_to_mesh(0.5, cell, True) == [5, 5, 5]
