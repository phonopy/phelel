"""Test utils."""

import numpy as np
import pytest

from phelel.velph.cli.utils import kspacing_to_mesh


@pytest.mark.parametrize("use_grg", [True, False])
def test_kspacing_to_mesh_FCC_prim(si_prim_cell, use_grg):
    """Test to convert kspacing to mesh by kspacing_to_mesh."""
    kpoints_dict = {"kspacing": 0.5}
    kspacing_to_mesh(kpoints_dict, si_prim_cell, use_grg)
    if use_grg:
        np.testing.assert_array_equal(
            kpoints_dict["mesh"], [[-2, 2, 2], [2, -2, 2], [2, 2, -2]]
        )
    else:
        np.testing.assert_array_equal(kpoints_dict["mesh"], [4, 4, 4])


@pytest.mark.parametrize("use_grg", [True, False])
def test_kspacing_to_mesh_BCT(tio2_prim_cell, use_grg):
    """Test to convert kspacing to mesh by kspacing_to_mesh."""
    kpoints_dict = {"kspacing": 0.5}
    if use_grg:
        kspacing_to_mesh(kpoints_dict, tio2_prim_cell, use_grg)
        np.testing.assert_array_equal(
            kpoints_dict["mesh"], [[0, 3, 3], [3, 0, 3], [1, 1, 0]]
        )
    else:
        with pytest.raises(RuntimeError):
            kspacing_to_mesh(kpoints_dict, tio2_prim_cell, use_grg)


@pytest.mark.parametrize("use_grg", [True, False])
def test_kspacing_to_mesh_hexagonal(aln_cell, use_grg):
    """Test to convert kspacing to mesh by kspacing_to_mesh."""
    kpoints_dict = {"kspacing": 0.5}
    kspacing_to_mesh(kpoints_dict, aln_cell, use_grg)
    np.testing.assert_array_equal(kpoints_dict["mesh"], [5, 5, 3])
