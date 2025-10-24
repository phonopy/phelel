"""Tests cli/utils.py."""

import copy
import io
import itertools
from collections.abc import Callable

import numpy as np
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.atoms import PhonopyAtoms

from phelel.velph.cli.utils import get_reduced_cell, get_scheduler_dict
from phelel.velph.templates import default_template_dict


def test_get_scheduler_dict():
    """Test get_scheduler_dict."""
    toml_dict = copy.deepcopy(default_template_dict)
    scheduler_dict = get_scheduler_dict(toml_dict, "phelel")
    assert scheduler_dict["vasp_binary"] == "vasp_std"

    phonon_dict = copy.deepcopy(toml_dict["vasp"]["phelel"])
    toml_dict["vasp"]["phelel"]["phonon"] = phonon_dict
    scheduler_dict = get_scheduler_dict(toml_dict, "phelel.phonon")
    assert scheduler_dict["vasp_binary"] == "vasp_std"

    toml_dict["vasp"]["phelel"]["scheduler"] = {"vasp_binary": "vasp_gam"}
    scheduler_dict = get_scheduler_dict(toml_dict, "phelel")
    assert scheduler_dict["vasp_binary"] == "vasp_gam"


def test_get_reduced_cell_bi2te3(
    helper_methods: Callable, bi2te3_prim_cell: PhonopyAtoms
):
    """Test of get_reduced_cell using Bi2Te3 primitive cell.

    Input cell is primitive rhombohedral.

    """
    ref_cell_str = """lattice:
- [    -2.221502746054457,     3.847755625320100,     0.000000000000000 ] # a
- [    -4.443005492108914,     0.000000000000000,     0.000000000000000 ] # b
- [    -2.221502746054457,     1.282585208440033,    10.479344379716814 ] # c
points:
- symbol: Bi # 1
  coordinates: [  0.398482502929835,  0.398482502929836,  0.804552491210493 ]
  mass: 208.980400
- symbol: Bi # 2
  coordinates: [  0.601517497070165,  0.601517497070165,  0.195447508789506 ]
  mass: 208.980400
- symbol: Te # 3
  coordinates: [  0.213380750346874,  0.213380750346874,  0.359857748959378 ]
  mass: 127.600000
- symbol: Te # 4
  coordinates: [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 127.600000
- symbol: Te # 5
  coordinates: [  0.786619249653126,  0.786619249653126,  0.640142251040622 ]
  mass: 127.600000
"""
    ref_cell = read_cell_yaml(io.StringIO(ref_cell_str))
    reduced_cell = get_reduced_cell(bi2te3_prim_cell)
    ref_lengths = np.linalg.norm(ref_cell.cell, axis=1)
    reduced_lengths = np.linalg.norm(reduced_cell.cell, axis=1)

    # Check lenghts of basis vectors.
    is_found = False
    for ref_perm in itertools.permutations(ref_lengths):
        for reduced_perm in itertools.permutations(reduced_lengths):
            if np.allclose(ref_perm, reduced_perm):
                is_found = True
                break
    assert is_found

    if np.allclose(ref_cell.cell, reduced_cell.cell):
        helper_methods.compare_positions_with_order(
            reduced_cell.scaled_positions, ref_cell.scaled_positions, ref_cell.cell
        )
    else:
        msg = (
            "Reduced cell algorithm may be sensitive to the numerical precision of "
            "computers. Therefore this failure might happen due to it. Please "
            "recondier this test."
        )
        raise AssertionError(msg)
