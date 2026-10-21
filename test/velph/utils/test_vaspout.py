"""Tests of the readers of vaspout.h5."""

from __future__ import annotations

import pathlib

import h5py
import numpy as np

from phelel.velph.utils.vaspout import (
    get_numbered_groups,
    read_crystal_structure_from_vaspout_h5,
    read_spglib_cell_from_vaspout_h5,
)


def test_get_numbered_groups(tmp_path: pathlib.Path):
    """The groups are in the order of their numbers, without the other groups.

    h5py gives transport_10 and transport_11 before transport_2.

    """
    with h5py.File(tmp_path / "vaspout.h5", "w") as f:
        parent = f.create_group("electrons")
        for n in range(1, 12):
            parent.create_group(f"transport_{n}").attrs["n"] = n
        for name in ("transport_meta", "self_energy_1", "transport_1_x"):
            parent.create_group(name)
        groups = get_numbered_groups(parent, "transport")
        assert [n for n, _ in groups] == list(range(1, 12))
        assert [group.attrs["n"] for _, group in groups] == list(range(1, 12))


def test_read_spglib_cell_from_vaspout_h5(tmp_path: pathlib.Path):
    """Two species of the same element have different numbers.

    The positions in Cartesian coordinates are given in direct coordinates.

    """
    lattice = np.diag([3.0, 4.0, 5.0])
    direct = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0]])
    with h5py.File(tmp_path / "vaspout.h5", "w") as f:
        group = f.create_group("results/positions")
        group["scale"] = 2.0
        group["lattice_vectors"] = lattice
        group["position_ions"] = direct @ (lattice * 2)
        group["direct_coordinates"] = 0
        group["number_ion_types"] = [1, 1, 1]
        group["ion_types"] = np.array([b"Fe", b"Fe", b"O"])
        cell_lattice, positions, numbers = read_spglib_cell_from_vaspout_h5(
            f, "results/positions"
        )
        cell = read_crystal_structure_from_vaspout_h5(f, "results/positions")
    np.testing.assert_allclose(cell_lattice, lattice * 2)
    np.testing.assert_allclose(positions, direct)
    np.testing.assert_array_equal(numbers, [1, 2, 3])
    assert cell.symbols == ["Fe", "Fe", "O"]
