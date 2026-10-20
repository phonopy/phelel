"""Tests of velph ph_selfenergy dump_phono3py."""

from __future__ import annotations

import io
import pathlib
from collections.abc import Callable

import h5py
import numpy as np
import pytest
import spglib
from click.testing import CliRunner
from numpy.typing import NDArray
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.phonon.grid import BZGrid, get_ir_grid_points
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms

from phelel.velph.cli.ph_selfenergy.dump_phono3py import _get_bz_grid_and_id_map
from phelel.velph.cli.velph_cmd_root import cmd_root

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


def _write_ph_selfenergy_vaspout(
    filename: pathlib.Path,
    cell: PhonopyAtoms,
    mesh: int,
    n_calcs: int,
    write_positions: Callable[[h5py.Group, PhonopyAtoms], None],
) -> NDArray:
    """Write vaspout.h5 of ph_selfenergy with n_calcs self-energy calculations.

    The k-point mesh is mesh x mesh x mesh, and the irreducible k-points are in
    an order shuffled as VASP may give them.  The frequency of band b at the
    k-point k of calculation c is 10 * k + b + 100 * c, in 2 pi THz, and the
    imaginary part of the self-energy is minus the frequency, in eV.  The index
    of the k-point of each irreducible grid point of phono3py is returned.

    """
    kpoints, weights, expected = _ir_kpoints_of_vasp(
        cell.totuple(), np.diag([mesh] * 3)
    )
    n_kpoints, n_bands, temps = len(kpoints), 2 * len(cell), [100.0, 300.0]
    with h5py.File(filename, "w") as f:
        write_positions(f.create_group("results/positions"), cell)
        electrons = f.create_group("results/electron_phonon/electrons")
        electrons["self_energy_meta/ncalculators"] = n_calcs
        electrons["eigenvalues/kpoint_coords"] = kpoints
        electrons["eigenvalues/kpoints_symmetry_weight"] = weights
        for key in ("nkpx", "nkpy", "nkpz"):
            f[f"input/kpoints_elph/{key}"] = mesh
        f["input/kpoints/coordinate_space"] = np.bytes_("R")
        for c in range(n_calcs):
            freqs = 10.0 * np.arange(n_kpoints)[:, None] + np.arange(n_bands) + 100 * c
            selfen = np.zeros((1, n_bands, n_kpoints, 1, len(temps), 2))
            selfen[0, :, :, 0, :, 1] = -freqs.T[:, :, None]
            g = f.create_group(f"results/electron_phonon/phonons/self_energy_{c + 1}")
            g["phonon_freqs_ibz"] = freqs
            g["selfen_ph"] = selfen
            g["temps"] = np.array(temps)
    return expected


def test_velph_ph_selfenergy_dump_phono3py(
    nacl_prim_cell: PhonopyAtoms,
    write_vaspout_positions: Callable[[h5py.Group, PhonopyAtoms], None],
    tmp_path: pathlib.Path,
):
    """The frequencies and gammas are written in the order of phono3py.

    The frequencies are in THz, and the gammas have the axes of phono3py,
    (spin, temperature, k-point, band), in THz.

    """
    vaspout = tmp_path / "vaspout.h5"
    expected = _write_ph_selfenergy_vaspout(
        vaspout, nacl_prim_cell, 4, 1, write_vaspout_positions
    )
    result = CliRunner().invoke(
        cmd_root,
        [
            "ph_selfenergy",
            "dump_phono3py",
            "--vaspout-filename",
            str(vaspout),
            "--output-filename",
            str(tmp_path / "gamma_elph{mesh}.hdf5"),
        ],
    )
    assert result.exit_code == 0, result.output
    thz_to_ev = get_physical_units().THzToEv
    with (
        h5py.File(vaspout) as f_in,
        h5py.File(tmp_path / "gamma_elph-m444.hdf5") as f,
    ):
        kpoints = f_in["results/electron_phonon/electrons/eigenvalues/kpoint_coords"][:]
        np.testing.assert_allclose(f["qpoint"][:], kpoints[expected])
        np.testing.assert_array_equal(f["mesh"][:], [4, 4, 4])
        assert f["weight"][:].sum() == 64
        freqs = 10.0 * expected[:, None] + np.arange(4)
        np.testing.assert_allclose(f["frequency_1"][:], freqs / (2 * np.pi))
        gammas = f["gamma_1"][:]
        assert gammas.shape == (1, 2, len(expected), 4)
        np.testing.assert_allclose(
            gammas[0, 0], freqs / (thz_to_ev / (2 * np.pi)) / (2 * np.pi)
        )
        np.testing.assert_array_equal(f["temperature_1"][:], [100.0, 300.0])


def test_velph_ph_selfenergy_plot_selfenergy(
    nacl_prim_cell: PhonopyAtoms,
    write_vaspout_positions: Callable[[h5py.Group, PhonopyAtoms], None],
    tmp_path: pathlib.Path,
):
    """The plot of the phonon self-energies is written."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    vaspout = tmp_path / "vaspout.h5"
    _write_ph_selfenergy_vaspout(vaspout, nacl_prim_cell, 4, 2, write_vaspout_positions)
    result = CliRunner().invoke(
        cmd_root,
        [
            "ph_selfenergy",
            "plot",
            "selfenergy",
            "--vaspout-filename",
            str(vaspout),
            "--save",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "selfenergy.pdf").exists()


def test_velph_ph_selfenergy_dump_phono3py_many_calculations(
    nacl_prim_cell: PhonopyAtoms,
    write_vaspout_positions: Callable[[h5py.Group, PhonopyAtoms], None],
    tmp_path: pathlib.Path,
):
    """frequency_<i> is written from self_energy_<i> also for i of 10 and more.

    h5py gives the groups in the order of their names, in which self_energy_10
    comes before self_energy_2.

    """
    vaspout = tmp_path / "vaspout.h5"
    expected = _write_ph_selfenergy_vaspout(
        vaspout, nacl_prim_cell, 4, 11, write_vaspout_positions
    )
    result = CliRunner().invoke(
        cmd_root,
        [
            "ph_selfenergy",
            "dump_phono3py",
            "--vaspout-filename",
            str(vaspout),
            "--output-filename",
            str(tmp_path / "gamma_elph{mesh}.hdf5"),
        ],
    )
    assert result.exit_code == 0, result.output
    with h5py.File(tmp_path / "gamma_elph-m444.hdf5") as f:
        for i in range(1, 12):
            freqs = 10.0 * expected[:, None] + np.arange(4) + 100 * (i - 1)
            np.testing.assert_allclose(f[f"frequency_{i}"][:], freqs / (2 * np.pi))
