"""velph command line tool / velph-ph_selfenergy-dump_phono3py."""

from typing import cast

import click
import h5py
import numpy as np
import spglib
from numpy.typing import NDArray
from phonopy.phonon.grid import BZGrid, get_ir_kpoint_map
from spglib import SpgCell

from phelel.velph.utils.vaspout import read_freqs_and_ph_gammas_from_vaspout_h5


def dump_phono3py(vaspout_filename: str, output_filename: str):
    """Dump phono3py input files from vaspout.h5.

    gammas: [ispin, ib, ikpt , nw, temp] -> [ispin, temp, ikpt, ib]
    freqs: [ikpt, ib] -> [ikpt, ib]

    """
    with h5py.File(vaspout_filename, "r") as f:
        id_map, ir_kpoints, weights, bz_grid = _collect_data_from_vaspout(f)
        freqs_calcs, gammas_calcs, temps_calcs, indices = (
            read_freqs_and_ph_gammas_from_vaspout_h5(f)
        )
        _output_filename = output_filename.format(
            mesh="-m" + "".join(map(str, bz_grid.D_diag))
        )
        with h5py.File(_output_filename, "w") as f_out:
            f_out.create_dataset("qpoint", data=ir_kpoints)
            f_out.create_dataset("weight", data=weights)
            f_out.create_dataset("mesh", data=bz_grid.D_diag)
            for i, freqs, gammas, temps in zip(
                indices, freqs_calcs, gammas_calcs, temps_calcs, strict=True
            ):
                _freqs = freqs[id_map] / (2 * np.pi)  # convert to THz
                _gammas = np.transpose(gammas[:, :, id_map, 0, :], (0, 3, 2, 1)) / (
                    2 * np.pi
                )  # convert to THz
                f_out.create_dataset(f"frequency_{i}", data=_freqs)
                f_out.create_dataset(f"gamma_{i}", data=_gammas)
                f_out.create_dataset(f"temperature_{i}", data=temps)
                click.echo(f"calc-{i} temperatures: {temps}")

            click.echo(f'Phonon self-energy data was written to "{_output_filename}".')


def _collect_data_from_vaspout(
    f_h5py: h5py.File,
) -> tuple[NDArray, NDArray, NDArray, BZGrid]:
    """Collect crystal structure information from vaspout.h5 file."""
    # [ikpt, 3]
    ir_kpoints: NDArray = f_h5py[
        "results/electron_phonon/electrons/eigenvalues/kpoint_coords"
    ][:]  # type: ignore
    ir_kpoints_weights: NDArray = f_h5py[
        "results/electron_phonon/electrons/eigenvalues/kpoints_symmetry_weight"
    ][:]  # type: ignore
    lat_scale = f_h5py["results/positions/scale"][()]  # type:ignore
    # lattice: row vectors
    lattice: NDArray = f_h5py["results/positions/lattice_vectors"][:] * lat_scale  # type: ignore
    positions: NDArray = f_h5py["results/positions/position_ions"][:]  # type: ignore
    number_ion_types: NDArray = f_h5py["results/positions/number_ion_types"][:]  # type: ignore
    # The atoms of the i-th species, counted from 1, have the number i.
    numbers = np.repeat(np.arange(1, len(number_ion_types) + 1), number_ion_types)
    k_gen_vecs: NDArray
    if "basis_vectors" in f_h5py["input/kpoints_elph"]:  # type: ignore
        # When reading "basis_vectors" in python, the following 3x3 ndarray
        # is obtained:
        #   [a_m*_x, a_m*_y, a_m*_z]
        #   [b_m*_x, b_m*_y, b_m*_z]
        #   [c_m*_x, c_m*_y, c_m*_z]
        k_gen_vecs = f_h5py["input/kpoints_elph/basis_vectors"][:]  # type:ignore
    else:
        k_gen_vecs = np.diag(
            [
                1.0 / f_h5py[f"input/kpoints_elph/{key}"][()]  # type: ignore
                for key in ("nkpx", "nkpy", "nkpz")
            ]
        )
    assert f_h5py["input/kpoints/coordinate_space"][()] == b"R"  # type: ignore

    id_map, bz_grid = _get_bz_grid_and_id_map(
        lattice, positions, numbers, k_gen_vecs, ir_kpoints, ir_kpoints_weights
    )
    weights = ir_kpoints_weights[id_map] / np.linalg.det(k_gen_vecs)
    weights = np.rint(weights).astype(int)
    return id_map, ir_kpoints[id_map], weights, bz_grid


def _get_bz_grid_and_id_map(
    lattice: NDArray,
    positions: NDArray,
    numbers: NDArray,
    k_gen_vecs: NDArray,
    ir_kpoints: NDArray,
    ir_kpoints_weights: NDArray,
) -> tuple[NDArray, BZGrid]:
    """Return the order of the irreducible k-points of VASP for phono3py.

    The BZGrid of phono3py is made of the crystal structure and of the
    generating vectors of the k-point mesh of the electron-phonon calculation,
    k_gen_vecs, whose rows are the vectors in the basis of the reciprocal basis
    vectors.  The mesh can be a generalized regular grid.

    Returns
    -------
    id_map : np.ndarray
        Indices of the irreducible k-points of VASP, ir_kpoints, in the order
        of the irreducible grid points of phono3py, so that
        values_of_vasp[id_map] are the values in the order of phono3py.
    bz_grid : BZGrid
        BZGrid of the mesh.

    """
    cell = cast(SpgCell, (lattice, positions, numbers))
    sym_dataset = spglib.get_symmetry_dataset(cell)
    mesh = np.rint(np.linalg.inv(k_gen_vecs.T)).astype(int)
    bz_grid = BZGrid(mesh, lattice=lattice, symmetry_dataset=sym_dataset)
    id_map = get_ir_kpoint_map(ir_kpoints, ir_kpoints_weights, bz_grid)
    return id_map, bz_grid
