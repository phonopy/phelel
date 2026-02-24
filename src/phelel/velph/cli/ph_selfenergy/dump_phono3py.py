"""velph command line tool / velph-ph_selfenergy-dump_phono3py."""

import click
import h5py
import numpy as np
from numpy.typing import NDArray
from phono3py.phonon.grid import BZGrid

from phelel.velph.utils.vasp import (
    convert_ir_kpoints_from_VASP_to_phono3py,
    read_freqs_and_ph_gammas_from_vaspout_h5,
)


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
        with h5py.File(output_filename, "w") as f_out:
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

            click.echo(f'Dumped ph_selfenergy data to "{output_filename}".')


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
    numbers = []
    for i, nums in enumerate(number_ion_types):
        numbers += [i + 1] * nums
    numbers = np.array(numbers, dtype=int)
    if "basis_vectors" in f_h5py["input/kpoints_elph"]:  # type: ignore
        # When reading "basis_vectors" in python, the following 3x3 ndarray
        # is obtained:
        #   [a_m*_x, a_m*_y, a_m*_z]
        #   [b_m*_x, b_m*_y, b_m*_z]
        #   [c_m*_x, c_m*_y, c_m*_z]
        k_gen_vecs: NDArray = f_h5py["input/kpoints_elph/basis_vectors"][:]  # type:ignore
    else:
        k_gen_vecs: NDArray = np.diag(
            [
                1.0 / f_h5py[f"input/kpoints_elph/{key}"][()]  # type: ignore
                for key in ("nkpx", "nkpy", "nkpz")
            ]
        )
    assert f_h5py["input/kpoints/coordinate_space"][()] == b"R"  # type: ignore

    id_map, bz_grid, _, _, _ = convert_ir_kpoints_from_VASP_to_phono3py(
        lattice, positions, numbers, k_gen_vecs, ir_kpoints, ir_kpoints_weights
    )
    weights = ir_kpoints_weights[id_map] / np.linalg.det(k_gen_vecs)
    weights = np.rint(weights).astype(int)
    return id_map, ir_kpoints[id_map], weights, bz_grid
