"""Calculation of Eliashberg function."""

from __future__ import annotations

import h5py
import numpy as np
import spglib
from phono3py.other.kaccum import KappaDOSTHM
from phono3py.phonon.grid import BZGrid, get_grid_point_from_address, get_ir_grid_points
from phonopy.units import THzToEv


def get_Eliashberg_function(
    h5_filename: str, num_sampling_points: int = 201, is_plot: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get Eliashberg function."""
    vals = _collect_data_from_vaspout(h5_filename)
    lattice = vals["lattice"]
    positions = vals["positions"]
    numbers = vals["numbers"]
    k_gen_vecs = vals["k_gen_vecs"]
    ir_kpoints = vals["ir_kpoints"]
    ir_kpoints_weights = vals["ir_kpoints_weights"]
    freqs = vals["freqs"]  # [ikpt, ib]
    temps = vals["temps"]
    dos_at_ef = vals["dos_at_ef"] * (THzToEv / (2 * np.pi))  # [ispin, itemp]
    gamma = vals["gamma"] / (THzToEv / (2 * np.pi))  # [ispin, ib, ikpt, itemp]
    a2f_vasp = vals["a2f_vasp"]
    freq_points_vasp = vals["freq_points_vasp"]

    sym_dataset = spglib.get_symmetry_dataset((lattice, positions, numbers))
    mesh = np.linalg.inv(lattice.T @ k_gen_vecs).T
    mesh = np.rint(mesh).astype(int)
    bz_grid = BZGrid(mesh, lattice=lattice, symmetry_dataset=sym_dataset)
    ir_grid_points, ir_grid_weights, ir_grid_map = get_ir_grid_points(bz_grid)

    # ir_gps indices in phonopy are mapped to those in vasp.
    ir_addresss = np.rint(ir_kpoints @ mesh).astype(int)
    gps = get_grid_point_from_address(ir_addresss, bz_grid.D_diag)
    irgp = ir_grid_map[gps]
    id_map = [np.where(irgp == gp)[0][0] for gp in ir_grid_points]
    ir_kpoints_weights *= np.linalg.det(mesh)
    assert (np.abs(ir_grid_weights - ir_kpoints_weights[id_map]) < 1e-8).all()

    # Convert kpoints order to that of phono3py.
    freqs = freqs[id_map]
    gamma = gamma[:, :, id_map, :].transpose(0, 3, 2, 1)

    freq_points, a2f = _calculate_Eliashberg_function(
        gamma,
        freqs,
        dos_at_ef,
        bz_grid,
        ir_grid_points,
        ir_grid_weights,
        ir_grid_map,
        num_sampling_points,
    )

    if is_plot:
        _plot_a2f_comparison(
            a2f_vasp,
            freq_points_vasp,
            a2f,
            freq_points,
            temps,
        )

    return freq_points, a2f, temps


def _calculate_Eliashberg_function(
    gamma: np.ndarray,
    freqs: np.ndarray,
    dos_at_ef: np.ndarray,
    bz_grid: BZGrid,
    ir_grid_points: np.ndarray,
    ir_grid_weights: np.ndarray,
    ir_grid_map: np.ndarray,
    num_sampling_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Eliashberg function.

    Returns
    -------
    freq_points : np.ndarray
        Frequency points in 2piTHz unit.
    a2f : np.ndarray
        Eliashberg function in 1/2piTHz unit.
        shape=(ispin, itemp, freq_points, (J,I))

    """
    # a2f at qj in 1/2piTHz unit [ispin, itemp, ikpt, ib]
    a2f_at_qj = np.zeros(gamma.shape, dtype="double")
    a2f = []
    for ispin in range(gamma.shape[0]):
        # delta function is in 1/2piTHz unit.
        for itemp in range(gamma.shape[1]):
            a2f_at_qj[ispin, itemp, :, :] = (
                1
                / (2 * np.pi)
                * gamma[ispin, itemp, :, :]
                / freqs
                / dos_at_ef[ispin, itemp]
            )

        kappados = KappaDOSTHM(
            a2f_at_qj[ispin, :, :, :, None],
            freqs,
            bz_grid,
            ir_grid_points=ir_grid_points,
            ir_grid_weights=ir_grid_weights,
            ir_grid_map=ir_grid_map,
            num_sampling_points=num_sampling_points,
        )
        freq_points, _a2f = kappados.get_kdos()
        a2f.append(_a2f[:, :, :, 0])

    return freq_points, np.array(a2f)


def _plot_a2f_comparison(
    a2f_vasp: np.ndarray,
    freq_points_vasp: np.ndarray,
    a2f: np.ndarray,
    freq_points: np.ndarray,
    temps: np.ndarray,
):
    """Plot a2F comparison."""
    import matplotlib.pyplot as plt

    plt.figure()
    if len(a2f) == 1:
        updown = [""]
    else:
        updown = ["up", "down"]
    for ispin, a2f_spin in enumerate(a2f):
        for itemp, temp in enumerate(temps):
            plt.plot(
                freq_points_vasp,
                a2f_vasp[ispin, itemp, :],
                label=f"{temp} K {updown[ispin]} (VASP)",
            )
            plt.plot(
                freq_points,
                a2f_spin[itemp, :, 1],
                ".",
                label=f"{temp} K {updown[ispin]} (phelel)",
            )
    plt.xlabel("Frequency (2piTHz)")
    plt.ylabel("a2F")
    plt.title("a2F vs Frequency")
    plt.legend()
    plt.show()


def _collect_data_from_vaspout(h5_filename: str) -> dict:
    """Collect data from vaspout.h5 file."""
    vals = {}
    with h5py.File(h5_filename) as f:
        # [itemp]
        vals["temps"] = f["results/electron_phonon/phonons/self_energy_1/temps"][:]
        # [ikpt, 3]
        vals["ir_kpoints"] = f[
            "results/electron_phonon/phonons/self_energy_1/kpoint_coords"
        ][:]
        vals["ir_kpoints_weights"] = f[
            "results/electron_phonon/phonons/self_energy_1/kpoints_symmetry_weight"
        ][:]
        lat_scale = f["results/positions/scale"][()]
        vals["lattice"] = f["results/positions/lattice_vectors"][:] * lat_scale
        number_ion_types = f["results/positions/number_ion_types"][:]
        numbers = []
        for i, nums in enumerate(number_ion_types):
            numbers += [i + 1] * nums
        vals["numbers"] = np.array(numbers)
        vals["positions"] = f["results/positions/position_ions"][:]
        # row vectors in python (and KPOINTS file, too)
        vals["k_gen_vecs"] = f[
            "results/electron_phonon/phonons/self_energy_1/kpoint_generating_vectors"
        ][:]
        # [ispin, itemp] in 1/eV
        vals["dos_at_ef"] = f[
            "results/electron_phonon/phonons/self_energy_1/dos_at_ef"
        ][:, :]
        # [ispin, ib, ikpt , 1, temp, (re,im)] in eV
        vals["gamma"] = (
            -2
            * f["results/electron_phonon/phonons/self_energy_1/selfen_ph"][
                :, :, :, 0, :, 1
            ]
        )
        # [ikpt, ib] in 2piTHz
        vals["freqs"] = f[
            "results/electron_phonon/phonons/self_energy_1/phonon_freqs_ibz"
        ][:]
        # [ispin, itemp, freq_points]
        vals["a2f_vasp"] = f["results/electron_phonon/phonons/self_energy_1/a2F"][:]
        # in 2piTHz
        vals["freq_points_vasp"] = f[
            "results/electron_phonon/phonons/self_energy_1/frequency_grid"
        ][:]

    return vals
