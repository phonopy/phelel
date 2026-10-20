"""Readers of vaspout.h5, the output file of VASP in HDF5."""

from __future__ import annotations

import dataclasses
import os
import pathlib

import h5py
import numpy as np
from numpy.typing import NDArray
from phonopy.physical_units import get_physical_units
from phonopy.structure.cells import PhonopyAtoms

from phelel.velph.errors import VelphError


def get_reclat_from_vaspout(fp_vaspout: h5py.File) -> NDArray[np.float64]:
    """Return reciprocal space basis vectors.

    Returns
    -------
    reclat : np.ndarray
        Reciprocal basis vectors in row vectors.
        shape=(3, 3)

    """
    # Basis vectors in direct space in column vectors
    lattice = np.transpose(
        fp_vaspout["input"]["poscar"]["lattice_vectors"][:]  # type: ignore
        * fp_vaspout["input"]["poscar"]["scale"][()]  # type: ignore
    )
    # Basis vectors in reciprocal space in row vectors
    reclat = 2 * np.pi * np.linalg.inv(lattice)
    return reclat


def read_crystal_structure_from_vaspout_h5(
    f_vaspout_h5: h5py.File, group: str
) -> PhonopyAtoms:
    """Read crystal structure from vaspout.h5."""
    direct = int(f_vaspout_h5[f"{group}/direct_coordinates"][()])  # type: ignore
    scale = f_vaspout_h5[f"{group}/scale"][()]  # type: ignore
    lattice: NDArray = f_vaspout_h5[f"{group}/lattice_vectors"][:] * scale  # type: ignore
    positions: NDArray = f_vaspout_h5[f"{group}/position_ions"][:]  # type: ignore
    number_ion_types: NDArray = f_vaspout_h5[f"{group}/number_ion_types"][:]  # type: ignore
    ion_types: NDArray = f_vaspout_h5[f"{group}/ion_types"][:]  # type: ignore

    symbols = []
    for symbol, number in zip(ion_types, number_ion_types, strict=True):
        symbols += [symbol.decode()] * number

    if not direct:
        positions = positions @ np.linalg.inv(lattice)

    assert len(symbols) == len(positions)

    cell = PhonopyAtoms(cell=lattice, scaled_positions=positions, symbols=symbols)

    return cell


def read_freqs_and_ph_gammas_from_vaspout_h5(
    f_h5py: h5py.File,
) -> tuple[list[NDArray], list[NDArray], list[NDArray], list[int]]:
    """Read phonon frequencies and phonon full-linewidths from vaspout.h5.

    Returns
    -------
    freqs_calcs : list[np.ndarray]
        List of phonon frequencies for each self-energy calculation in 2piTHz.
        [[ikpt, ib], ...]
    gammas_calcs : list[np.ndarray]
        List of imaginary part of phonon self-energies for each self-energy
        calculation in 2piTHz. Positive sign is chosen.
        [[ispin, ib, ikpt , nw, temp], ...]
    indices : list[int]
        List of self-energy calculation indices in increasing order, [1, 2,
        ...] for self_energy_1, self_energy_2, ...  The lists above are in this
        order.

    """
    THzToEv = get_physical_units().THzToEv

    f_elph = f_h5py["results/electron_phonon/phonons"]
    indices = []
    for key in f_elph:  # type: ignore
        if "self_energy_" in key:
            # self_energy_1, self_energy_2, ...
            index = key.split("_")[-1]
            if index.isdigit():
                indices.append(int(index))

    # h5py gives self_energy_10 before self_energy_2, in the order of names.
    indices.sort()
    gammas_calcs = []
    freqs_calcs = []
    temps_calcs = []
    for index in indices:
        selfen = f_elph[f"self_energy_{index}"]  # type: ignore
        # Imag part of self-energy [ispin, ib, ikpt , nw, temp] in eV
        selfen_ph: NDArray = selfen["selfen_ph"][:, :, :, :, :, 1]  # type: ignore
        # [ikpt, ib] in 2piTHz
        freqs: NDArray = selfen["phonon_freqs_ibz"][:, :]  # type: ignore
        temps: NDArray = selfen["temps"][:]  # type: ignore
        gammas_calcs.append(-selfen_ph / (THzToEv / (2 * np.pi)))
        freqs_calcs.append(freqs)
        temps_calcs.append(temps)

    assert len(gammas_calcs) == int(
        f_h5py["results/electron_phonon/electrons/self_energy_meta/ncalculators"][()]  # type: ignore
    )

    return freqs_calcs, gammas_calcs, temps_calcs, indices


@dataclasses.dataclass(frozen=True)
class ElectronDos:
    """Electronic DOS of vaspout.h5.

    energies has shape (energies,).  dos has shape (spin, energies), where
    spin is 1, or 2 for a spin-polarized calculation.  efermi is the Fermi
    level of the DOS calculation.

    """

    energies: NDArray
    dos: NDArray
    efermi: float


def read_electron_dos(vaspout_filename: pathlib.Path) -> ElectronDos:
    """Return the electronic DOS of vaspout.h5.

    The DOS of KPOINTS_OPT is read when vaspout.h5 has it.  For a
    non-collinear calculation, whose dos has four rows, only the first row,
    the total DOS, is kept.  VelphError is raised when vaspout.h5 has no
    electronic DOS.

    """
    with h5py.File(vaspout_filename, "r") as f:
        for name in ("results/electron_dos_kpoints_opt", "results/electron_dos"):
            if name in f:
                group = f[name]
                break
        else:
            raise VelphError(f'No electronic DOS is found in "{vaspout_filename}".')
        dos: NDArray = group["dos"][:]  # type: ignore
        energies: NDArray = group["energies"][:]  # type: ignore
        efermi = float(group["efermi"][()])  # type: ignore
    if len(dos) == 4:
        dos = dos[:1]
    return ElectronDos(energies=energies, dos=dos, efermi=efermi)


def estimate_elph_selfen_band_stop(
    vaspout_path: str | os.PathLike,
    energy_threshold: float = 0.5,
    occupation_condition: float = 1e-10,
) -> int | None:
    """Estimate elph_selfen_band_stop from eigenvalues in el-DOS result.

    velph does not use this function, because VASP estimates a similar value
    itself.  It is kept for use by hand.

    Parameters
    ----------
    vaspout_path : str or os.PathLike
        "vaspout.h5" path.
    energy_threshold : float
        Energy threshold (gap) in eV to estimate elph_selfen_band_stop. Default
        is 0.5 eV.
    occupation_condition : float
        Condition to determine either if bands are occupied or not. This value
        is used as occupation_number > occupation_condition. Default is 1e-10.

    Returns
    -------
    int or None
        Possibly occupied band index by counting from 0. Even if
        occupation number is zero, if some of bands can be occuped by
        excitation, this band index is returned.

    """
    with h5py.File(vaspout_path) as f:
        eigenvalues: NDArray = f[
            "results/electron_eigenvalues_kpoints_opt/eigenvalues"
        ][:]  # type: ignore
        occupations: NDArray = f[
            "results/electron_eigenvalues_kpoints_opt/fermiweights"
        ][:]  # type: ignore

    nbands = eigenvalues.shape[2]
    unoccupied_band_index = None
    for i in range(nbands):
        if (occupations[:, :, i] < occupation_condition).all():
            unoccupied_band_index = i
            break
    if unoccupied_band_index is None:
        return None

    occupied_eigvals = np.sort(
        np.extract(
            occupations[:, :, unoccupied_band_index - 1] > occupation_condition * 0.9,
            eigenvalues[:, :, unoccupied_band_index - 1],
        )
    )

    max_occupied_eigval = np.max(occupied_eigvals)

    for band_index in range(unoccupied_band_index, nbands):
        min_unoccupied = np.min(eigenvalues[:, :, band_index])
        if max_occupied_eigval + energy_threshold < min_unoccupied:
            return band_index - 1

    return None
