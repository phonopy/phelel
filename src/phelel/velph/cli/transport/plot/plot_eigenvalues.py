"""Implementation of velph-transport-plot-eigenvalues."""

from __future__ import annotations

from typing import Optional

import h5py
import numpy as np
from phonopy.units import Kb


def fermi_dirac_distribution(energy: np.ndarray, temperature: float) -> np.ndarray:
    """Calculate the Fermi-Dirac distribution.

    energy in eV
    temperature in K

    """
    return 1 / (1 + np.exp((energy) / (Kb * temperature)))


def plot_eigenvalues(
    f_h5py: h5py.File,
    temperature: float,
    cutoff_occupancy: float,
    mu: Optional[float],
):
    """Show eigenvalues, occupation, k-points and Fermi-Dirac distribution.

    Parameters
    ----------
    f_h5py : h5py.File
        vaspout.h5 file object.
    temperature : float
        Temperature for Fermi-Dirac distribution in K.
    cutoff_occupancy : float
        Cutoff for the occupancy to show eigenvalues in eV. Eigenvalus with
        occupances in interval [cutoff_occupancy, 1 - cutoff_occupancy] is
        shown.
    mu : float or None
        Chemical potential in eV. If None, the Fermi energy.

    """
    if mu is None:
        _mu = f_h5py["results/electron_phonon/electrons/dos/efermi"][()]
    else:
        _mu = mu
    dir_eigenvalues = "results/electron_phonon/electrons/eigenvalues"
    eigenvals = f_h5py[f"{dir_eigenvalues}/eigenvalues"][:] - _mu
    # weights = f_h5py[f"{dir_eigenvalues}/fermiweights"][:]
    kpoints = f_h5py[f"{dir_eigenvalues}/kpoint_coords"][:]
    ind = np.unravel_index(np.argsort(-eigenvals, axis=None), eigenvals.shape)
    weights = fermi_dirac_distribution(eigenvals, temperature)
    for i, (e, w) in enumerate(zip(eigenvals[ind], weights[ind])):
        k = kpoints[ind[1][i]]
        print(f"{i + 1} : {e:.6f} {w:.6f} [{k[0]:.6f} {k[1]:.6f} {k[2]:.6f}]")
        if w < cutoff_occupancy or w > 1 - cutoff_occupancy:
            break
