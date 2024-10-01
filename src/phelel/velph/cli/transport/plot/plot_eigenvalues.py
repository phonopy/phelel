"""Implementation of velph-transport-plot-eigenvalues."""

from __future__ import annotations

from typing import Optional

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
from phonopy.structure.brillouin_zone import get_qpoints_in_Brillouin_zone
from phonopy.structure.cells import get_reduced_bases
from phonopy.units import Kb
from scipy.spatial import Voronoi

from phelel.velph.cli.utils import get_symmetry_dataset
from phelel.velph.utils.vasp import read_crystal_structure_from_h5


def fermi_dirac_distribution(energy: np.ndarray, temperature: float) -> np.ndarray:
    """Calculate the Fermi-Dirac distribution.

    energy in eV measured from chemical potential
    temperature in K

    """
    de = energy / (Kb * temperature)
    de = np.where(de < 100, de, 100.0)  # To avoid overflow
    de = np.where(de > -100, de, -100.0)  # To avoid underflow
    return 1 / (1 + np.exp(de))


def plot_eigenvalues(
    f_h5py: h5py.File,
    tid: Optional[int] = None,
    temperature: Optional[float] = None,
    cutoff_occupancy: float = 1e-2,
    mu: Optional[float] = None,
    time_reversal: bool = True,
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
    cell = read_crystal_structure_from_h5(f_h5py, "results/positions")
    sym_dataset = get_symmetry_dataset(cell)
    rotations = [r.T for r in sym_dataset.rotations]

    if tid is not None:
        transport = f_h5py[f"results/electron_phonon/electrons/transport_{tid}"]
        _temperature = transport["temperature"][()]
        _mu = transport["mu"][()]
    else:
        if temperature is None:
            _temperature = 300
        else:
            _temperature = temperature
        if mu is None:
            _mu = f_h5py["results/electron_phonon/electrons/dos/efermi"][()]
        else:
            _mu = mu

    click.echo(f"mu: {_mu:.6f} eV")
    click.echo(f"temperature: {_temperature:.6f} K")

    if time_reversal:
        has_inversion = False
        for r in rotations:
            if (r == -np.eye(3, dtype=int)).all():
                has_inversion = True
                break
        if not has_inversion:
            rotations += [-r for r in rotations]

    dir_eigenvalues = "results/electron_phonon/electrons/eigenvalues"
    eigenvals = f_h5py[f"{dir_eigenvalues}/eigenvalues"][:] - _mu
    # weights = f_h5py[f"{dir_eigenvalues}/fermiweights"][:]
    kpoints = f_h5py[f"{dir_eigenvalues}/kpoint_coords"][:]
    ind = np.unravel_index(np.argsort(-eigenvals, axis=None), eigenvals.shape)
    weights = fermi_dirac_distribution(eigenvals, _temperature)

    all_kpoints = []
    all_weights = []
    all_eigenvals = []
    for i, (e, wt) in enumerate(zip(eigenvals[ind], weights[ind])):
        k = kpoints[ind[1][i]]
        if wt < cutoff_occupancy or wt > 1 - cutoff_occupancy:
            continue
        all_kpoints += list(rotations @ k.T)
        all_weights += [wt] * len(rotations)
        all_eigenvals += [e] * len(rotations)

    all_kpoints = np.array(all_kpoints)
    all_kpoints -= np.rint(all_kpoints)
    all_kpoints = get_qpoints_in_Brillouin_zone(
        np.linalg.inv(cell.cell), all_kpoints, only_unique=True
    )
    all_kpoints = all_kpoints @ np.linalg.inv(cell.cell).T
    all_weights = np.array(all_weights)
    all_eigenvals = np.array(all_eigenvals)

    with open("bz.dat", "w") as w:
        for i, (e, wt, rk) in enumerate(zip(all_eigenvals, all_weights, all_kpoints)):
            print(
                f"{i + 1} {e:.6f} {wt:.6f} [{rk[0]:.6f} {rk[1]:.6f} {rk[2]:.6f}]",
                file=w,
            )

    _plot_eigenvalues_in_BZ(
        all_kpoints,
        all_weights,
        np.linalg.inv(cell.cell),
        title=f"mu={_mu:.6f} eV, temperature={_temperature:.1f} K",
    )


def _plot_eigenvalues_in_BZ(
    data: np.ndarray,
    weights: np.ndarray,
    bz_lattice: np.ndarray,
    title: Optional[str] = None,
):
    """Plot kpoints in Brillouin zone."""
    ax = _get_ax_3D()
    point_sizes = (1 - np.abs(weights)) * 5
    scatter = ax.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        s=point_sizes,
        c=weights,
        cmap="viridis",
    )
    _plot_Brillouin_zone(bz_lattice, ax)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Occupancy")
    cbar.ax.yaxis.label.set_rotation(-90)
    cbar.ax.yaxis.labelpad = 20

    if title:
        ax.set_title(title)
    plt.show()


def _get_ax_3D() -> plt.Axes:
    """Get 3D axis."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])
    return ax


def _plot_Brillouin_zone(bz_lattice: np.ndarray, ax: plt.Axes):
    """Plot Brillouin zone.

    Parameters
    ----------
    bz_lattice : np.ndarray
        Reciprocal basis vectors in column vectors.

    """
    bz_red_lattice = get_reduced_bases(bz_lattice.T).T
    points = np.dot(np.array(list(np.ndindex(3, 3, 3))) - [1, 1, 1], bz_red_lattice.T)
    vor = Voronoi(points)

    distances_to_origin = np.linalg.norm(points, axis=1)
    origin_point_index = np.argmin(distances_to_origin)

    for ridge_points, ridge_cells in zip(vor.ridge_vertices, vor.ridge_points):
        if -1 in ridge_points:
            continue

        if origin_point_index in ridge_cells:
            # The first point is appended to the end to close the loop.
            _ridge_points = ridge_points[:] + [ridge_points[0]]
            ridge_vertices = np.array([vor.vertices[i] for i in _ridge_points])
            ax.plot(
                ridge_vertices[:, 0],
                ridge_vertices[:, 1],
                ridge_vertices[:, 2],
                ":",
                linewidth=0.5,
            )
