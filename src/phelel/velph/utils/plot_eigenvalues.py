"""Implementation of velph plot-eigenvalues."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Literal

import click
import h5py
import numpy as np
from numpy.typing import NDArray
from phonopy.physical_units import get_physical_units
from phonopy.structure.brillouin_zone import get_qpoints_in_Brillouin_zone
from phonopy.structure.cells import get_reduced_bases
from scipy.spatial import Voronoi

from phelel.velph.utils.structure import get_symmetry_dataset
from phelel.velph.utils.vasp import read_crystal_structure_from_vaspout_h5

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


import functools


def eigenvalue_plot_options(func):
    """Return click options for eigenvalues plot."""

    @click.option(
        "--cutoff-occupancy",
        nargs=1,
        type=float,
        default=1e-2,
        help=(
            "Cutoff for the occupancy to show eigenvalues in eV. Eigenvalus with "
            "occupances in interval [cutoff_occupancy, 1 - cutoff_occupancy] is "
            "shown. (cutoff_occupancy: float, default=1e-2)"
        ),
    )
    @click.option(
        "--mu",
        nargs=1,
        type=float,
        default=None,
        help=(
            "Chemical potential in eV unless --tid is specified. "
            "(mu: float, default=None, which means Fermi energy)"
        ),
    )
    @click.option(
        "--temperature",
        nargs=1,
        type=float,
        default=None,
        help=(
            "Temperature for Fermi-Dirac distribution in K unless --tid is specified. "
            "(temperature: float, default=None, which means 300 K)"
        ),
    )
    @click.help_option("-h", "--help")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def cmd_plot_eigenvalues(
    vaspout_filename: str,
    temperature: float,
    cutoff_occupancy: float,
    mu: float | None,
    tid: int | None,
    calcid: int | None,
    calc_type: Literal["transport", "ph_selfenergy", "el_bands"] = "transport",
):
    """Show eigenvalues in transports.

    When tid is specified, temperature and mu are set according to the tid-th
    temperature and chemical potential in self-energy calculation. In this case,
    these are consistently determined values. Otherwise, temperature and mu are
    set independently.

    """
    _vaspout_filename = pathlib.Path(vaspout_filename)
    if not _vaspout_filename.exists():
        click.echo(
            f'"{_vaspout_filename}" not found. Please specify vaspout.h5 file path.',
            err=True,
        )
        return None

    with h5py.File(_vaspout_filename) as f_h5py:
        if calc_type in ("transport", "ph_selfenergy"):
            n_calculators: int = f_h5py[
                "results/electron_phonon/electrons/self_energy_meta/ncalculators"
            ][()]  # type: ignore

            click.echo(f"Possible calculator IDs: {np.arange(n_calculators) + 1}.")

        retvals = _plot_eigenvalues(
            f_h5py,
            tid=tid,
            calcid=calcid,
            temperature=temperature,
            cutoff_occupancy=cutoff_occupancy,
            mu=mu,
            calc_type=calc_type,
        )

    if retvals is not None:
        with open(f"{calc_type}/bz.dat", "w") as w:
            for i, (e, wt, rk) in enumerate(zip(*retvals, strict=True)):
                print(
                    f"{i + 1} {e:.6f} {wt:.6f} [{rk[0]:.6f} {rk[1]:.6f} {rk[2]:.6f}]",
                    file=w,
                )
        click.echo(f'"{calc_type}/bz.dat" file was created.')


def _plot_eigenvalues(
    f_h5py: h5py.File,
    tid: int | None = None,
    calcid: int | None = None,
    temperature: float | None = None,
    cutoff_occupancy: float = 1e-2,
    mu: float | None = None,
    time_reversal: bool = True,
    calc_type: Literal["transport", "ph_selfenergy", "el_bands"] = "transport",
) -> tuple[NDArray, NDArray, NDArray] | None:
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
    cell = read_crystal_structure_from_vaspout_h5(f_h5py, "results/positions")
    sym_dataset = get_symmetry_dataset(cell)
    rotations = [r.T for r in sym_dataset.rotations]

    if calcid is None:
        _calcid = 1
    else:
        _calcid = calcid

    if calc_type == "transport":
        params = f_h5py[f"results/electron_phonon/electrons/transport_{_calcid}"]  # type: ignore
    elif calc_type == "ph_selfenergy":
        params = f_h5py[f"results/electron_phonon/phonons/self_energy_{_calcid}"]  # type: ignore

    if calc_type in ("transport", "ph_selfenergy") and tid is not None:
        _temperature: float = params["temps"][tid - 1]  # type: ignore
        _mu = params["efermi"][tid - 1]  # type: ignore
    else:
        if calc_type in ("transport", "ph_selfenergy"):
            click.echo(
                f"Possible temperature IDs {np.arange(params['temps'].shape[0]) + 1}."  # type: ignore
            )
        if temperature is None:
            _temperature = 300
        else:
            _temperature = temperature
        if mu is None:
            if calc_type in ("transport", "ph_selfenergy"):
                _mu = f_h5py["results/electron_phonon/electrons/dos/efermi"][()]  # type: ignore
            elif calc_type == "el_bands":
                _mu = f_h5py["results/electron_dos/efermi"][()]  # type: ignore
            else:
                raise ValueError(f"Invalid calc_type: {calc_type}")
        else:
            _mu = mu
        click.echo("Note that mu and temperature are set independently.")

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

    if calc_type in ("transport", "ph_selfenergy"):
        dir_eigenvalues = "results/electron_phonon/electrons/eigenvalues"
    else:
        dir_eigenvalues = "results/electron_eigenvalues_kpoints_opt"
    eigenvals: NDArray = f_h5py[f"{dir_eigenvalues}/eigenvalues"][:] - _mu  # type: ignore
    # weights = f_h5py[f"{dir_eigenvalues}/fermiweights"][:]
    kpoints: NDArray = f_h5py[f"{dir_eigenvalues}/kpoint_coords"][:]  # type: ignore
    ind = np.unravel_index(np.argsort(-eigenvals, axis=None), eigenvals.shape)
    weights = _fermi_dirac_distribution(eigenvals, _temperature)

    all_kpoints = []
    all_weights = []
    all_eigenvals = []
    for i, (e, wt) in enumerate(zip(eigenvals[ind], weights[ind], strict=True)):
        k = kpoints[ind[1][i]]
        if wt < cutoff_occupancy or wt > 1 - cutoff_occupancy:
            continue
        all_kpoints += list(rotations @ k.T)
        all_weights += [wt] * len(rotations)
        all_eigenvals += [e] * len(rotations)

    if not all_kpoints:
        click.echo("No eigenvalues to plot.")
        return

    all_kpoints = np.array(all_kpoints)
    all_kpoints -= np.rint(all_kpoints)
    all_kpoints = get_qpoints_in_Brillouin_zone(
        np.linalg.inv(cell.cell), all_kpoints, only_unique=True
    )
    all_kpoints = all_kpoints @ np.linalg.inv(cell.cell).T
    all_weights = np.array(all_weights)
    all_eigenvals = np.array(all_eigenvals)

    _plot_eigenvalues_in_BZ(
        all_kpoints,
        all_weights,
        np.linalg.inv(cell.cell),
        title=f"mu={_mu:.6f} eV, temperature={_temperature:.1f} K",
    )

    return all_eigenvals, all_weights, all_kpoints


def _plot_eigenvalues_in_BZ(
    data: NDArray,
    weights: NDArray,
    bz_lattice: NDArray,
    title: str | None = None,
):
    """Plot kpoints in Brillouin zone."""
    import matplotlib.pyplot as plt

    ax = _get_ax_3D()
    point_sizes = (1 - np.abs(weights)) * 5
    scatter = ax.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],  # type: ignore
        s=point_sizes,  # type: ignore
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


def _get_ax_3D() -> Axes3D:
    """Get 3D axis.

    Returns
    -------
    Axes3D
        3D axis.

    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])  # type: ignore
    return ax  # type: ignore


def _plot_Brillouin_zone(bz_lattice: NDArray, ax: Axes3D):
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

    for ridge_points, ridge_cells in zip(
        vor.ridge_vertices, vor.ridge_points, strict=True
    ):
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


def _fermi_dirac_distribution(energy: NDArray, temperature: float) -> NDArray:
    """Calculate the Fermi-Dirac distribution.

    energy in eV measured from chemical potential
    temperature in K

    """
    de = energy / (get_physical_units().KB * temperature)
    de = np.where(de < 100, de, 100.0)  # To avoid overflow
    de = np.where(de > -100, de, -100.0)  # To avoid underflow
    return 1 / (1 + np.exp(de))
