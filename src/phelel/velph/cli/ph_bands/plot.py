"""Implementation of velph-ph_bands-plot."""

from __future__ import annotations

import pathlib

import click
import h5py
import numpy as np
from numpy.typing import NDArray

from phelel.velph.jobs.project_directory import ProjectDirectory
from phelel.velph.utils.band_path import get_bands_data
from phelel.velph.utils.vaspout import get_reclat_from_vaspout


def plot_ph_bandstructures(
    vaspout_filename: pathlib.Path,
    use_ordinary_frequency: bool = False,
    save_plot: bool = False,
    plot_filename: pathlib.Path | None = None,
):
    """Plot phonon band structure.

    Parameters
    ----------
    vaspout_filename : pathlib.Path
        Filename of vaspout.h5.
    use_ordinary_frequency : bool, optional
        When True, phonon frequency unit becomes ordinary frequency of THz,
        otherwise angular frequency of THz. Default is False.
    save_plot : bool, optional
        Save plot to file.
    plot_filename : pathlib.Path, optional
        File name of band structure plot.  Default is ph_bands/ph_bands.pdf.

    """
    import matplotlib.pyplot as plt

    if plot_filename is None:
        plot_filename = ProjectDirectory().plot_pdf("ph_bands")

    with h5py.File(vaspout_filename) as f:
        eigvals: NDArray = f["results/phonons/eigenvalues"][:]  # type: ignore
        reclat = get_reclat_from_vaspout(f)
        labels = [
            label.decode("utf-8")
            for label in f["input/qpoints/labels_kpoints"][:]  # type: ignore
        ]
        nk_per_seg: int = f["input/qpoints/number_kpoints"][()]  # type: ignore
        kpoint_coords: NDArray = f["results/phonons/kpoint_coords"][:]  # type: ignore
    if use_ordinary_frequency:
        eigvals /= 2 * np.pi
    omega_max = 1.1 * eigvals.max()
    distances, points, labels_at_points = get_bands_data(
        kpoint_coords, reclat, nk_per_seg, labels
    )

    _, ax = plt.subplots()
    ax.plot(distances, eigvals, "-k", linewidth=1)
    for x in points[1:-1]:
        ax.vlines(x, 0, omega_max, "k", linewidth=1, linestyles=":")
    ax.set_xlim(distances[0], distances[-1])
    ax.set_xticks(points)
    ax.set_xticklabels(labels_at_points)
    ax.set_ylim(0, omega_max)
    if use_ordinary_frequency:
        ax.set_ylabel(r"$\nu$[THz]", fontsize=14)
    else:
        ax.set_ylabel(r"$\omega$[THz]", fontsize=14)

    # Finalize
    plt.tight_layout()
    if save_plot:
        plt.rcParams["pdf.fonttype"] = 42
        plt.savefig(plot_filename)
        click.echo(f'Phonon band structure plot was written to "{plot_filename}".')
    else:
        plt.show()
    plt.close()
