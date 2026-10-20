"""Implementation of velph-el_bands-plot."""

from __future__ import annotations

import pathlib

import click
import h5py
from numpy.typing import NDArray

from phelel.velph.cli.el_dos.plot import draw_dos, get_dos_max
from phelel.velph.jobs.project_directory import ProjectDirectory
from phelel.velph.utils.band_path import get_bands_data
from phelel.velph.utils.vaspout import get_reclat_from_vaspout, read_electron_dos


def plot_el_bandstructures(
    window: tuple[float, float],
    vaspout_filename_bands: pathlib.Path,
    vaspout_filename_dos: pathlib.Path,
    save_plot: bool = False,
    plot_filename: pathlib.Path | None = None,
):
    """Plot electronic band structure.

    Parameters
    ----------
    window : tuple(float, float)
        Energy window to draw band structure in eV with respect to Fermi level.
    vaspout_filename_bands : pathlib.Path
        Filename of vaspout.h5 of band structure.
    vaspout_filename_dos : pathlib.Path
        Filename of vaspout.h5 of DOS.
    save_plot : bool, optional
        Save plot to file.
    plot_filename : pathlib.Path, optional
        File name of band structure plot.  Default is el_bands/el_bands.pdf.

    """
    import matplotlib.pyplot as plt

    if plot_filename is None:
        plot_filename = ProjectDirectory().plot_pdf("el_bands")

    electron_dos = read_electron_dos(vaspout_filename_dos)
    efermi = electron_dos.efermi
    with h5py.File(vaspout_filename_bands) as f_h5py_bands:
        if "results" not in f_h5py_bands:
            raise click.ClickException(
                f'No electronic band structure is found in "{vaspout_filename_bands}".'
            )
        reclat = get_reclat_from_vaspout(f_h5py_bands)
        distances, eigvals, points, labels_at_points = _get_bands_data(
            reclat,
            f_h5py_bands["results/electron_eigenvalues_kpoints_opt"],  # type: ignore
            f_h5py_bands["input/kpoints_opt"],  # type: ignore
        )
    emin = window[0]
    emax = window[1]
    _, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})
    ax0, ax1 = axs

    lines = ["-k", "--k"]
    for i, eigvals_spin in enumerate(eigvals):
        ax0.plot(distances, eigvals_spin[:, :], lines[i], linewidth=1)
    ax0.hlines(efermi, distances[0], distances[-1], "r", linewidth=1)
    for x in points[1:-1]:
        ax0.vlines(x, efermi + emin, efermi + emax, "k", linewidth=1)
    ax0.set_xlim(distances[0], distances[-1])
    ax0.set_xticks(points)
    ax0.set_xticklabels(labels_at_points)
    ax0.set_ylim(efermi + emin, efermi + emax)
    ax0.set_ylabel("E[eV]", fontsize=14)
    ymin, ymax = ax0.get_ylim()

    dos_max = get_dos_max(electron_dos, ymin, ymax)
    draw_dos(ax1, electron_dos, dos_max, energy_axis="y")
    ax1.set_xlim(0, dos_max)
    ax1.set_ylim(ymin, ymax)
    ax1.yaxis.tick_right()

    plt.tight_layout()
    if save_plot:
        plt.rcParams["pdf.fonttype"] = 42
        plt.savefig(plot_filename)
        click.echo(f'Electronic band structure plot was written to "{plot_filename}".')
    else:
        plt.show()
    plt.close()


def _get_bands_data(
    reclat: NDArray, f_h5py_bands_results: h5py.Group, f_h5py_bands_input: h5py.Group
):
    eigvals: NDArray = f_h5py_bands_results["eigenvalues"][:]  # type: ignore

    # k-points in reduced coordinates
    kpoint_coords: NDArray = f_h5py_bands_results["kpoint_coords"][:]  # type: ignore
    # Special point labels
    labels = [
        label.decode("utf-8")
        for label in f_h5py_bands_input["labels_kpoints"][:]  # type: ignore
    ]
    nk_per_seg: int = f_h5py_bands_input["number_kpoints"][()]  # type: ignore

    distances, points, labels_at_points = get_bands_data(
        kpoint_coords, reclat, nk_per_seg, labels
    )

    return distances, eigvals, points, labels_at_points
