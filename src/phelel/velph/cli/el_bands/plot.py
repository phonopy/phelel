"""Implementation of velph-el_bands-plot."""

from __future__ import annotations

import pathlib
from typing import Optional

import click
import h5py
import matplotlib.pyplot as plt

from phelel.velph.cli.utils import (
    get_distances_along_BZ_path,
    get_reclat_from_vaspout,
    get_special_points,
)


def plot_el_bandstructures(
    window: tuple[float, float],
    vaspout_filename_bands: pathlib.Path,
    vaspout_filename_dos: Optional[pathlib.Path] = None,
    plot_filename: pathlib.Path = pathlib.Path("el_bands/el_bands.pdf"),
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
    plot_filename : pathlib.Path, optional
        File name of band structure plot.

    """
    f_h5py_bands = h5py.File(vaspout_filename_bands)

    if vaspout_filename_dos:
        f_h5py_dos = h5py.File(vaspout_filename_dos)
        _, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})
        _plot_bands(axs[0], f_h5py_bands, window)
        _plot_dos(axs[1], f_h5py_dos, axs[0])
    else:
        _, ax = plt.subplots()
        _plot_bands(ax, f_h5py_bands, window)
    plt.rcParams["pdf.fonttype"] = 42
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

    click.echo(f'Electronic band structure plot was saved in "{plot_filename}".')


def _plot_bands(ax, f_h5py, window: tuple[float, float]):
    emin = window[0]
    emax = window[1]

    efermi = f_h5py["results"]["electron_dos"]["efermi"][()]
    eigvals = f_h5py["results"]["electron_eigenvalues_kpoints_opt"]["eigenvalues"]

    reclat = get_reclat_from_vaspout(f_h5py)
    # k-points in reduced coordinates
    kpoint_coords = f_h5py["results"]["electron_eigenvalues_kpoints_opt"][
        "kpoint_coords"
    ]
    # Special point labels
    labels = [
        label.decode("utf-8")
        for label in f_h5py["input"]["kpoints_opt"]["labels_kpoints"][:]
    ]
    nk_per_seg = f_h5py["input"]["kpoints_opt"]["number_kpoints"][()]
    nk_total = len(kpoint_coords)
    k_cart = kpoint_coords @ reclat
    n_segments = nk_total // nk_per_seg
    assert n_segments * nk_per_seg == nk_total
    distances = get_distances_along_BZ_path(nk_total, n_segments, nk_per_seg, k_cart)
    points, labels_at_points = get_special_points(
        labels, distances, n_segments, nk_per_seg, nk_total
    )

    ax.plot(distances, eigvals[0, :, :], ".k")
    ax.hlines(efermi, distances[0], distances[-1], "r")
    for x in points[1:-1]:
        ax.vlines(x, efermi + emin, efermi + emax, "k")
    ax.set_xlim(distances[0], distances[-1])
    ax.set_xticks(points)
    ax.set_xticklabels(labels_at_points)
    ax.set_ylim(efermi + emin, efermi + emax)
    ax.set_ylabel("E[eV]", fontsize=14)


def _plot_dos(ax, f_h5py, ax_bands):
    efermi = f_h5py["results"]["electron_dos_kpoints_opt"]["efermi"][()]
    dos = f_h5py["results"]["electron_dos_kpoints_opt"]["dos"][0, :]
    energies = f_h5py["results"]["electron_dos_kpoints_opt"]["energies"][:]
    ax.plot(dos, energies, "-k")
    ymin, ymax = ax_bands.get_ylim()
    i_min = 0
    i_max = 0
    for i, val in enumerate(energies):
        if val < ymin:
            i_min = i
        if val > ymax:
            i_max = i
            break
    xmax = dos[i_min:i_max].max() * 1.1
    ax.hlines(efermi, 0, xmax, "r")
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.tick_right()
