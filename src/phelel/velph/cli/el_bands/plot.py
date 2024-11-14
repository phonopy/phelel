"""Implementation of velph-el_bands-plot."""

from __future__ import annotations

import pathlib

import click
import h5py

from phelel.velph.cli.utils import (
    get_distances_along_BZ_path,
    get_reclat_from_vaspout,
    get_special_points,
)


def plot_el_bandstructures(
    window: tuple[float, float],
    vaspout_filename_bands: pathlib.Path,
    vaspout_filename_dos: pathlib.Path,
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
    import matplotlib.pyplot as plt

    f_h5py_bands = h5py.File(vaspout_filename_bands)
    f_h5py_dos = h5py.File(vaspout_filename_dos)
    efermi = f_h5py_dos["results"]["electron_dos_kpoints_opt"]["efermi"][()]
    emin = window[0]
    emax = window[1]
    _, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})
    ax0, ax1 = axs

    distances, eigvals, points, labels_at_points = _get_bands_data(f_h5py_bands)

    ax0.plot(distances, eigvals[0, :, :], "ok", markersize=1)
    ax0.hlines(efermi, distances[0], distances[-1], "r", linewidth=1)
    for x in points[1:-1]:
        ax0.vlines(x, efermi + emin, efermi + emax, "k", linewidth=1)
    ax0.set_xlim(distances[0], distances[-1])
    ax0.set_xticks(points)
    ax0.set_xticklabels(labels_at_points)
    ax0.set_ylim(efermi + emin, efermi + emax)
    ax0.set_ylabel("E[eV]", fontsize=14)
    ymin, ymax = ax0.get_ylim()

    dos, energies, xmax = _get_dos_data(f_h5py_dos, ymin, ymax)

    ax1.plot(dos, energies, "-k", linewidth=1)
    ax1.hlines(efermi, 0, xmax, "r", linewidth=1)
    ax1.set_xlim(0, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.yaxis.tick_right()

    plt.rcParams["pdf.fonttype"] = 42
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

    click.echo(f'Electronic band structure plot was saved in "{plot_filename}".')


def _get_bands_data(f_h5py: h5py.File):
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

    return distances, eigvals, points, labels_at_points


def _get_dos_data(f_h5py: h5py.File, ymin: float, ymax: float):
    dos = f_h5py["results"]["electron_dos_kpoints_opt"]["dos"][0, :]
    energies = f_h5py["results"]["electron_dos_kpoints_opt"]["energies"][:]
    i_min = 0
    i_max = len(energies)
    for i, val in enumerate(energies):
        if val < ymin:
            i_min = i
        if val > ymax:
            i_max = i
            break
    xmax = dos[i_min : i_max + 1].max() * 1.1
    return dos, energies, xmax
