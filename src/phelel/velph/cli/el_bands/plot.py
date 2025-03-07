"""Implementation of velph-el_bands-plot."""

from __future__ import annotations

import pathlib

import click
import h5py
import numpy as np

from phelel.velph.cli.utils import (
    get_distances_along_BZ_path,
    get_reclat_from_vaspout,
    get_special_points,
)


def plot_el_bandstructures(
    window: tuple[float, float],
    vaspout_filename_bands: pathlib.Path,
    vaspout_filename_dos: pathlib.Path,
    save_plot: bool = False,
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
    save_plot : bool, optional
        Save plot to file.
    plot_filename : pathlib.Path, optional
        File name of band structure plot.

    """
    import matplotlib.pyplot as plt

    f_h5py_bands = h5py.File(vaspout_filename_bands)
    f_h5py_dos = h5py.File(vaspout_filename_dos)
    if "results" not in f_h5py_bands:
        raise ValueError(
            f"No electronic band structure results found in {vaspout_filename_bands}."
        )
    if "results" not in f_h5py_dos:
        raise ValueError(f"No electronic DOS results found in {vaspout_filename_dos}.")
    if "electron_dos_kpoints_opt" in f_h5py_dos["results"]:
        f_h5py_dos_results = f_h5py_dos["results/electron_dos_kpoints_opt"]
    elif "electron_dos" in f_h5py_dos["results"]:
        f_h5py_dos_results = f_h5py_dos["results/electron_dos"]
    else:
        raise ValueError("No electron DOS data found in vaspout.h5.")

    efermi = f_h5py_dos_results["efermi"][()]
    emin = window[0]
    emax = window[1]
    _, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})
    ax0, ax1 = axs

    reclat = get_reclat_from_vaspout(f_h5py_bands)
    distances, eigvals, points, labels_at_points = _get_bands_data(
        reclat,
        f_h5py_bands["results/electron_eigenvalues_kpoints_opt"],
        f_h5py_bands["input/kpoints_opt"],
    )

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

    dos, energies, xmax = _get_dos_data(f_h5py_dos_results, ymin, ymax)

    lines = ["-k", "--k"]
    for i, dos_spin in enumerate(dos):
        ax1.plot(dos_spin, energies, lines[i], linewidth=1)
    if len(dos) == 2:
        ax1.plot(dos.sum(axis=0), energies, ":k", linewidth=1)
    ax1.hlines(efermi, 0, xmax, "r", linewidth=1)
    ax1.set_xlim(0, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.yaxis.tick_right()

    plt.tight_layout()
    if save_plot:
        plt.rcParams["pdf.fonttype"] = 42
        plt.savefig(plot_filename)
        click.echo(f'Electronic band structure plot was saved in "{plot_filename}".')
    else:
        plt.show()
    plt.close()


def _get_bands_data(
    reclat: np.ndarray, f_h5py_bands_results: h5py.Group, f_h5py_bands_input: h5py.Group
):
    eigvals = f_h5py_bands_results["eigenvalues"][:]

    # k-points in reduced coordinates
    kpoint_coords = f_h5py_bands_results["kpoint_coords"]
    # Special point labels
    labels = [
        label.decode("utf-8") for label in f_h5py_bands_input["labels_kpoints"][:]
    ]
    nk_per_seg = f_h5py_bands_input["number_kpoints"][()]
    nk_total = len(kpoint_coords)
    k_cart = kpoint_coords @ reclat
    n_segments = nk_total // nk_per_seg
    assert n_segments * nk_per_seg == nk_total
    distances = get_distances_along_BZ_path(nk_total, n_segments, nk_per_seg, k_cart)
    points, labels_at_points = get_special_points(
        labels, distances, n_segments, nk_per_seg, nk_total
    )

    return distances, eigvals, points, labels_at_points


def _get_dos_data(f_h5py_dos_results: h5py.Group, ymin: float, ymax: float):
    dos = f_h5py_dos_results["dos"][:]
    energies = f_h5py_dos_results["energies"][:]
    i_min = 0
    i_max = len(energies)
    for i, val in enumerate(energies):
        if val < ymin:
            i_min = i
        if val > ymax:
            i_max = i
            break

    # Take only total for non-collinear case.
    if len(dos) == 4:
        dos = dos[0][None, :]

    # NaN elements are replaced by 0.
    if np.isfinite(dos).any():
        dos = np.nan_to_num(dos, nan=0.0, posinf=0.0, neginf=0.0)

    xmax = dos[:, i_min : i_max + 1].max() * 1.1

    # Check unphysical values.
    if xmax > 10000:
        dos = np.where(dos > 10000, 0, dos)
        xmax = dos[:, i_min : i_max + 1].max() * 1.1

    return dos, energies, xmax
