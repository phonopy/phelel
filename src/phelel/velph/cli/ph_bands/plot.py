"""Implementation of velph-ph_bands-plot."""

import pathlib

import click
import h5py
import numpy as np

from phelel.velph.cli.utils import (
    get_distances_along_BZ_path,
    get_reclat_from_vaspout,
    get_special_points,
)


def plot_ph_bandstructures(
    vaspout_filename: pathlib.Path,
    use_ordinary_frequency: bool = False,
    save_plot: bool = False,
    plot_filename: pathlib.Path = pathlib.Path("ph_bands/ph_bands.pdf"),
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
        File name of band structure plot.

    """
    import matplotlib.pyplot as plt

    f = h5py.File(vaspout_filename)
    eigvals = f["results"]["phonons"]["eigenvalues"][:]  # phonon eigenvalues
    if use_ordinary_frequency:
        eigvals /= 2 * np.pi
    omega_max = 1.1 * eigvals.max()

    reclat = get_reclat_from_vaspout(f)
    labels = [
        label.decode("utf-8") for label in f["input"]["qpoints"]["labels_kpoints"][:]
    ]
    nk_per_seg = f["input"]["qpoints"]["number_kpoints"][()]
    kpoint_coords = f["results"]["phonons"]["kpoint_coords"]
    nk_total = len(kpoint_coords)
    k_cart = kpoint_coords @ reclat
    n_segments = nk_total // nk_per_seg
    assert n_segments * nk_per_seg == nk_total
    distances = get_distances_along_BZ_path(nk_total, n_segments, nk_per_seg, k_cart)
    points, labels_at_points = get_special_points(
        labels, distances, n_segments, nk_per_seg, nk_total
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
        click.echo(f'Phonon band structure plot was saved in "{plot_filename}".')
    else:
        plt.show()
    plt.close()
