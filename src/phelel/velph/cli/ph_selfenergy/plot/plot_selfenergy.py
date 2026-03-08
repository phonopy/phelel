"""Implementation of velph-ph_selfenergy-plot-selfenergy."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import click
import h5py
from numpy.typing import NDArray

from phelel.velph.utils.vasp import read_freqs_and_ph_gammas_from_vaspout_h5

if TYPE_CHECKING:
    import matplotlib.axes as axes


def plot_selfenergy(
    f_h5py: h5py.File,
    plot_filename: str | os.PathLike,
    save_plot: bool = False,
):
    """Plot imaginary part of self-energies.

    Number of "self_energy_*" is

    (N(delta) * N(nbands_sum_array) * N(selfen_approx))
      * N(ncarrier_per_cell) * N(ncarrier_den) * N(mu)

    sefeln_approx includes
    - scattering_approximation (CRTA, ERTA, MRTA, MRTA2)
    - static_approximation (True or False)

    """
    import matplotlib.pyplot as plt

    freqs_calcs, gammas_calcs, temps_calcs, indices = (
        read_freqs_and_ph_gammas_from_vaspout_h5(f_h5py)
    )

    n_temp = len(temps_calcs[0])
    n_spin = gammas_calcs[0].shape[0]
    n_plots = len(gammas_calcs) * n_spin * n_temp
    if n_plots == 1:
        _, axs = plt.subplots(1, 1, figsize=(4, 4))
    else:
        nrows = n_plots
        _, axs = plt.subplots(nrows, 1, figsize=(4, 4 * nrows), squeeze=True)

    i_plot = 0
    for i, (gammas, freqs, temps) in enumerate(
        zip(gammas_calcs, freqs_calcs, temps_calcs, strict=True)
    ):
        for i_spin, gammas_at_spin in enumerate(gammas):
            for i_temp, temp in enumerate(temps):
                gammas_at_spin_temp = gammas_at_spin[:, :, :, i_temp]
                if len(gammas_calcs) * n_spin * n_temp == 1:
                    ax = axs
                else:
                    ax = axs[i_plot]  # type: ignore
                    i_plot += 1
                _plot(ax, gammas_at_spin_temp, freqs)
                ax.set_yscale("log")
                ax.set_title(
                    f"$\\tau^{{-1}}$ vs $\\omega$ at T={temp}K spn={i_spin} "
                    f"({indices[i]}) in 2$\\pi$THz"
                )

    plt.tight_layout()
    if save_plot:
        plt.rcParams["pdf.fonttype"] = 42
        plt.savefig(plot_filename)
        click.echo(f'Transport plot was saved in "{plot_filename}".')
    else:
        plt.show()
    plt.close()


def _plot(ax: axes.Axes, gammas: NDArray, freqs: NDArray):
    nw = gammas.shape[2]
    for i_nw in range(nw):
        for gammas_at_band, freqs_at_band in zip(gammas, freqs.T, strict=True):
            ax.plot(freqs_at_band, gammas_at_band[:, i_nw], ".")
