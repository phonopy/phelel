"""Implementation of velph-ph_selfenergy-plot-selfenergy."""

from __future__ import annotations

import os

import click
import h5py


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

    selfens = {}
    f_elph = f_h5py["results/electron_phonon/phonons"]
    for key in f_elph:  # type: ignore
        if "self_energy_" in key:
            index = key.split("_")[-1]
            if index.isdigit():
                selfens[int(index)] = f_elph[key]  # type: ignore

    assert len(selfens) == int(
        f_elph["results/electron_phonon/electrons/self_energy_meta/ncalculators"][()]  # type: ignore
    )

    if len(selfens) == 1:
        fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    else:
        nrows = len(selfens) // 2
        fig, axs = plt.subplots(nrows, 2, figsize=(8, 4 * nrows), squeeze=True)

    for i in range(len(selfens)):
        selfen = selfens[i + 1]
        if len(selfens) == 1:
            _plot(axs, selfen)
        else:
            _plot(axs[i], selfen)  # type: ignore

    plt.tight_layout()
    if save_plot:
        plt.rcParams["pdf.fonttype"] = 42
        plt.savefig(plot_filename)
        click.echo(f'Transport plot was saved in "{plot_filename}".')
    else:
        plt.show()
    plt.close()


def _plot(ax, selfen):
    for i_nw in range(selfen["nw"][()]):
        for i_temp, _ in enumerate(selfen["temps"]):
            ax.plot(
                selfen["energies"][:, i_nw],
                selfen["selfen_fan"][:, i_nw, i_temp, 1],
                ".",
            )
