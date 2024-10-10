"""Implementation of velph-transport-plot-selfenergy."""

from __future__ import annotations

import click
import h5py
import matplotlib.pyplot as plt


def plot_selfenergy(f_h5py: h5py.File, plot_filename: str, save_plot: bool = False):
    """Plot imaginary part of self-energies.

    Number of "self_energy_*" is

    (N(delta) * N(nbands_sum_array) * N(selfen_approx))
      * N(ncarrier_per_cell) * N(ncarrier_den) * N(mu)

    sefeln_approx includes
    - scattering_approximation (CRTA, ERTA, MRTA, MRTA2)
    - static_approximation (True or False)

    """
    selfens = {}
    for key in f_h5py["results"]["electron_phonon"]["electrons"]:
        if "self_energy_" in key:
            selfens[int(key.split("_")[2])] = f_h5py["results"]["electron_phonon"][
                "electrons"
            ][key]

    if len(selfens) == 1:
        fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    else:
        nrows = len(selfens) // 2
        fig, axs = plt.subplots(nrows, 2, figsize=(8, 4 * nrows), squeeze=True)

    for i in range(len(selfens)):
        selfen = selfens[i + 1]
        _show(selfen, i + 1)
        _plot(axs[i], selfen)

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


def _show(selfen: h5py.Group, index: int):
    """Show self-energy properties.

    ['band_start', 'band_stop', 'bks_idx', 'carrier_per_cell',
    'carrier_per_cell0', 'delta', 'efermi', 'energies', 'enwin', 'nbands',
    'nbands_sum', 'nw', 'scattering_approximation', 'select_energy_window',
    'selfen_dw', 'selfen_fan', 'static', 'tetrahedron']

    """
    print(f"- parameters:  # {index}")
    print(
        "    scattering_approximation:",
        selfen["scattering_approximation"][()].decode("utf-8"),
    )
    print(f"    static_approximation: {bool(selfen['static'][()])}")
    print(f"    use_tetrahedron_method: {bool(selfen["tetrahedron"][()])}")
    if not selfen["tetrahedron"][()]:
        print(f"    smearing_width: {selfen['delta'][()]}")
    print(
        f"    band_start_stop: [{selfen['band_start'][()]}, {selfen['band_stop'][()]}]"
    )
    print(f"    nbands: {selfen['nbands'][()]}")
    print(f"    nbands_sum: {selfen['nbands_sum'][()]}")
    print(f"    nw: {selfen['nw'][()]}")
    print("    temperatures:")
    for i, t in enumerate(selfen["temps"]):
        print(f"    - {t}  # {i + 1}")

    print("  data_array_shapes:")
    print(f"    carrier_per_cell: {list(selfen["carrier_per_cell"].shape)}")
    print(f"    Fan_self_energy: {list(selfen["selfen_fan"].shape)}")
    print(f"    sampling_energy_points: {list(selfen["energies"].shape)}")
    print(f"    Fermi_energies: {list(selfen["efermi"].shape)}")
