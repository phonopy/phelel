"""Implementation of velph-transport-plot-selfenergy."""

from __future__ import annotations

import os
import pathlib

import click
import h5py


def plot_selfenergy(
    f_h5py: h5py.File,
    plot_filename: str | os.PathLike,
    dir_name: os.PathLike,
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
    f_elph = f_h5py["results/electron_phonon/electrons"]
    for key in f_elph:  # type: ignore
        if "self_energy_" in key:
            index = key.split("_")[-1]
            if index.isdigit():
                selfens[int(index)] = f_elph[key]  # type: ignore

    assert len(selfens) == int(f_elph["self_energy_meta/ncalculators"][()])  # type: ignore

    if len(selfens) == 1:
        fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    else:
        nrows = len(selfens) // 2
        fig, axs = plt.subplots(nrows, 2, figsize=(8, 4 * nrows), squeeze=True)

    lines = []
    for i in range(len(selfens)):
        selfen = selfens[i + 1]
        lines += _get_text_lines(selfen, i + 1)
        if len(selfens) == 1:
            _plot(axs, selfen)
        else:
            _plot(axs[i], selfen)  # type: ignore

    txt_filename = pathlib.Path(dir_name) / "selfenergy.txt"
    with open(txt_filename, "w") as w:
        print("\n".join(lines), file=w)
    click.echo(f'Summary of data structure was saved in "{txt_filename}".')

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


def _get_text_lines(selfen: h5py.Group, index: int) -> list[str]:
    """Show self-energy properties.

    ['band_start', 'band_stop', 'bks_idx', 'carrier_per_cell',
    'carrier_per_cell0', 'delta', 'efermi', 'energies', 'enwin', 'nbands',
    'nbands_sum', 'nw', 'scattering_approximation', 'select_energy_window',
    'selfen_dw', 'selfen_fan', 'static', 'tetrahedron']

    """
    lines = [
        f"- parameters:  # {index}",
        "    scattering_approximation: {}".format(
            selfen["scattering_approximation"][()].decode("utf-8")  # type: ignore
        ),
        f"    static_approximation: {bool(selfen['static'][()])}",  # type: ignore
        f"    use_tetrahedron_method: {bool(selfen['tetrahedron'][()])}",  # type: ignore
    ]
    if not selfen["tetrahedron"][()]:  # type: ignore
        lines.append(f"    smearing_width: {selfen['delta'][()]}")  # type: ignore
    lines += [
        f"    band_start_stop: [{selfen['band_start'][()]}, {selfen['band_stop'][()]}]",  # type: ignore
        f"    nbands: {selfen['nbands'][()]}",  # type: ignore
        f"    nbands_sum: {selfen['nbands_sum'][()]}",  # type: ignore
        f"    nw: {selfen['nw'][()]}",  # type: ignore
        "    temperatures:",
    ]
    for i, t in enumerate(selfen["temps"]):  # type: ignore
        lines.append(f"    - {t}  # {i + 1}")

    lines += [
        "  data_array_shapes:",
        f"    carrier_per_cell: {list(selfen['carrier_per_cell'].shape)}",  # type: ignore
        f"    Fan_self_energy: {list(selfen['selfen_fan'].shape)}",  # type: ignore
        f"    sampling_energy_points: {list(selfen['energies'].shape)}",  # type: ignore
        f"    Fermi_energies: {list(selfen['efermi'].shape)}",  # type: ignore
    ]
    return lines
