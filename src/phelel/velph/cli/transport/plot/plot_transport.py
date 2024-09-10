"""Implementation of velph-el_bands-plot."""

from __future__ import annotations

import click
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_transport(f_h5py: h5py.File, plot_filename: str, save_plot: bool = False):
    """Plot transport properties.

    Number of "transport_*" is

    ((N(delta) * N(nbands_sum_array) * N(selfen_approx))
      * N(ncarrier_per_cell) * N(ncarrier_den) * N(mu)) * N(temps)

    N(temps) runs in the inner loop.

    sefeln_approx includes
    - scattering_approximation
      (CRTA, SERTA, ERTA_LAMBDA, ERTA_TAU, MRTA_LAMBDA, MRTA_TAU)
    - static_approximation (True or False)

    In the following codes, N(scattering_approximation) * N(temps) are only considered.

    """
    property_names = (
        "e_conductivity",
        # "mobility",
        "e_t_conductivity",
        "peltier",
        "seebeck",
    )

    temps = f_h5py["results"]["electron_phonon"]["electrons"]["self_energy_1"]["temps"][
        :
    ]

    transports = {}
    for key in f_h5py["results"]["electron_phonon"]["electrons"]:
        if "transport_" in key:
            transports[int(key.split("_")[1])] = f_h5py["results"]["electron_phonon"][
                "electrons"
            ][key]

    for i in range(len(transports)):
        transport = transports[i + 1]
        _show(transport, i + 1)

    transports_temps = {}

    n_blocks = len(transports) // len(temps)
    assert n_blocks * len(temps) == len(transports)

    # Nested loops mimic subroutine transport_elphon.
    for i_block in range(n_blocks):
        transports_temps[i_block] = []
        for i_temp, temp in enumerate(temps):
            idx = i_block * len(temps) + i_temp + 1
            transport = transports[idx]
            assert np.isclose(transport["temperature"], temp)
            transports_temps[i_block].append(transports[idx])

    n_props = len(property_names)
    _, axs = plt.subplots(
        n_blocks, n_props, figsize=(4 * n_props, 4 * n_blocks), squeeze=False
    )
    for i_block in range(n_blocks):
        _plot(axs[i_block, :], transports_temps[i_block], property_names)

    plt.tight_layout()
    if save_plot:
        plt.rcParams["pdf.fonttype"] = 42
        plt.savefig(plot_filename)
        click.echo(f'Transport plot was saved in "{plot_filename}".')
    else:
        plt.show()
    plt.close()


def _collect_data(transports_temps: dict, property_names: tuple) -> tuple[list, list]:
    properties = [[] for _ in property_names]
    temps = []
    for trpt in transports_temps:
        temps.append(trpt["temperature"][()])
        for i, property in enumerate(property_names):
            properties[i].append(np.trace(trpt[property][:]) / 3)

    return properties, temps


def _plot(axs: np.ndarray, transports_temps: dict, property_names: tuple):
    # Here only one key is considered, but there are many of those...
    key = transports_temps[0]["scattering_approximation"][()].decode("utf-8")

    properties, temps = _collect_data(transports_temps, property_names)
    with open(f"{key}.dat", "w") as w:
        print("# temperature", *property_names, file=w)
        for temp, props in zip(temps, np.transpose(properties)):
            print(temp, *props, file=w)

    for i, property in enumerate(property_names):
        if property == "e_conductivity":
            axs[i].semilogy(temps, properties[i], ".-")
        else:
            axs[i].plot(temps, properties[i], ".-")
        axs[i].tick_params(axis="both", which="both", direction="in")
        axs[i].tick_params(axis="y", direction="in")
        axs[i].set_xlabel("temperature (K)")
        axs[i].set_ylabel(f"{property} ({key})")
        axs[i].set_xlim(left=0, right=max(temps))
        axs[i].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        axs[i].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax = axs[i].secondary_yaxis("right")
        ax.tick_params(axis="y", which="both", direction="in")
        ax.set_yticklabels([])


def _show(transport: h5py.Group, index: int):
    """Show transport properties.

    ['cell_volume', 'dfermi_tol', 'e_conductivity', 'e_t_conductivity', 'emax',
    'emin', 'energy', 'lab', 'mobility', 'mu', 'n', 'n0', 'ne', 'nedos',
    'nelect', 'nh', 'peltier', 'scattering_approximation', 'seebeck', 'static',
    'tau_average', 'temperature', 'transport_function']

    """
    print(f"- parameters:  # {index}")
    print(
        "    scattering_approximation:",
        transport["scattering_approximation"][()].decode("utf-8"),
    )
    print(f"    temperature: {transport["temperature"][()]}")
    for key in (
        "cell_volume",
        "n",
        ("n0", "number_of_electrons_gausslegendre"),
        "nedos",
        "nelect",
        "cell_volume",
        "dfermi_tol",
    ):
        if isinstance(key, tuple):
            print(f"    {key[1]}: {transport[key[0]][()]}")
        else:
            print(f"    {key}: {transport[key][()]}")
    print(f"    static_approximation: {bool(transport["static"][()])}")

    print("  data_scalar:")
    for key in (
        ("emax", "emax_for_transport_function"),
        ("emin", "emin_for_transport_function"),
        ("mu", "chemical_potential"),
        ("ne", "ne_in_conduction_band"),
        ("nh", "nh_in_valence_band"),
        "tau_average",
    ):
        if isinstance(key, tuple):
            print(f"    {key[1]}: {transport[key[0]][()]}")
        else:
            print(f"    {key}: {transport[key][()]}")

    print("  data_array_diagonal:")
    for key in (
        "e_conductivity",
        "e_t_conductivity",
        "mobility",
        "peltier",
        "seebeck",
    ):
        v = transport[key][:].ravel()
        print(
            f"    {key}: [{v[0]:.3e}, {v[4]:.3e}, {v[8]:.3e}, "
            f"{v[5]:.3e}, {v[2]:.3e}, {v[1]:.3e}]"
        )

    print("  data_array_shapes:")
    for key in (
        "energy",
        ("lab", "Onsager_coefficients"),
        "transport_function",
    ):
        if isinstance(key, tuple):
            print(f"    {key[1]}: {list(transport[key[0]].shape)}")
        else:
            print(f"    {key}: {list(transport[key].shape)}")
