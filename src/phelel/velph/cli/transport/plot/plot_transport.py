"""Implementation of velph-el_bands-plot."""

from __future__ import annotations

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_transport(f_h5py: h5py._hl.files.File, plot_filename: str, show: bool = True):
    """Plot transport properties."""
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
    for i in range(len(transports)):
        transport = transports[i + 1]
        key = transport["scattering_approximation"][()].decode("utf-8")
        if key in transports_temps:
            transports_temps[key].append(transports[i + 1])
        else:
            transports_temps[key] = [
                transports[i + 1],
            ]

    nrows = len(transports_temps)
    fig, axs = plt.subplots(nrows, 5, figsize=(20, 4 * nrows))

    for i, key in enumerate(transports_temps):
        _plot(axs[i, :], transports_temps[key])

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.rcParams["pdf.fonttype"] = 42
        plt.savefig(plot_filename)
    plt.close()

    click.echo(f'Transport plot was saved in "{plot_filename}".')


def _plot(axs, tranports_temps):
    property_names = (
        "e_conductivity",
        "mobility",
        "e_t_conductivity",
        "peltier",
        "seebeck",
    )
    properties = [[] for _ in property_names]
    temps = []
    for trpt in tranports_temps:
        temps.append(trpt["temperature"][()])
        for i, property in enumerate(property_names):
            properties[i].append(np.trace(trpt[property][:]) / 3)

    for i, property in enumerate(property_names):
        axs[i].plot(temps, properties[i], ".-")
        axs[i].set_xlabel("temperature (K)")
        axs[i].set_ylabel(property)


def _show(transport: h5py._hl.group.Group, index: int):
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
    print("  data_array_shapes:")
    for key in (
        "e_conductivity",
        "e_t_conductivity",
        "energy",
        ("lab", "Onsager_coefficients"),
        "mobility",
        "peltier",
        "seebeck",
        "transport_function",
    ):
        if isinstance(key, tuple):
            print(f"    {key[1]}: {list(transport[key[0]].shape)}")
        else:
            print(f"    {key}: {list(transport[key].shape)}")
