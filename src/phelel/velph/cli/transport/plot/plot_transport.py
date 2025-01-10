"""Implementation of velph-transport-plot-transport."""

from __future__ import annotations

import pathlib

import click
import h5py
import numpy as np


def plot_transport(
    f_h5py: h5py.File,
    plot_filename: str,
    dir_name: pathlib.Path,
    save_plot: bool = False,
):
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
    import matplotlib.pyplot as plt

    property_names = (
        "e_conductivity",
        "e_conductivity",
        "e_t_conductivity",
        "peltier",
        "seebeck",
    )

    f_elph = f_h5py["results/electron_phonon/electrons"]

    n_transport = len(
        [key for key in f_elph if "transport_" in key and key.split("_")[1].isdigit()]
    )
    assert n_transport == int(f_elph["transport_meta/ncalculators"][()])
    transports = [
        f_h5py[f"results/electron_phonon/electrons/transport_{n + 1}"]
        for n in range(n_transport)
    ]

    lines = []
    for i, transport in enumerate(transports):
        lines += _show(transport, i + 1)
    with open(dir_name / "transport.txt", "w") as w:
        print("\n".join(lines), file=w)
    click.echo(
        f'Summary of data structure was saved in "{dir_name / "transport.txt"}".'
    )

    n_props = len(property_names)
    _, axs = plt.subplots(
        len(transports),
        n_props,
        figsize=(4 * n_props, 4 * len(transports)),
        squeeze=False,
    )
    last_transport_idx = transports[-1]["id_idx"][:]
    for i, transport in enumerate(transports):
        _plot(
            axs[i, :],
            transport,
            property_names,
            i,
            transport["id_idx"][:],
            last_transport_idx,
            transport_dir_name=dir_name,
        )

    plt.tight_layout()
    if save_plot:
        plt.rcParams["pdf.fonttype"] = 42
        plt.savefig(plot_filename)
        click.echo(f'Transport plot was saved in "{plot_filename}".')
    else:
        plt.show()
    plt.close()


def _plot(
    axs: np.ndarray,
    transport: h5py.Group,
    property_names: tuple,
    index: int,
    transport_idx: np.ndarray,
    last_transport_idx: np.ndarray,
    transport_dir_name: pathlib.Path = pathlib.Path("transport"),
):
    """Plot transport properties.

    Second property with the property name "e_conductivity" is resistivity.

    """
    import matplotlib.ticker as ticker

    temps = transport["temps"][:]
    dat_filename = transport_dir_name / f"transport-{index + 1}.dat"
    properties = []

    label_property_names = []
    for prop_name in property_names:
        if prop_name == "e_conductivity" and prop_name in label_property_names:
            label_property_names.append("e_resistivity")
        else:
            label_property_names.append(prop_name)

    for key, label_prop_name in zip(property_names, label_property_names):
        if label_prop_name == "e_resistivity":
            properties.append([3 / np.trace(tensor) for tensor in transport[key][:]])
        else:
            properties.append([np.trace(tensor) / 3 for tensor in transport[key][:]])

    names = [name.decode("utf-8") for name in transport["id_name"][:]]
    labels = []
    click.echo(
        f"{index + 1}. "
        + " / ".join([f"{name} {int(i)}" for name, i in zip(names, transport_idx)])
    )
    for i, name in enumerate(names):
        if last_transport_idx[i] > 1:
            if name == "selfen_approx":
                labels.append(transport["scattering_approximation"][()].decode("utf-8"))

    with open(dat_filename, "w") as w:
        print("# temperature", *label_property_names, file=w)
        for temp, props in zip(temps, np.transpose(properties)):
            print(temp, *props, file=w)
        click.echo(f'Transport data {index + 1} was saved in "{dat_filename}".')

    for i, label_property in enumerate(label_property_names):
        if label_property == "e_conductivity":
            axs[i].semilogy(temps, properties[i], ".-")
        else:
            axs[i].plot(temps, properties[i], ".-")
        axs[i].tick_params(axis="both", which="both", direction="in")
        axs[i].tick_params(axis="y", direction="in")
        axs[i].set_xlabel("temperature (K)")
        if labels:
            axs[i].set_ylabel(f"[{index + 1}] {label_property} ({'-'.join(labels)})")
        else:
            axs[i].set_ylabel(f"[{index + 1}] {label_property}")
        axs[i].set_xlim(left=0, right=max(temps))
        axs[i].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        axs[i].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax = axs[i].secondary_yaxis("right")
        ax.tick_params(axis="y", which="both", direction="in")
        ax.set_yticklabels([])


def _show(transport: h5py.Group, index: int) -> list[str]:
    """Show transport properties.

    ['cell_volume', 'dfermi_tol', 'e_conductivity', 'e_t_conductivity', 'emax',
    'emin', 'energy', 'lab', 'mobility', 'mu', 'n', 'n0', 'ne', 'nedos',
    'nelect', 'nh', 'peltier', 'scattering_approximation', 'seebeck', 'static',
    'tau_average', 'temperature', 'transport_function']

    """
    lines = [f"- parameters:  # {index}"]
    for i, temp in enumerate(transport["temps"][:]):
        lines += [
            f"  - temperature: {temp}",
            "    scattering_approximation: {}".format(
                transport["scattering_approximation"][()].decode("utf-8")
            ),
        ]

        for key in (
            "cell_volume",
            "dfermi_tol",
            ("n0", "number_of_electrons_gausslegendre"),
            "nedos",
        ):
            if isinstance(key, tuple):
                lines.append(f"    {key[1]}: {transport[key[0]][()]}")
            else:
                lines.append(f"    {key}: {transport[key][()]}")
        lines.append(f"    static_approximation: {bool(transport['static'][()])}")

        lines.append("    data_scalar:")
        for key in (
            ("emax", "emax_for_transport_function"),
            ("emin", "emin_for_transport_function"),
            ("mu", "chemical_potential"),
            "n",
            ("ne", "ne_in_conduction_band"),
            ("nh", "nh_in_valence_band"),
            "tau_average",
        ):
            if isinstance(key, tuple):
                lines.append(f"      {key[1]}: {transport[key[0]][i]}")
            else:
                lines.append(f"      {key}: {transport[key][i]}")

        lines.append("    data_array_diagonal:")
        for key in (
            "e_conductivity",
            "e_t_conductivity",
            "mobility",
            "peltier",
            "seebeck",
        ):
            v = transport[key][:].ravel()
            lines.append(
                f"      {key}: [{v[0]:.3e}, {v[4]:.3e}, {v[8]:.3e}, "
                f"{v[5]:.3e}, {v[2]:.3e}, {v[1]:.3e}]"
            )

        lines.append("    data_array_shapes:")
        for key in (
            "energy",
            ("lab", "Onsager_coefficients"),
            "transport_function",
        ):
            if isinstance(key, tuple):
                lines.append(f"      {key[1]}: {list(transport[key[0]][i].shape)}")
            else:
                lines.append(f"      {key}: {list(transport[key][i].shape)}")

        return lines
