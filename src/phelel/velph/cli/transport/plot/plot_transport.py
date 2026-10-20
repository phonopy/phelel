"""Implementation of velph-transport-plot-transport."""

from __future__ import annotations

import dataclasses
import os
import pathlib

import click
import h5py
import numpy as np

from phelel.velph.cli.utils import save_or_show_plot


@dataclasses.dataclass(frozen=True)
class _TransportProperty:
    """A property of transport_<n> of vaspout.h5 to plot.

    The value at each temperature is the average of the diagonal elements of
    the tensor of key, or the inverse of the average when inverse is True.
    log_scale draws the values with a logarithmic y axis.

    """

    label: str
    key: str
    inverse: bool = False
    log_scale: bool = False

    def values(self, transport: h5py.Group) -> list[float]:
        """Return the values at the temperatures of transport."""
        if self.inverse:
            return [3 / np.trace(tensor) for tensor in transport[self.key][:]]
        return [np.trace(tensor) / 3 for tensor in transport[self.key][:]]


# The properties plotted, in the order of the panels and of the columns of
# transport-<n>.dat.  The resistivity is the inverse of the conductivity.
_PROPERTIES = (
    _TransportProperty("e_conductivity", "e_conductivity", log_scale=True),
    _TransportProperty("e_resistivity", "e_conductivity", inverse=True),
    _TransportProperty("e_t_conductivity", "e_t_conductivity"),
    _TransportProperty("peltier", "peltier"),
    _TransportProperty("seebeck", "seebeck"),
)


def plot_transport(
    f_h5py: h5py.File,
    plot_filename: str | os.PathLike,
    dir_name: os.PathLike,
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
        f'Summary of the data structure was written to "{dir_name / "transport.txt"}".'
    )

    n_props = len(_PROPERTIES)
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
            i,
            transport["id_idx"][:],
            last_transport_idx,
            transport_dir_name=dir_name,
        )

    save_or_show_plot(plot_filename, save_plot, "Transport")


def _plot(
    axs: np.ndarray,
    transport: h5py.Group,
    index: int,
    transport_idx: np.ndarray,
    last_transport_idx: np.ndarray,
    transport_dir_name: pathlib.Path = pathlib.Path("transport"),
):
    """Plot the properties of _PROPERTIES and write them in transport-<n>.dat."""
    import matplotlib.ticker as ticker

    temps = transport["temps"][:]
    dat_filename = transport_dir_name / f"transport-{index + 1}.dat"
    properties = [prop.values(transport) for prop in _PROPERTIES]

    names = [name.decode("utf-8") for name in transport["id_name"][:]]
    labels = []
    click.echo(
        f"{index + 1}. "
        + " / ".join(
            [f"{name} {int(i)}" for name, i in zip(names, transport_idx, strict=True)]
        )
    )
    for i, name in enumerate(names):
        if last_transport_idx[i] > 1:
            if name == "selfen_approx":
                labels.append(transport["scattering_approximation"][()].decode("utf-8"))

    with open(dat_filename, "w") as w:
        print("# temperature", *[prop.label for prop in _PROPERTIES], file=w)
        for temp, props in zip(temps, np.transpose(properties), strict=True):
            print(temp, *props, file=w)
        click.echo(f'Transport data {index + 1} was written to "{dat_filename}".')

    for i, prop in enumerate(_PROPERTIES):
        if prop.log_scale:
            axs[i].semilogy(temps, properties[i], ".-")
        else:
            axs[i].plot(temps, properties[i], ".-")
        axs[i].tick_params(axis="both", which="both", direction="in")
        axs[i].tick_params(axis="y", direction="in")
        axs[i].set_xlabel("temperature (K)")
        if labels:
            axs[i].set_ylabel(f"[{index + 1}] {prop.label} ({'-'.join(labels)})")
        else:
            axs[i].set_ylabel(f"[{index + 1}] {prop.label}")
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
            v = transport[key][i].ravel()
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
