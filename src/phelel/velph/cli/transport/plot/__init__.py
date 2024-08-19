"""velph command line tool / velph-transport."""

from __future__ import annotations

import pathlib
from typing import Optional

import click
import h5py

from phelel.velph.cli.transport import cmd_transport
from phelel.velph.cli.transport.plot.plot_selfenergy import plot_selfenergy
from phelel.velph.cli.transport.plot.plot_transport import plot_transport


@cmd_transport.group("plot")
@click.help_option("-h", "--help")
def cmd_plot():
    """Choose plot options."""
    pass


@cmd_plot.command("selfenergy")
@click.argument(
    "vaspout_filename",
    nargs=1,
    type=click.Path(),
    default="transport/vaspout.h5",
)
@click.option(
    "--save",
    "save_plot",
    is_flag=bool,
    default=False,
    help=("Save plot to file."),
)
@click.help_option("-h", "--help")
def cmd_plot_selfenergy(vaspout_filename: str, save_plot: bool):
    """Plot self-energy in transports."""
    args = _get_f_h5py_and_plot_filename(
        "selfenergy", vaspout_filename=pathlib.Path(vaspout_filename)
    )
    if args[0] is not None:
        plot_selfenergy(*args, save_plot=save_plot)


@cmd_plot.command("transport")
@click.argument(
    "vaspout_filename",
    nargs=1,
    type=click.Path(),
    default="transport/vaspout.h5",
)
@click.option(
    "--save",
    "save_plot",
    is_flag=bool,
    default=False,
    help=("Save plot to file."),
)
@click.help_option("-h", "--help")
def cmd_plot_transport(vaspout_filename: str, save_plot: bool):
    """Plot transport in transports."""
    args = _get_f_h5py_and_plot_filename(
        "transport", vaspout_filename=pathlib.Path(vaspout_filename)
    )
    if args[0] is not None:
        plot_transport(*args, save_plot=save_plot)


def _get_f_h5py_and_plot_filename(
    property_name: str,
    vaspout_filename: pathlib.Path = pathlib.Path("transport/vaspout.h5"),
    plot_filename: Optional[pathlib.Path] = None,
) -> tuple[h5py.File, pathlib.Path]:
    if not vaspout_filename.exists():
        click.echo(f'"{vaspout_filename}" (default path) not found.')
        click.echo("Please specify vaspout.h5 file path.")
        return None, None

    if plot_filename is None:
        _plot_filename = vaspout_filename.parent / f"{property_name}.pdf"
    else:
        _plot_filename = plot_filename

    f_h5py = h5py.File(vaspout_filename)

    return f_h5py, _plot_filename
