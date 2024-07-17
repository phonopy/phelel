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
@click.help_option("-h", "--help")
def cmd_plot_selfenergy(vaspout_filename: str):
    """Plot self-energy in transports."""
    args = _get_f_h5py_and_plot_filename(pathlib.Path(vaspout_filename))
    if args[0] is not None:
        plot_selfenergy(*args)


@cmd_plot.command("transport")
@click.argument(
    "vaspout_filename",
    nargs=1,
    type=click.Path(),
    default="transport/vaspout.h5",
)
@click.help_option("-h", "--help")
def cmd_plot_transport(vaspout_filename: str):
    """Plot transport in transports."""
    args = _get_f_h5py_and_plot_filename(pathlib.Path(vaspout_filename))
    if args[0] is not None:
        plot_transport(*args)


def _get_f_h5py_and_plot_filename(
    vaspout_filename: pathlib.Path = pathlib.Path("transport/vaspout.h5"),
    plot_filename: Optional[pathlib.Path] = None,
) -> tuple[h5py._hl.files.File, pathlib.Path]:
    if not vaspout_filename.exists():
        click.echo(f'"{vaspout_filename}" not found. Please specific vaspout.h5 file.')
        return None, None

    if plot_filename is None:
        _plot_filename = vaspout_filename.parent / "transport.pdf"
    else:
        _plot_filename = plot_filename

    f_h5py = h5py.File(vaspout_filename)

    return f_h5py, _plot_filename
