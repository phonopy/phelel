"""velph command line tool / velph-transport-plot."""

from __future__ import annotations

import os
import pathlib

import click
import h5py

from phelel.velph.cli.transport.cmd_transport import cmd_transport
from phelel.velph.cli.transport.plot.plot_selfenergy import plot_selfenergy
from phelel.velph.cli.transport.plot.plot_transport import plot_transport
from phelel.velph.utils.plot_eigenvalues import (
    cmd_plot_eigenvalues,
    eigenvalue_plot_options,
)


@cmd_transport.group("plot")
@click.help_option("-h", "--help")
def cmd_plot():
    """Choose plot options."""
    pass


@cmd_plot.command("selfenergy")
@click.option(
    "--vaspout-filename",
    "vaspout_filename",
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
    _vaspout_filename, plot_filename, dir_name = _get_h5py_and_plot_filenames(
        "selfenergy", vaspout_filename=pathlib.Path(vaspout_filename)
    )
    if (
        _vaspout_filename is not None
        and plot_filename is not None
        and dir_name is not None
    ):
        with h5py.File(_vaspout_filename, "r") as f_h5py:
            plot_selfenergy(f_h5py, plot_filename, dir_name, save_plot=save_plot)


@cmd_plot.command("transport")
@click.option(
    "--vaspout-filename",
    "vaspout_filename",
    type=click.Path(),
    default="transport/vaspout.h5",
)
@click.option(
    "--save",
    "save_plot",
    is_flag=bool,
    default=False,
    help="Save plot to file.",
)
@click.help_option("-h", "--help")
def cmd_plot_transport(vaspout_filename: str, save_plot: bool):
    """Plot transport in transports."""
    _vaspout_filename, plot_filename, dir_name = _get_h5py_and_plot_filenames(
        "transport", vaspout_filename=pathlib.Path(vaspout_filename)
    )
    if (
        _vaspout_filename is not None
        and plot_filename is not None
        and dir_name is not None
    ):
        with h5py.File(_vaspout_filename, "r") as f_h5py:
            plot_transport(f_h5py, plot_filename, dir_name, save_plot=save_plot)


@cmd_plot.command("eigenvalues")
@click.option(
    "--vaspout-filename",
    "vaspout_filename",
    type=click.Path(),
    default="transport/vaspout.h5",
)
@click.option(
    "--tid",
    "tid",
    type=int,
    default=None,
    help=(
        "Index of self-energy calculation for collecting chemical potential and "
        "temperature. (tid: int, default=None)"
    ),
)
@click.option(
    "--calcid",
    "calcid",
    type=int,
    default=None,
    help=(
        "Index of self-energy calculation for collecting chemical potential and "
        "temperature. (calcid: int, default=None)"
    ),
)
@eigenvalue_plot_options
def cmd_plot_transport_eigenvalues(
    vaspout_filename: str,
    temperature: float,
    cutoff_occupancy: float,
    mu: float | None,
    tid: int | None,
    calcid: int | None,
):
    """Show eigenvalues in transports."""
    cmd_plot_eigenvalues(
        vaspout_filename,
        temperature,
        cutoff_occupancy,
        mu,
        calcid,
        tid,
    )


def _get_h5py_and_plot_filenames(
    property_name: str,
    vaspout_filename: str | os.PathLike | None = None,
    plot_filename: str | os.PathLike | None = None,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path] | tuple[None, None, None]:
    """Return h5py.File object and plot filename."""
    if vaspout_filename is None:
        _vaspout_filename = pathlib.Path(f"{property_name}/vaspout.h5")
    else:
        _vaspout_filename = pathlib.Path(vaspout_filename)

    if not _vaspout_filename.exists():
        click.echo(f'"{_vaspout_filename}" (default path) not found.')
        click.echo("Please specify vaspout.h5 file path.")
        return None, None, None

    dir_name = _vaspout_filename.parent

    if plot_filename is None:
        _plot_filename = dir_name / f"{property_name}.pdf"
    else:
        _plot_filename = pathlib.Path(plot_filename)

    return _vaspout_filename, _plot_filename, dir_name
