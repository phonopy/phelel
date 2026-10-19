"""velph command line tool / velph-transport-plot."""

from __future__ import annotations

import os
import pathlib

import click
import h5py

from phelel.velph.cli.plot_eigenvalues import (
    cmd_plot_eigenvalues,
    eigenvalue_plot_options,
)
from phelel.velph.cli.transport.cmd_transport import cmd_transport
from phelel.velph.cli.transport.plot.plot_selfenergy import plot_selfenergy
from phelel.velph.cli.transport.plot.plot_transport import plot_transport
from phelel.velph.jobs.project_directory import ProjectDirectory


@cmd_transport.group("plot")
@click.help_option("-h", "--help")
def cmd_plot():
    """Plot the results of the electronic transport calculation."""
    pass


@cmd_plot.command("selfenergy")
@click.option(
    "--vaspout-filename",
    "vaspout_filename",
    type=click.Path(),
    default=str(ProjectDirectory().vaspout("transport")),
)
@click.option(
    "--save",
    "save_plot",
    is_flag=True,
    default=False,
    help=("Save plot to file."),
)
@click.help_option("-h", "--help")
def cmd_plot_selfenergy(vaspout_filename: str, save_plot: bool):
    """Plot self-energy in transports."""
    _vaspout_filename, plot_filename, dir_name = _get_h5py_and_plot_filenames(
        "selfenergy", vaspout_filename=pathlib.Path(vaspout_filename)
    )
    with h5py.File(_vaspout_filename, "r") as f_h5py:
        plot_selfenergy(f_h5py, plot_filename, dir_name, save_plot=save_plot)


@cmd_plot.command("transport")
@click.option(
    "--vaspout-filename",
    "vaspout_filename",
    type=click.Path(),
    default=str(ProjectDirectory().vaspout("transport")),
)
@click.option(
    "--save",
    "save_plot",
    is_flag=True,
    default=False,
    help="Save plot to file.",
)
@click.help_option("-h", "--help")
def cmd_plot_transport(vaspout_filename: str, save_plot: bool):
    """Plot transport in transports."""
    _vaspout_filename, plot_filename, dir_name = _get_h5py_and_plot_filenames(
        "transport", vaspout_filename=pathlib.Path(vaspout_filename)
    )
    with h5py.File(_vaspout_filename, "r") as f_h5py:
        plot_transport(f_h5py, plot_filename, dir_name, save_plot=save_plot)


@cmd_plot.command("eigenvalues")
@click.option(
    "--vaspout-filename",
    "vaspout_filename",
    type=click.Path(),
    default=str(ProjectDirectory().vaspout("transport")),
)
@click.option(
    "--tid",
    "tid",
    type=int,
    default=None,
    help=(
        "Index of temperature of the self-energy calculation. The temperature "
        "and the chemical potential of this index are used in place of "
        "--temperature and --mu. (tid: int, default=None)"
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
        tid=tid,
        calcid=calcid,
        calc_type="transport",
    )


def _get_h5py_and_plot_filenames(
    property_name: str, vaspout_filename: str | os.PathLike
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Return vaspout.h5, plot filename and the directory of vaspout.h5.

    The plot is <property_name>.pdf in the directory of vaspout.h5.
    click.ClickException is raised when vaspout.h5 does not exist.

    """
    _vaspout_filename = pathlib.Path(vaspout_filename)
    if not _vaspout_filename.exists():
        raise click.ClickException(
            f'"{_vaspout_filename}" not found.  Give the path of vaspout.h5 with '
            "--vaspout-filename."
        )
    dir_name = _vaspout_filename.parent
    _plot_filename = dir_name / f"{property_name}.pdf"
    return _vaspout_filename, _plot_filename, dir_name
