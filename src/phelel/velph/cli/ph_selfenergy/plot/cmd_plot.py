"""velph command line tool / velph-ph_selfenergy-plot."""

from __future__ import annotations

import os
import pathlib

import click
import h5py

from phelel.velph.cli.ph_selfenergy.cmd_ph_selfenergy import cmd_ph_selfenergy
from phelel.velph.cli.ph_selfenergy.plot.plot_selfenergy import plot_selfenergy
from phelel.velph.utils.plot_eigenvalues import (
    cmd_plot_eigenvalues,
    eigenvalue_plot_options,
)


@cmd_ph_selfenergy.group("plot")
@click.help_option("-h", "--help")
def cmd_plot():
    """Choose plot options."""
    pass


@cmd_plot.command("selfenergy")
@click.option(
    "--vaspout-filename",
    "vaspout_filename",
    type=click.Path(),
    default="ph_selfenergy/vaspout.h5",
)
@click.option(
    "--save",
    "save_plot",
    is_flag=bool,
    default=False,
    help="Save plot to file.",
)
@click.help_option("-h", "--help")
def cmd_plot_selfenergy(vaspout_filename: str, save_plot: bool):
    """Plot self-energy in ph_selfenergy."""
    _vaspout_filename, plot_filename = _get_h5py_and_plot_filenames(
        "selfenergy", vaspout_filename=pathlib.Path(vaspout_filename)
    )
    with h5py.File(_vaspout_filename, "r") as f_h5py:
        if f_h5py is not None and plot_filename is not None:
            plot_selfenergy(f_h5py, plot_filename, save_plot=save_plot)


@cmd_plot.command("eigenvalues")
@click.option(
    "--vaspout-filename",
    "vaspout_filename",
    type=click.Path(),
    default="ph_selfenergy/vaspout.h5",
)
@click.option(
    "--tid",
    type=int,
    default=None,
    help=(
        "Index of temperature for collecting chemical potential and "
        "temperature. (tid: int, default=None)"
    ),
)
@click.option(
    "--calcid",
    type=int,
    default=None,
    help=(
        "Index of self-energy calculation for collecting chemical potential and "
        "temperature. (calcid: int, default=None)"
    ),
)
@eigenvalue_plot_options
def cmd_plot_ph_selfenergy_eigenvalues(
    vaspout_filename: str,
    temperature: float,
    cutoff_occupancy: float,
    mu: float | None,
    tid: int | None,
    calcid: int | None,
):
    """Show eigenvalues in ph_selfenergy."""
    cmd_plot_eigenvalues(
        vaspout_filename,
        temperature,
        cutoff_occupancy,
        mu,
        tid,
        calcid,
        calc_type="ph_selfenergy",
    )


def _get_h5py_and_plot_filenames(
    property_name: str,
    vaspout_filename: str | os.PathLike | None = None,
    plot_filename: str | os.PathLike | None = None,
) -> tuple[pathlib.Path, pathlib.Path] | tuple[None, None]:
    """Return h5py.File object and plot filename."""
    if vaspout_filename is None:
        _vaspout_filename = pathlib.Path(f"{property_name}/vaspout.h5")
    else:
        _vaspout_filename = pathlib.Path(vaspout_filename)

    if not _vaspout_filename.exists():
        click.echo(f'"{_vaspout_filename}" (default path) not found.')
        click.echo("Please specify vaspout.h5 file path.")
        return None, None

    dir_name = _vaspout_filename.parent

    if plot_filename is None:
        _plot_filename = dir_name / f"{property_name}.pdf"
    else:
        _plot_filename = pathlib.Path(plot_filename)

    return _vaspout_filename, _plot_filename
