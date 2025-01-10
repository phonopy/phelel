"""velph command line tool / velph-transport."""

from __future__ import annotations

import pathlib
from typing import Optional

import click
import h5py

from phelel.velph.cli.transport import cmd_transport
from phelel.velph.cli.transport.plot.plot_eigenvalues import plot_eigenvalues
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


@cmd_plot.command("eigenvalues")
@click.argument(
    "vaspout_filename",
    nargs=1,
    type=click.Path(),
    default="transport/vaspout.h5",
)
@click.option(
    "--cutoff-occupancy",
    nargs=1,
    type=float,
    default=1e-2,
    help=(
        "Cutoff for the occupancy to show eigenvalues in eV. Eigenvalus with "
        "occupances in interval [cutoff_occupancy, 1 - cutoff_occupancy] is "
        "shown. (cutoff_occupancy: float, default=1e-2)"
    ),
)
@click.option(
    "--mu",
    nargs=1,
    type=float,
    default=None,
    help=(
        "Chemical potential in eV unless --tid is specified. "
        "(mu: float, default=None, which means Fermi energy)"
    ),
)
@click.option(
    "--temperature",
    nargs=1,
    type=float,
    default=None,
    help=(
        "Temperature for Fermi-Dirac distribution in K unless --tid is specified. "
        "(temperature: float, default=None, which means 300 K)"
    ),
)
@click.option(
    "--tid",
    nargs=1,
    type=int,
    default=None,
    help=("Index of temperature. (tid: int, default=None)"),
)
@click.help_option("-h", "--help")
def cmd_plot_eigenvalues(
    vaspout_filename: str,
    temperature: float,
    cutoff_occupancy: float,
    mu: Optional[float],
    tid: Optional[int],
):
    """Show eigenvalues in transports."""
    args = _get_f_h5py_and_plot_filename(
        "transport", vaspout_filename=pathlib.Path(vaspout_filename)
    )
    if args[0] is None:
        return

    retvals = plot_eigenvalues(
        args[0],
        tid=tid,
        temperature=temperature,
        cutoff_occupancy=cutoff_occupancy,
        mu=mu,
    )

    if retvals is not None:
        with open("transport/bz.dat", "w") as w:
            for i, (e, wt, rk) in enumerate(zip(*retvals)):
                print(
                    f"{i + 1} {e:.6f} {wt:.6f} [{rk[0]:.6f} {rk[1]:.6f} {rk[2]:.6f}]",
                    file=w,
                )
        click.echo('"transport/bz.dat" file was created.')


def _get_f_h5py_and_plot_filename(
    property_name: str,
    vaspout_filename: pathlib.Path = pathlib.Path("transport/vaspout.h5"),
    plot_filename: Optional[pathlib.Path] = None,
) -> tuple[h5py.File, pathlib.Path]:
    if not vaspout_filename.exists():
        click.echo(f'"{vaspout_filename}" (default path) not found.')
        click.echo("Please specify vaspout.h5 file path.")
        return None, None

    dir_name = vaspout_filename.parent

    if plot_filename is None:
        _plot_filename = dir_name / f"{property_name}.pdf"
    else:
        _plot_filename = plot_filename

    f_h5py = h5py.File(vaspout_filename)

    return f_h5py, _plot_filename, dir_name
