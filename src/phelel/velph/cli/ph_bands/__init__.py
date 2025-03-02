"""velph command line tool / velph-ph_bands."""

import pathlib

import click

from phelel.velph.cli import cmd_root
from phelel.velph.cli.ph_bands.generate import (
    write_input_files as write_input_files_ph_bandstructures,
)
from phelel.velph.cli.ph_bands.plot import plot_ph_bandstructures


@cmd_root.group("ph_bands")
@click.help_option("-h", "--help")
def cmd_ph_bands():
    """Choose phonon band structure options."""
    pass


#
# velph ph_bands generate
#
@cmd_ph_bands.command("generate")
@click.argument(
    "toml_filename",
    nargs=1,
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate input files to plot phonon band structure."""
    if not pathlib.Path("POTCAR").exists():
        click.echo('"POTCAR" not found in current directory.')

    write_input_files_ph_bandstructures(pathlib.Path(toml_filename))


#
# velph ph_bands plot
#
@cmd_ph_bands.command("plot")
@click.option(
    "--ordinary-frequency/--angular-frequency",
    "use_ordinary_frequency",
    type=bool,
    default=False,
    help=("Use ordinary frequency instead of angular frequency."),
)
@click.option(
    "--save",
    "save_plot",
    is_flag=bool,
    default=False,
    help=("Save plot to file."),
)
@click.help_option("-h", "--help")
def cmd_plot(use_ordinary_frequency: bool, save_plot: bool):
    """Plot phonon band structure."""
    vaspout_filename = pathlib.Path("ph_bands/bands/vaspout.h5")
    if not vaspout_filename.exists():
        click.echo(f'"{vaspout_filename}" not found.')
    plot_ph_bandstructures(
        vaspout_filename, use_ordinary_frequency, save_plot=save_plot
    )


def plot(save_plot: bool = False):
    """Plot phonon band structure."""
    with click.Context(cmd_plot) as ctx:
        ctx.params = {"save_plot": save_plot}
        cmd_plot.invoke(ctx)
