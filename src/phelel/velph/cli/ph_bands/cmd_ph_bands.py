"""velph command line tool / velph-ph_bands."""

from __future__ import annotations

import pathlib

import click

from phelel.velph.cli.ph_bands.generate import (
    write_input_files as write_input_files_ph_bandstructures,
)
from phelel.velph.cli.ph_bands.plot import plot_ph_bandstructures
from phelel.velph.cli.utils import toml_filename_option
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.jobs.project_directory import ProjectDirectory


@cmd_root.group("ph_bands")
@click.help_option("-h", "--help")
def cmd_ph_bands():
    """Phonon band structure.

    Write VASP input files of the phonon band structure and plot it.

    """
    pass


#
# velph ph_bands generate
#
@cmd_ph_bands.command("generate")
@toml_filename_option
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate input files to plot phonon band structure."""
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
    is_flag=True,
    default=False,
    help=("Save plot to file."),
)
@click.help_option("-h", "--help")
def cmd_plot(use_ordinary_frequency: bool, save_plot: bool):
    """Plot phonon band structure."""
    vaspout_filename = ProjectDirectory().vaspout("ph_bands")
    if not vaspout_filename.exists():
        raise click.ClickException(f'"{vaspout_filename}" not found.')
    plot_ph_bandstructures(
        vaspout_filename, use_ordinary_frequency, save_plot=save_plot
    )
