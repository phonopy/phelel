"""velph command line tool / velph-el_bands."""

from __future__ import annotations

import pathlib

import click

from velph.cli import cmd_root
from velph.cli.el_bands.generate import write_input_files
from velph.cli.el_bands.plot import plot_el_bandstructures


@cmd_root.group("el_bands")
@click.help_option("-h", "--help")
def cmd_el_bands():
    """Choose electronic band structure options."""
    pass


#
# velph el_bands generate
#
@cmd_el_bands.command("generate")
@click.argument(
    "toml_filename",
    nargs=1,
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate input files to plot electronic band structure."""
    if not pathlib.Path("POTCAR").exists():
        click.echo('"POTCAR" not found in current directory.')

    write_input_files(pathlib.Path(toml_filename))


#
# velph el_bands plot
#
@cmd_el_bands.command("plot")
@click.option(
    "--window",
    required=True,
    type=(float, float),
    help="Energy window, emin and emax with respect to Fermi level.",
)
@click.help_option("-h", "--help")
def cmd_plot(window: tuple[float, float]):
    """Plot electronic band structure."""
    vaspout_filename_bands = pathlib.Path("el_bands/bands/vaspout.h5")
    if vaspout_filename_bands.exists():
        vaspout_filename_dos = pathlib.Path("el_bands/dos/vaspout.h5")
        if vaspout_filename_bands.exists():
            click.echo(f'Found "{vaspout_filename_dos}". DOS will be plotted.')
            plot_el_bandstructures(
                window,
                vaspout_filename_bands,
                vaspout_filename_dos=vaspout_filename_dos,
            )
        else:
            plot_el_bandstructures(window, vaspout_filename_bands)
    else:
        click.echo(f'"{vaspout_filename_bands}" not found.')
