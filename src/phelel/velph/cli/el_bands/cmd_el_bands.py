"""velph command line tool / velph-el_bands."""

from __future__ import annotations

import click

from phelel.velph.cli.el_bands.plot import plot_el_bandstructures
from phelel.velph.cli.utils import write_input_set_files
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.jobs.project_directory import ProjectDirectory


@cmd_root.group("el_bands")
@click.help_option("-h", "--help")
def cmd_el_bands():
    """Electronic band structure.

    Write VASP input files of the electronic band structure and plot it.

    """
    pass


#
# velph el_bands generate
#
@cmd_el_bands.command("generate")
@click.option(
    "--toml-filename",
    "toml_filename",
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate input files to plot electronic band structure."""
    write_input_set_files(toml_filename, "el_bands")


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
@click.option(
    "--save",
    "save_plot",
    is_flag=True,
    default=False,
    help=("Save plot to file."),
)
@click.help_option("-h", "--help")
def cmd_plot(window: tuple[float, float], save_plot: bool):
    """Plot electronic band structure."""
    vaspout_filename_bands = ProjectDirectory().vaspout("el_bands")
    vaspout_filename_dos = ProjectDirectory().vaspout("el_dos")
    messages = []
    if not vaspout_filename_bands.exists():
        messages.append(f'"{vaspout_filename_bands}" not found.')
    if not vaspout_filename_dos.exists():
        messages.append(
            f'"{vaspout_filename_dos}" not found.  The Fermi level is taken from the '
            'DOS calculation.\nRun "velph el_dos generate" and VASP in "el_dos".'
        )
    if messages:
        raise click.ClickException("\n".join(messages))
    plot_el_bandstructures(
        window, vaspout_filename_bands, vaspout_filename_dos, save_plot=save_plot
    )
