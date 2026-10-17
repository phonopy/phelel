"""velph command line tool / velph-el_dos."""

from __future__ import annotations

import pathlib

import click

from phelel.velph.cli.el_dos.generate import write_input_files
from phelel.velph.cli.el_dos.plot import plot_el_dos
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.utils.plot_eigenvalues import (
    cmd_plot_eigenvalues,
    eigenvalue_plot_options,
)


@cmd_root.group("el_dos")
@click.help_option("-h", "--help")
def cmd_el_dos():
    """Choose electronic DOS options."""
    pass


#
# velph el_dos generate
#
@cmd_el_dos.command("generate")
@click.option(
    "--toml-filename",
    "toml_filename",
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate input files to calculate electronic DOS."""
    if not pathlib.Path("POTCAR").exists():
        click.echo('"POTCAR" not found in current directory.')

    write_input_files(pathlib.Path(toml_filename))


#
# velph el_dos plot
#
@cmd_el_dos.command("plot")
@click.option(
    "--window",
    required=True,
    type=(float, float),
    help="Energy window, emin and emax with respect to Fermi level.",
)
@click.option(
    "--vaspout-filename",
    "vaspout_filename",
    type=click.Path(),
    default="el_dos/vaspout.h5",
    show_default=True,
)
@click.option(
    "--save",
    "save_plot",
    is_flag=True,
    default=False,
    help="Save plot to el_dos.pdf in the directory of vaspout.h5.",
)
@click.help_option("-h", "--help")
def cmd_plot(window: tuple[float, float], vaspout_filename: str, save_plot: bool):
    """Plot electronic DOS."""
    path = pathlib.Path(vaspout_filename)
    if not path.exists():
        raise click.ClickException(
            f'"{path}" not found.  Give the path of vaspout.h5 with --vaspout-filename.'
        )
    plot_el_dos(window, path, save_plot=save_plot)


#
# velph el_dos plot_eigenvalues
#
@cmd_el_dos.command("plot_eigenvalues")
@click.option(
    "--vaspout-filename",
    "vaspout_filename",
    type=click.Path(),
    default="el_dos/vaspout.h5",
)
@eigenvalue_plot_options
def cmd_plot_dos_eigenvalues(
    vaspout_filename: str,
    temperature: float,
    cutoff_occupancy: float,
    mu: float | None,
):
    """Show eigenvalues of the DOS calculation."""
    cmd_plot_eigenvalues(
        vaspout_filename,
        temperature,
        cutoff_occupancy,
        mu,
        tid=None,
        calcid=None,
        calc_type="el_dos",
    )
