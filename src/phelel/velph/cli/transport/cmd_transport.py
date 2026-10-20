"""velph command line tool / velph-transport."""

import pathlib

import click

from phelel.velph.cli.selfenergy.generate import write_input_files
from phelel.velph.cli.velph_cmd_root import cmd_root


@cmd_root.group("transport")
@click.help_option("-h", "--help")
def cmd_transport():
    """Electronic transport.

    Write VASP input files of the electronic transport calculation and plot its
    results.

    """
    pass


@cmd_transport.command("generate")
@click.option(
    "--toml-filename",
    "toml_filename",
    type=click.Path(),
    default="velph.toml",
)
@click.option(
    "--dry-run/--no-dry-run",
    "-d",
    "dry_run",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str, dry_run: bool):
    """Generate transport input files."""
    write_input_files(pathlib.Path(toml_filename), "transport", dry_run)


from phelel.velph.cli.transport.plot.cmd_plot import cmd_plot  # noqa: E402, F401
