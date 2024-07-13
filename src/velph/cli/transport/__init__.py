"""velph command line tool / velph-transport."""

import pathlib

import click

from velph.cli import cmd_root
from velph.cli.transport.generate import write_input_files
from velph.cli.utils import check_fft


@cmd_root.group("transport")
@click.help_option("-h", "--help")
def cmd_transport():
    """Choose transport options."""
    pass


@cmd_transport.command("generate")
@click.argument(
    "toml_filename",
    nargs=1,
    type=click.Path(),
    default="velph.toml",
)
@click.option(
    "--hdf5-filename",
    "hdf5_filename",
    nargs=1,
    type=click.Path(),
    default="supercell/phelel_params.hdf5",
    show_default=True,
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
def cmd_generate(toml_filename: str, hdf5_filename: str, dry_run: bool):
    """Generate transport input files."""
    if not pathlib.Path("POTCAR").exists():
        click.echo('"POTCAR" not found in current directory.')

    write_input_files(
        pathlib.Path(toml_filename),
        pathlib.Path(hdf5_filename),
        dry_run,
    )


@cmd_transport.command("check-fft")
@click.argument(
    "toml_filename",
    nargs=1,
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_check_fft(toml_filename: str):
    """Show [NGX, NGY, NGZ] in vasprun.xml."""
    check_fft(toml_filename, "transport")
