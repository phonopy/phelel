"""velph command line tool / velph-ph_selfenergy."""

import pathlib

import click

from phelel.velph.cli.cmd_root import cmd_root
from phelel.velph.cli.ph_selfenergy.dump_phono3py import dump_phono3py
from phelel.velph.cli.selfenergy.generate import write_selfenergy_input_files
from phelel.velph.cli.utils import check_fft


@cmd_root.group("ph_selfenergy")
@click.help_option("-h", "--help")
def cmd_ph_selfenergy():
    """Choose ph_selfenergy options."""
    pass


@cmd_ph_selfenergy.command("generate")
@click.argument(
    "toml_filename",
    nargs=1,
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
    """Generate ph_selfenergy input files."""
    if not pathlib.Path("POTCAR").exists():
        click.echo('"POTCAR" not found in current directory.')

    write_selfenergy_input_files(pathlib.Path(toml_filename), dry_run, "ph_selfenergy")


@cmd_ph_selfenergy.command("check-fft")
@click.argument(
    "toml_filename",
    nargs=1,
    type=click.Path(),
    default="velph.toml",
)
@click.help_option("-h", "--help")
def cmd_check_fft(toml_filename: str):
    """Show [NGX, NGY, NGZ] in vasprun.xml."""
    check_fft(toml_filename, "ph_selfenergy")


@cmd_ph_selfenergy.command("dump_phono3py")
@click.argument(
    "vaspout_filename",
    nargs=1,
    type=click.Path(),
    default="ph_selfenergy/vaspout.h5",
)
@click.argument(
    "output_filename",
    nargs=1,
    type=click.Path(),
    default="ph_selfenergy/phono3py_elph.hdf5",
)
@click.help_option("-h", "--help")
def cmd_dump_phono3py(vaspout_filename: str, output_filename: str):
    """Dump ph_selfenergy data to HDF5 file."""
    dump_phono3py(vaspout_filename, output_filename)


from phelel.velph.cli.ph_selfenergy.plot.cmd_plot import cmd_plot  # noqa: E402, F401
