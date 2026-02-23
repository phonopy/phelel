"""velph command line tool / velph-ph_selfenergy."""

import pathlib

import click
import h5py
import numpy as np

from phelel.velph.cli.cmd_root import cmd_root
from phelel.velph.cli.selfenergy.generate import write_selfenergy_input_files
from phelel.velph.cli.utils import check_fft
from phelel.velph.utils.vasp import read_freqs_and_ph_gammas_from_vaspout_h5


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
def cmd_dump(vaspout_filename: str, output_filename: str):
    """Dump ph_selfenergy data to HDF5 file.

    gammas: [ispin, ib, ikpt , nw, temp] -> [ispin, temp, ikpt, ib]
    freqs: [ikpt, ib] -> [ikpt, ib]

    """
    with h5py.File(vaspout_filename, "r") as f:
        freqs_calcs, gammas_calcs, temps_calcs, indices = (
            read_freqs_and_ph_gammas_from_vaspout_h5(f)
        )
        with h5py.File(output_filename, "w") as f_out:
            for i, freqs, gammas, temps in zip(
                indices, freqs_calcs, gammas_calcs, temps_calcs, strict=True
            ):
                _freqs = freqs / (2 * np.pi)  # convert to THz
                _gammas = np.transpose(gammas[:, :, :, 0, :], (0, 3, 2, 1)) / (
                    2 * np.pi
                )  # convert to THz
                f_out.create_dataset(f"frequency_{i}", data=_freqs)
                f_out.create_dataset(f"gamma_{i}", data=_gammas)
                f_out.create_dataset(f"temperature_{i}", data=temps)

            click.echo(f'Dumped ph_selfenergy data to "{output_filename}".')


from phelel.velph.cli.ph_selfenergy.plot.cmd_plot import cmd_plot  # noqa: E402, F401
