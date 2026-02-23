"""velph command line tool / velph-ph_selfenergy."""

import pathlib

import click
import h5py
import numpy as np
from numpy.typing import NDArray

from phelel.velph.cli.cmd_root import cmd_root
from phelel.velph.cli.selfenergy.generate import write_selfenergy_input_files
from phelel.velph.cli.utils import check_fft
from phelel.velph.utils.vasp import (
    convert_ir_kpoints_from_VASP_to_phono3py,
    read_freqs_and_ph_gammas_from_vaspout_h5,
)


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
    """Dump ph_selfenergy data to HDF5 file.

    gammas: [ispin, ib, ikpt , nw, temp] -> [ispin, temp, ikpt, ib]
    freqs: [ikpt, ib] -> [ikpt, ib]

    """
    with h5py.File(vaspout_filename, "r") as f:
        id_map, ir_kpoints, weights = _collect_data_from_vaspout(f)
        freqs_calcs, gammas_calcs, temps_calcs, indices = (
            read_freqs_and_ph_gammas_from_vaspout_h5(f)
        )
        with h5py.File(output_filename, "w") as f_out:
            for i, freqs, gammas, temps in zip(
                indices, freqs_calcs, gammas_calcs, temps_calcs, strict=True
            ):
                _freqs = freqs[id_map] / (2 * np.pi)  # convert to THz
                _gammas = np.transpose(gammas[:, :, id_map, 0, :], (0, 3, 2, 1)) / (
                    2 * np.pi
                )  # convert to THz
                f_out.create_dataset(f"frequency_{i}", data=_freqs)
                f_out.create_dataset(f"gamma_{i}", data=_gammas)
                f_out.create_dataset(f"temperature_{i}", data=temps)
                f_out.create_dataset("qpoint", data=ir_kpoints)
                f_out.create_dataset("weight", data=weights)

            click.echo(f'Dumped ph_selfenergy data to "{output_filename}".')


def _collect_data_from_vaspout(f_h5py: h5py.File) -> tuple[NDArray, NDArray, NDArray]:
    """Collect crystal structure information from vaspout.h5 file."""
    # [ikpt, 3]
    ir_kpoints: NDArray = f_h5py[
        "results/electron_phonon/electrons/eigenvalues/kpoint_coords"
    ][:]  # type: ignore
    ir_kpoints_weights: NDArray = f_h5py[
        "results/electron_phonon/electrons/eigenvalues/kpoints_symmetry_weight"
    ][:]  # type: ignore
    lat_scale = f_h5py["results/positions/scale"][()]  # type:ignore
    # lattice: row vectors
    lattice: NDArray = f_h5py["results/positions/lattice_vectors"][:] * lat_scale  # type: ignore
    positions: NDArray = f_h5py["results/positions/position_ions"][:]  # type: ignore
    number_ion_types: NDArray = f_h5py["results/positions/number_ion_types"][:]  # type: ignore
    numbers = []
    for i, nums in enumerate(number_ion_types):
        numbers += [i + 1] * nums
    numbers = np.array(numbers, dtype=int)
    if "basis_vectors" in f_h5py["input/kpoints_elph"]:  # type: ignore
        # When reading "basis_vectors" in python, the following 3x3 ndarray
        # is obtained:
        #   [a_m*_x, a_m*_y, a_m*_z]
        #   [b_m*_x, b_m*_y, b_m*_z]
        #   [c_m*_x, c_m*_y, c_m*_z]
        k_gen_vecs: NDArray = f_h5py["input/kpoints_elph/basis_vectors"][:]  # type:ignore
    else:
        k_gen_vecs: NDArray = np.diag(
            [
                1.0 / f_h5py[f"input/kpoints_elph/{key}"][()]  # type: ignore
                for key in ("nkpx", "nkpy", "nkpz")
            ]
        )
    assert f_h5py["input/kpoints/coordinate_space"][()] == b"R"  # type: ignore

    id_map, _, _, _, _ = convert_ir_kpoints_from_VASP_to_phono3py(
        lattice, positions, numbers, k_gen_vecs, ir_kpoints, ir_kpoints_weights
    )
    weights = ir_kpoints_weights[id_map] / np.linalg.det(k_gen_vecs)
    weights = np.rint(weights).astype(int)
    return id_map, ir_kpoints[id_map], weights


from phelel.velph.cli.ph_selfenergy.plot.cmd_plot import cmd_plot  # noqa: E402, F401
