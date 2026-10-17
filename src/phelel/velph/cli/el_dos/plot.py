"""Implementation of velph-el_dos-plot.

The electronic DOS is read from vaspout.h5 of the DOS calculation, and drawn
by velph el_dos plot alone and by velph el_bands plot next to the band
structure.

"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import TYPE_CHECKING, Literal

import click
import h5py
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclasses.dataclass(frozen=True)
class ElectronDos:
    """Electronic DOS of vaspout.h5.

    energies has shape (energies,).  dos has shape (spin, energies), where
    spin is 1, or 2 for a spin-polarized calculation.  efermi is the Fermi
    level of the DOS calculation.

    """

    energies: NDArray
    dos: NDArray
    efermi: float


def read_electron_dos(vaspout_filename: pathlib.Path) -> ElectronDos:
    """Return the electronic DOS of vaspout.h5.

    The DOS of KPOINTS_OPT is read when vaspout.h5 has it.  For a
    non-collinear calculation, whose dos has four rows, only the first row,
    the total DOS, is kept.  click.ClickException is raised when vaspout.h5
    has no electronic DOS.

    """
    with h5py.File(vaspout_filename, "r") as f:
        for name in ("results/electron_dos_kpoints_opt", "results/electron_dos"):
            if name in f:
                group = f[name]
                break
        else:
            raise click.ClickException(
                f'No electronic DOS is found in "{vaspout_filename}".'
            )
        dos: NDArray = group["dos"][:]  # type: ignore
        energies: NDArray = group["energies"][:]  # type: ignore
        efermi = float(group["efermi"][()])  # type: ignore
    if len(dos) == 4:
        dos = dos[:1]
    return ElectronDos(energies=energies, dos=dos, efermi=efermi)


def get_dos_max(electron_dos: ElectronDos, emin: float, emax: float) -> float:
    """Return the upper limit of the DOS axis for the energies from emin to emax.

    It is 1.1 times the largest DOS in the range.  NaN is taken as 0, and
    values above 10000, which are not physical, are left out when the largest
    DOS is above 10000.

    """
    energies = electron_dos.energies
    dos = electron_dos.dos
    i_min = 0
    i_max = len(energies)
    for i, val in enumerate(energies):
        if val < emin:
            i_min = i
        if val > emax:
            i_max = i
            break

    # NaN elements are replaced by 0.
    if np.isfinite(dos).any():
        dos = np.nan_to_num(dos, nan=0.0, posinf=0.0, neginf=0.0)

    dos_max = dos[:, i_min : i_max + 1].max() * 1.1

    # Check unphysical values.
    if dos_max > 10000:
        dos = np.where(dos > 10000, 0, dos)
        dos_max = dos[:, i_min : i_max + 1].max() * 1.1

    return dos_max


def draw_dos(
    ax: Axes,
    electron_dos: ElectronDos,
    dos_max: float,
    energy_axis: Literal["x", "y"],
) -> None:
    """Draw the DOS and the Fermi level on ax.

    The DOS of each spin is drawn by a solid and a dashed line, and their sum
    by a dotted line for a spin-polarized calculation.  The Fermi level is a
    red line from 0 to dos_max.  energy_axis is the axis of the energies.

    """
    energies = electron_dos.energies
    curves = list(electron_dos.dos)
    styles = ["-k", "--k"][: len(curves)]
    if len(curves) == 2:
        curves.append(electron_dos.dos.sum(axis=0))
        styles.append(":k")
    for curve, style in zip(curves, styles, strict=True):
        if energy_axis == "x":
            ax.plot(energies, curve, style, linewidth=1)
        else:
            ax.plot(curve, energies, style, linewidth=1)
    if energy_axis == "x":
        ax.vlines(electron_dos.efermi, 0, dos_max, "r", linewidth=1)
    else:
        ax.hlines(electron_dos.efermi, 0, dos_max, "r", linewidth=1)


def plot_el_dos(
    window: tuple[float, float],
    vaspout_filename: pathlib.Path,
    save_plot: bool = False,
) -> None:
    """Plot electronic DOS.

    Parameters
    ----------
    window : tuple(float, float)
        Energy window to draw DOS in eV with respect to Fermi level.
    vaspout_filename : pathlib.Path
        Filename of vaspout.h5 of DOS.
    save_plot : bool, optional
        Save plot to el_dos.pdf in the directory of vaspout_filename.

    """
    import matplotlib.pyplot as plt

    electron_dos = read_electron_dos(vaspout_filename)
    emin = electron_dos.efermi + window[0]
    emax = electron_dos.efermi + window[1]
    dos_max = get_dos_max(electron_dos, emin, emax)

    _, ax = plt.subplots()
    draw_dos(ax, electron_dos, dos_max, energy_axis="x")
    ax.set_xlim(emin, emax)
    ax.set_ylim(0, dos_max)
    ax.set_xlabel("E[eV]", fontsize=14)
    ax.set_ylabel("DOS", fontsize=14)

    plt.tight_layout()
    if save_plot:
        plot_filename = vaspout_filename.parent / "el_dos.pdf"
        plt.rcParams["pdf.fonttype"] = 42
        plt.savefig(plot_filename)
        click.echo(f'Electronic DOS plot was written to "{plot_filename}".')
    else:
        plt.show()
    plt.close()
