"""Tests of phelel.velph.cli.plot_eigenvalues."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import h5py
import numpy as np
import pytest
from click.testing import CliRunner
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms

from phelel.velph.cli import plot_eigenvalues
from phelel.velph.cli.velph_cmd_root import cmd_root


def test_bz_dat_in_directory_of_vaspout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """bz.dat is written in the directory of vaspout.h5.

    The directory differs from the default "el_dos", which does not exist in
    the current directory.

    """

    def _plot_eigenvalues(f_h5py, **kwargs):
        return np.array([0.1]), np.array([0.5]), np.array([[0.0, 0.25, 0.5]])

    monkeypatch.setattr(plot_eigenvalues, "_plot_eigenvalues", _plot_eigenvalues)
    directory = tmp_path / "other"
    directory.mkdir()
    vaspout_filename = directory / "vaspout.h5"
    h5py.File(vaspout_filename, "w").close()
    monkeypatch.chdir(tmp_path)
    plot_eigenvalues.cmd_plot_eigenvalues(
        str(vaspout_filename),
        300.0,
        1e-2,
        None,
        tid=None,
        calcid=None,
        calc_type="el_dos",
    )
    assert (directory / "bz.dat").read_text() == (
        "1 0.100000 0.500000 [0.000000 0.250000 0.500000]\n"
    )
    assert not (tmp_path / "el_dos").exists()


# Chemical potential and temperature of the electron-phonon calculation, and a
# k-point at a general position of the Brillouin zone.
_MU = 5.0
_TEMPERATURE = 300.0
_KPOINT = [0.1, 0.2, 0.3]


def _write_eigenvalue_vaspout(
    filename: pathlib.Path,
    calc_type: str,
    cell: PhonopyAtoms,
    eigenvalues: list[float],
    write_positions: Callable[[h5py.Group, PhonopyAtoms], None],
) -> None:
    """Write vaspout.h5 with eigenvalues at _KPOINT for an eigenvalue plot.

    Only the datasets that the eigenvalue plot of calc_type reads are written.
    The chemical potential and the temperature are _MU and _TEMPERATURE, the
    first ones of the calculation for transport and ph_selfenergy.

    """
    with h5py.File(filename, "w") as f:
        write_positions(f.create_group("results/positions"), cell)
        values = np.array([[eigenvalues]])
        if calc_type == "el_dos":
            f["results/electron_dos/efermi"] = _MU
            group = f.create_group("results/electron_eigenvalues_kpoints_opt")
        else:
            electrons = f.create_group("results/electron_phonon/electrons")
            electrons["self_energy_meta/ncalculators"] = 1
            electrons["dos/efermi"] = _MU
            name = (
                "electrons/transport_1"
                if calc_type == "transport"
                else "phonons/self_energy_1"
            )
            params = f.create_group(f"results/electron_phonon/{name}")
            params["temps"] = np.array([_TEMPERATURE, 500.0])
            params["efermi"] = np.array([_MU, _MU + 1])
            group = electrons.create_group("eigenvalues")
        group["eigenvalues"] = values
        group["kpoint_coords"] = np.array([_KPOINT])


_EIGENVALUE_COMMANDS = {
    "el_dos": ["el_dos", "plot_eigenvalues"],
    "transport": ["transport", "plot", "eigenvalues", "--tid", "1", "--calcid", "1"],
    "ph_selfenergy": [
        "ph_selfenergy",
        "plot",
        "eigenvalues",
        "--tid",
        "1",
        "--calcid",
        "1",
    ],
}


@pytest.mark.parametrize("calc_type", ["el_dos", "transport", "ph_selfenergy"])
def test_plot_eigenvalues_bz_dat(
    calc_type: str,
    nacl_prim_cell: PhonopyAtoms,
    write_vaspout_positions: Callable[[h5py.Group, PhonopyAtoms], None],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """bz.dat has the eigenvalue at the k-points of the star of the k-point.

    The point group of NaCl has 48 rotations and inversion, so the k-point at a
    general position has 48 points in its star.  The eigenvalue is measured
    from the chemical potential, and its occupancy is that of the Fermi-Dirac
    distribution.

    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    eigenvalue = _MU + 0.01
    vaspout = tmp_path / "calc" / "vaspout.h5"
    vaspout.parent.mkdir()
    _write_eigenvalue_vaspout(
        vaspout, calc_type, nacl_prim_cell, [eigenvalue], write_vaspout_positions
    )
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cmd_root, _EIGENVALUE_COMMANDS[calc_type] + ["--vaspout-filename", str(vaspout)]
    )
    assert result.exit_code == 0, result.output
    lines = (vaspout.parent / "bz.dat").read_text().splitlines()
    assert len(lines) == 48
    occupancy = 1 / (
        1 + np.exp((eigenvalue - _MU) / (get_physical_units().KB * _TEMPERATURE))
    )
    for line in lines:
        values = line.split()
        assert float(values[1]) == pytest.approx(eigenvalue - _MU, abs=1e-6)
        assert float(values[2]) == pytest.approx(occupancy, abs=1e-6)
