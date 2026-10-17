"""Tests of velph transport plot and velph ph_selfenergy plot."""

from __future__ import annotations

import inspect
import pathlib

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from phelel.velph.cli.transport.plot.plot_selfenergy import plot_selfenergy
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.utils import plot_eigenvalues


@pytest.mark.parametrize("group", ["transport", "ph_selfenergy"])
def test_plot_eigenvalues_tid_and_calcid(
    group: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """--tid and --calcid are given to cmd_plot_eigenvalues as tid and calcid."""
    calls = []
    signature = inspect.signature(plot_eigenvalues.cmd_plot_eigenvalues)

    def _cmd_plot_eigenvalues(*args, **kwargs):
        calls.append(signature.bind(*args, **kwargs).arguments)

    monkeypatch.setattr(
        f"phelel.velph.cli.{group}.plot.cmd_plot.cmd_plot_eigenvalues",
        _cmd_plot_eigenvalues,
    )
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cmd_root, [group, "plot", "eigenvalues", "--tid", "2", "--calcid", "3"]
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert (calls[0]["tid"], calls[0]["calcid"]) == (2, 3)


def _write_selfenergy_vaspout(filename: pathlib.Path, n_calcs: int) -> None:
    """Write vaspout.h5 with n_calcs self-energy calculations of transport.

    Only the datasets that velph transport plot selfenergy reads are written.

    """
    n_energies, n_w, n_temps = 5, 1, 2
    with h5py.File(filename, "w") as f:
        electrons = f.create_group("results/electron_phonon/electrons")
        electrons["self_energy_meta/ncalculators"] = n_calcs
        for i in range(n_calcs):
            g = electrons.create_group(f"self_energy_{i + 1}")
            g["scattering_approximation"] = np.bytes_("SERTA")
            g["static"] = 0
            g["tetrahedron"] = 0
            g["delta"] = 0.01
            g["band_start"] = 1
            g["band_stop"] = 4
            g["nbands"] = 8
            g["nbands_sum"] = 8
            g["nw"] = n_w
            g["temps"] = np.linspace(100, 300, n_temps)
            g["carrier_per_cell"] = np.zeros(n_temps)
            g["efermi"] = np.zeros(n_temps)
            g["energies"] = np.zeros((n_energies, n_w))
            g["selfen_fan"] = np.zeros((n_energies, n_w, n_temps, 2))


@pytest.mark.parametrize("n_calcs", [1, 2, 3, 4])
def test_plot_selfenergy_number_of_calculations(n_calcs: int, tmp_path: pathlib.Path):
    """The self-energies of any number of calculations are plotted."""
    pytest.importorskip("matplotlib")
    vaspout_filename = tmp_path / "vaspout.h5"
    _write_selfenergy_vaspout(vaspout_filename, n_calcs)
    plot_filename = tmp_path / "selfenergy.pdf"
    with h5py.File(vaspout_filename, "r") as f:
        plot_selfenergy(f, plot_filename, tmp_path, save_plot=True)
    assert plot_filename.exists()
    text = (tmp_path / "selfenergy.txt").read_text()
    assert text.count("- parameters:") == n_calcs
