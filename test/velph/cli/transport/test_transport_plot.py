"""Tests of velph transport plot and velph ph_selfenergy plot."""

from __future__ import annotations

import inspect
import pathlib

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from phelel.velph.cli import plot_eigenvalues
from phelel.velph.cli.transport.plot.plot_selfenergy import plot_selfenergy
from phelel.velph.cli.velph_cmd_root import cmd_root


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


# Diagonal elements of the tensors of the transport calculations, as
# (e_conductivity, e_t_conductivity, peltier, seebeck), at temperature t of
# calculation i.
def _transport_values(i: int, t: float) -> tuple[float, float, float, float]:
    return (1e6 * (i + 1) * t, 2.0 * t, 3e-3 * t, 4e-6 * t)


def _write_transport_vaspout(filename: pathlib.Path, temps: list[float]) -> None:
    """Write vaspout.h5 with two transport calculations of transport.

    Only the datasets that velph transport plot transport reads are written.
    The tensors are diagonal with the values of _transport_values.

    """
    n_temps = len(temps)
    with h5py.File(filename, "w") as f:
        electrons = f.create_group("results/electron_phonon/electrons")
        electrons["transport_meta/ncalculators"] = 2
        for i in range(2):
            g = electrons.create_group(f"transport_{i + 1}")
            g["temps"] = np.array(temps)
            g["scattering_approximation"] = np.bytes_(["SERTA", "MRTA"][i])
            g["id_name"] = np.array([b"selfen_approx"])
            g["id_idx"] = np.array([i + 1])
            for key in ("cell_volume", "dfermi_tol", "n0", "nedos", "static"):
                g[key] = 1
            for key in ("emax", "emin", "mu", "n", "ne", "nh", "tau_average"):
                g[key] = np.zeros(n_temps)
            tensors = {
                key: np.array([np.eye(3) * _transport_values(i, t)[k] for t in temps])
                for k, key in enumerate(
                    ("e_conductivity", "e_t_conductivity", "peltier", "seebeck")
                )
            }
            for key, value in tensors.items():
                g[key] = value
            g["mobility"] = np.zeros((n_temps, 3, 3))
            for key in ("energy", "lab", "transport_function"):
                g[key] = np.zeros((n_temps, 4))


def test_velph_transport_plot_transport(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """The plot, the summary and the data of each calculation are written.

    The data are the averages of the diagonal elements of the tensors, and the
    resistivity is the inverse of the conductivity.

    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    temps = [100.0, 200.0, 300.0]
    directory = tmp_path / "calc"
    directory.mkdir()
    _write_transport_vaspout(directory / "vaspout.h5", temps)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cmd_root,
        [
            "transport",
            "plot",
            "transport",
            "--vaspout-filename",
            "calc/vaspout.h5",
            "--save",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (directory / "transport.pdf").exists()
    summary = (directory / "transport.txt").read_text()
    assert summary.count("- parameters:") == 2
    for i in range(2):
        lines = (directory / f"transport-{i + 1}.dat").read_text().splitlines()
        assert lines[0].split() == [
            "#",
            "temperature",
            "e_conductivity",
            "e_resistivity",
            "e_t_conductivity",
            "peltier",
            "seebeck",
        ]
        for line, t in zip(lines[1:], temps, strict=True):
            sigma, kappa, peltier, seebeck = _transport_values(i, t)
            np.testing.assert_allclose(
                [float(x) for x in line.split()],
                [t, sigma, 1 / sigma, kappa, peltier, seebeck],
            )


def test_velph_transport_plot_selfenergy(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """The plot and the summary of the self-energies are written."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    directory = tmp_path / "calc"
    directory.mkdir()
    _write_selfenergy_vaspout(directory / "vaspout.h5", 3)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cmd_root,
        [
            "transport",
            "plot",
            "selfenergy",
            "--vaspout-filename",
            "calc/vaspout.h5",
            "--save",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (directory / "selfenergy.pdf").exists()
    assert (directory / "selfenergy.txt").read_text().count("- parameters:") == 3
