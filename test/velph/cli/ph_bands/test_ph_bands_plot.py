"""Tests of velph ph_bands plot."""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from phelel.velph.cli.velph_cmd_root import cmd_root


def _write_ph_bands_vaspout(filename: pathlib.Path) -> None:
    """Write vaspout.h5 of ph_bands with a band path of two segments.

    Only the datasets that velph ph_bands plot reads are written.  Each segment
    has five q-points, and the frequencies increase along the path.

    """
    a = 5.69
    path = [
        ([0.0, 0.0, 0.0], [0.5, 0.0, 0.5]),
        ([0.5, 0.0, 0.5], [0.5, 0.25, 0.75]),
    ]
    qpoints = np.array(
        [
            np.array(start) + (np.array(end) - np.array(start)) * x
            for start, end in path
            for x in np.linspace(0, 1, 5)
        ]
    )
    with h5py.File(filename, "w") as f:
        f["input/poscar/lattice_vectors"] = [
            [0, a / 2, a / 2],
            [a / 2, 0, a / 2],
            [a / 2, a / 2, 0],
        ]
        f["input/poscar/scale"] = 1.0
        f["input/qpoints/labels_kpoints"] = np.array([b"G", b"X", b"X", b"W"])
        f["input/qpoints/number_kpoints"] = 5
        f["results/phonons/kpoint_coords"] = qpoints
        f["results/phonons/eigenvalues"] = np.outer(
            np.arange(1, len(qpoints) + 1), np.arange(1, 7)
        ).astype(float)


def test_velph_ph_bands_plot(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """The plot of ph_bands/vaspout.h5 is written in ph_bands/ph_bands.pdf."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    (tmp_path / "ph_bands").mkdir()
    _write_ph_bands_vaspout(tmp_path / "ph_bands" / "vaspout.h5")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cmd_root, ["ph_bands", "plot", "--save"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ph_bands" / "ph_bands.pdf").exists()
    assert result.output == (
        'Phonon band structure plot was written to "ph_bands/ph_bands.pdf".\n'
    )
