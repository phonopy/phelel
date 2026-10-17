"""Tests of phelel.velph.cli.plot_eigenvalues."""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest

from phelel.velph.cli import plot_eigenvalues


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
