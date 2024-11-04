"""Tests velph-el_bands-plot."""

import pathlib

import h5py
import numpy as np
import pytest

from phelel.velph.cli.el_bands.plot import _get_bands_data, _get_dos_data

cwd = pathlib.Path(__file__).parent


def test_velph_el_bands_plot_TiNiSn():
    """Test of velph-el_bands-plot."""
    pytest.importorskip("seekpath")

    vaspout_filename_dos = cwd / "TiNiSn" / "dos" / "vaspout.h5"
    vaspout_filename_bands = cwd / "TiNiSn" / "bands" / "vaspout.h5"
    assert vaspout_filename_dos.exists()
    assert vaspout_filename_bands.exists()
    f_h5py_bands = h5py.File(vaspout_filename_bands)
    f_h5py_dos = h5py.File(vaspout_filename_dos)
    distances, eigvals, points, labels_at_points = _get_bands_data(f_h5py_bands)
    ymin, ymax = 3.575980267703933, 17.575980267703933
    dos, energies, xmax = _get_dos_data(f_h5py_dos, ymin, ymax)

    assert len(distances) == 306
    assert pytest.approx(distances[100], 1e-5) == 1.421887803385511
    assert eigvals.shape == (1, 306, 24)
    assert pytest.approx(eigvals[0, 100, 0], 1e-5) == -48.4663014156104
    np.testing.assert_allclose(
        points,
        [
            0.0,
            1.0560018348019153,
            1.4293548639688496,
            2.549413951469657,
            3.4639383668511003,
            4.21064442518497,
            4.7386453425859205,
        ],
    )
    assert labels_at_points == ["G", "X", "U|K", "G", "L", "W", "X"]

    assert len(dos) == 5001
    assert len(energies) == 5001
    assert pytest.approx(dos[4000], 1e-5) == 1.57397389
    assert pytest.approx(energies[4000], 1e-5) == 4.63468648
    assert pytest.approx(xmax, 1e-5) == 32.237937064036856
