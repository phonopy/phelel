"""Tests velph-el_bands-plot."""

import pathlib
import shutil

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from phelel.velph.cli.el_bands.plot import _get_bands_data
from phelel.velph.cli.el_dos.plot import get_dos_max, read_electron_dos
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.utils.vasp import get_reclat_from_vaspout

cwd = pathlib.Path(__file__).parent


def test_velph_el_bands_plot_TiNiSn():
    """Test of velph-el_bands-plot."""
    pytest.importorskip("seekpath")

    vaspout_filename_dos = cwd / "TiNiSn" / "dos" / "vaspout.h5"
    vaspout_filename_bands = cwd / "TiNiSn" / "bands" / "vaspout.h5"
    assert vaspout_filename_dos.exists()
    assert vaspout_filename_bands.exists()
    f_h5py_bands = h5py.File(vaspout_filename_bands)
    reclat = get_reclat_from_vaspout(f_h5py_bands)
    distances, eigvals, points, labels_at_points = _get_bands_data(
        reclat,
        f_h5py_bands["results/electron_eigenvalues_kpoints_opt"],  # type: ignore
        f_h5py_bands["input/kpoints_opt"],  # type: ignore
    )
    ymin, ymax = 3.575980267703933, 17.575980267703933
    electron_dos = read_electron_dos(vaspout_filename_dos)
    dos = electron_dos.dos
    energies = electron_dos.energies
    xmax = get_dos_max(electron_dos, ymin, ymax)

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

    assert dos.shape == (1, 5001)
    assert len(energies) == 5001
    assert pytest.approx(dos[0, 4000], 1e-5) == 1.57397389
    assert pytest.approx(energies[4000], 1e-5) == 4.63468648
    assert pytest.approx(xmax, 1e-5) == 32.237937064036856


def test_velph_el_bands_plot_cli(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """The DOS is drawn on the right of the band structure against the energies.

    With --save, el_bands/el_bands.pdf is written.

    """
    pytest.importorskip("seekpath")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    for name, directory in (("bands", "el_bands"), ("dos", "el_dos")):
        (tmp_path / directory).mkdir()
        shutil.copy(cwd / "TiNiSn" / name / "vaspout.h5", tmp_path / directory)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "close", lambda *args: None)
    result = CliRunner().invoke(
        cmd_root, ["el_bands", "plot", "--window", "-5", "9", "--save"]
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == (
        'Electronic band structure plot was written to "el_bands/el_bands.pdf".\n'
    )
    assert (tmp_path / "el_bands" / "el_bands.pdf").read_bytes()[:4] == b"%PDF"

    efermi = 8.575980267703933
    ax = plt.gcf().axes[1]
    assert ax.get_xlim() == pytest.approx((0, 32.237937064036856))
    assert ax.get_ylim() == pytest.approx((efermi - 5, efermi + 9))
    electron_dos = read_electron_dos(cwd / "TiNiSn" / "dos" / "vaspout.h5")
    (line,) = ax.lines
    np.testing.assert_array_equal(line.get_xdata(), electron_dos.dos[0])
    np.testing.assert_array_equal(line.get_ydata(), electron_dos.energies)
    (fermi_level,) = ax.collections
    assert isinstance(fermi_level, LineCollection)
    (segment,) = fermi_level.get_segments()
    np.testing.assert_allclose(segment, [[0, efermi], [32.237937064036856, efermi]])
    plt.close("all")
