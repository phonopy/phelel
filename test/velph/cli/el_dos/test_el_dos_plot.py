"""Tests of velph el_dos plot."""

from __future__ import annotations

import pathlib
import shutil

import click
import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from phelel.velph.cli.el_dos.plot import ElectronDos, draw_dos, read_electron_dos
from phelel.velph.cli.velph_cmd_root import cmd_root

TINISN_DOS = (
    pathlib.Path(__file__).parent.parent / "el_bands" / "plot" / "TiNiSn" / "dos"
)


def test_el_dos_plot_TiNiSn(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """The DOS is drawn against the energies in the window around the Fermi level.

    The DOS axis is from 0 to 1.1 times the largest DOS in the window, and the
    Fermi level is a vertical line.  With --save, el_dos/el_dos.pdf is written.

    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    (tmp_path / "el_dos").mkdir()
    shutil.copy(TINISN_DOS / "vaspout.h5", tmp_path / "el_dos" / "vaspout.h5")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "close", lambda *args: None)
    result = CliRunner().invoke(
        cmd_root, ["el_dos", "plot", "--window", "-5", "9", "--save"]
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == 'Electronic DOS plot was written to "el_dos/el_dos.pdf".\n'
    assert (tmp_path / "el_dos" / "el_dos.pdf").read_bytes()[:4] == b"%PDF"

    efermi = 8.575980267703933
    ax = plt.gcf().axes[0]
    assert ax.get_xlim() == pytest.approx((efermi - 5, efermi + 9))
    assert ax.get_ylim() == pytest.approx((0, 32.237937064036856))
    electron_dos = read_electron_dos(TINISN_DOS / "vaspout.h5")
    (line,) = ax.lines
    np.testing.assert_array_equal(line.get_xdata(), electron_dos.energies)
    np.testing.assert_array_equal(line.get_ydata(), electron_dos.dos[0])
    (fermi_level,) = ax.collections
    assert isinstance(fermi_level, LineCollection)
    (segment,) = fermi_level.get_segments()
    np.testing.assert_allclose(segment, [[efermi, 0], [efermi, 32.237937064036856]])
    plt.close("all")


def test_read_electron_dos(tmp_path: pathlib.Path):
    """The DOS of KPOINTS_OPT is read, and the total DOS of a non-collinear one.

    Without DOS in vaspout.h5, click.ClickException is raised.

    """
    filename = tmp_path / "vaspout.h5"
    with h5py.File(filename, "w") as w:
        for name, efermi in (("electron_dos", 1.0), ("electron_dos_kpoints_opt", 2.0)):
            group = w.create_group(f"results/{name}")
            group.create_dataset("dos", data=np.arange(12.0).reshape(4, 3) * efermi)
            group.create_dataset("energies", data=[1.0, 2.0, 3.0])
            group.create_dataset("efermi", data=efermi)
    electron_dos = read_electron_dos(filename)
    np.testing.assert_array_equal(electron_dos.dos, [[0.0, 2.0, 4.0]])
    assert electron_dos.efermi == 2.0

    with h5py.File(filename, "w") as w:
        w.create_group("results/electron_eigenvalues")
    with pytest.raises(click.ClickException) as e:
        read_electron_dos(filename)
    assert e.value.message == f'No electronic DOS is found in "{filename}".'


def test_draw_dos_spin_polarized():
    """The two spins are solid and dashed lines, and their sum is dotted."""
    matplotlib = pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    electron_dos = ElectronDos(
        energies=np.array([1.0, 2.0]),
        dos=np.array([[1.0, 2.0], [3.0, 5.0]]),
        efermi=1.5,
    )
    ax = Figure().subplots()
    draw_dos(ax, electron_dos, 10.0, energy_axis="x")
    assert [line.get_linestyle() for line in ax.lines] == ["-", "--", ":"]
    assert [matplotlib.colors.to_hex(line.get_color()) for line in ax.lines] == [
        "#000000"
    ] * 3
    np.testing.assert_array_equal(ax.lines[2].get_ydata(), [4.0, 7.0])
