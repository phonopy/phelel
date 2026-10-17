"""Tests of the plot commands of velph without vaspout.h5."""

from __future__ import annotations

import pathlib

import pytest
from click.testing import CliRunner

from phelel.velph.cli.velph_cmd_root import cmd_root

_GIVE_PATH = "Give the path of vaspout.h5 with --vaspout-filename."


@pytest.mark.parametrize(
    "args,message",
    [
        (
            ["el_bands", "plot", "--window", "-1", "1"],
            '"el_bands/vaspout.h5" not found.\n"el_dos/vaspout.h5" not found.  The '
            'Fermi level is taken from the DOS calculation.\nRun "velph el_dos '
            'generate" and VASP in "el_dos".',
        ),
        (["ph_bands", "plot"], '"ph_bands/vaspout.h5" not found.'),
        (
            ["el_dos", "plot_eigenvalues"],
            f'"el_dos/vaspout.h5" not found.  {_GIVE_PATH}',
        ),
        (
            ["ph_selfenergy", "plot", "selfenergy"],
            f'"ph_selfenergy/vaspout.h5" not found.  {_GIVE_PATH}',
        ),
        (
            ["ph_selfenergy", "plot", "eigenvalues"],
            f'"ph_selfenergy/vaspout.h5" not found.  {_GIVE_PATH}',
        ),
        (
            ["transport", "plot", "selfenergy"],
            f'"transport/vaspout.h5" not found.  {_GIVE_PATH}',
        ),
        (
            ["transport", "plot", "transport"],
            f'"transport/vaspout.h5" not found.  {_GIVE_PATH}',
        ),
        (
            ["transport", "plot", "eigenvalues"],
            f'"transport/vaspout.h5" not found.  {_GIVE_PATH}',
        ),
    ],
)
def test_plot_without_vaspout(
    args: list[str],
    message: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Without vaspout.h5, a plot command stops with exit code 1 and an error."""
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cmd_root, args)
    assert result.exit_code == 1
    assert result.stderr == f"Error: {message}\n"
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []
