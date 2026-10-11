"""Tests of velph generate."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pytest
import tomli_w
from click.testing import CliRunner
from phonopy.interface.vasp import read_vasp

from phelel.velph.cli.velph_cmd_root import cmd_root


def test_generate(
    nacl_velph_toml_v014: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """POSCAR-unitcell and POSCAR-primitive are written, and not overwritten."""
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(nacl_velph_toml_v014()))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cmd_root, ["generate"])
    assert result.exit_code == 0, result.output
    assert len(read_vasp(tmp_path / "POSCAR-unitcell")) == 8
    assert len(read_vasp(tmp_path / "POSCAR-primitive")) == 2

    (tmp_path / "POSCAR-unitcell").write_text("kept\n")
    result = runner.invoke(cmd_root, ["generate"])
    assert result.exit_code == 0, result.output
    assert '"POSCAR-unitcell" was not overwritten because it exists.' in result.output
    assert (tmp_path / "POSCAR-unitcell").read_text() == "kept\n"
