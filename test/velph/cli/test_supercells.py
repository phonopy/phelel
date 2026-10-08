"""Tests of phelel.velph.cli.supercells."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pytest

from phelel.velph.cli.phelel.init import run_init
from phelel.velph.cli.supercells import write_supercells


def test_write_supercells_site_mixture_merge(
    site_mixture_velph_toml: Callable[[bool], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Supercells of the merge scheme are written with the unmerged atoms.

    The perfect supercell (disp-000) is the unmerged supercell, the INCAR files
    have the VCA tag, and the hint of the species rows is shown once.

    """
    toml_dict = site_mixture_velph_toml(False)
    phe = run_init(toml_dict, current_directory=tmp_path)
    assert phe.unmerged_supercell is not None
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()
    write_supercells(phe, toml_dict)

    directories = sorted(tmp_path.glob("phelel/disp-*"))
    assert len(directories) == 1 + len(phe.supercells_with_displacements or [])
    for directory in directories:
        lines = (directory / "POSCAR").read_text().splitlines()
        assert lines[5].split() == ["Ge", "Sn", "Te"]
        assert lines[6].split() == ["8", "8", "8"]
        assert "VCA = 0.5 0.5 1.0" in (directory / "INCAR").read_text()
    assert capsys.readouterr().out.count("VASP VCA hint:") == 1
