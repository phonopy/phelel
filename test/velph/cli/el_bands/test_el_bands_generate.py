"""Tests of velph el_bands generate and velph el_dos generate."""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from collections.abc import Callable

import click
import pytest
from click.testing import CliRunner

from phelel.velph.cli.el_bands.generate import plan_el_bands
from phelel.velph.cli.el_dos.generate import plan_el_dos
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.io import parse_velph_toml, write_velph_toml
from phelel.velph.config.schema import KpointsMesh, LinePath, VaspSection, VelphConfig


def _with_el_bands_path(config: VelphConfig) -> VelphConfig:
    """Return config whose [vasp.el_bands.kpoints_opt] gives a path."""
    assert config.vasp is not None and config.vasp.input_sets is not None
    el_bands = dataclasses.replace(
        config.vasp.input_sets["el_bands"],
        kpoints_opt=LinePath(
            line=21, path=(((0.0, 0.0, 0.0), (0.5, 0.0, 0.5)),), label=(("G", "X"),)
        ),
    )
    input_sets = {**config.vasp.input_sets, "el_bands": el_bands}
    return dataclasses.replace(config, vasp=VaspSection(input_sets=input_sets))


def test_plan_el_bands(nacl_velph_toml: Callable[[], dict]):
    """KPOINTS_OPT is the band path that seekpath gives, without "path"."""
    pytest.importorskip("seekpath")
    data = nacl_velph_toml()
    assert "path" not in data["vasp"]["el_bands"]["kpoints_opt"]
    job = plan_el_bands(parse_velph_toml(data))
    assert job.directory == pathlib.Path("el_bands")
    kpoints_opt = job.kpoints["KPOINTS_OPT"]
    assert isinstance(kpoints_opt, LinePath)
    assert kpoints_opt.line == 51
    assert kpoints_opt.label is not None
    assert kpoints_opt.label[0] == ("GAMMA", "X")
    assert isinstance(job.kpoints["KPOINTS"], KpointsMesh)


def test_plan_el_bands_without_seekpath(
    nacl_velph_toml: Callable[[], dict], monkeypatch: pytest.MonkeyPatch
):
    """Without seekpath, a line block has to give "path"."""
    monkeypatch.setitem(sys.modules, "seekpath", None)
    config = parse_velph_toml(nacl_velph_toml())
    with pytest.raises(click.ClickException) as e:
        plan_el_bands(config)
    assert e.value.message == (
        '[vasp.el_bands.kpoints_opt] does not give "path", and seekpath, which '
        "makes the band path, is not installed.\n"
        'Install seekpath, or give "path" and "label" in '
        "[vasp.el_bands.kpoints_opt]."
    )
    job = plan_el_bands(_with_el_bands_path(config))
    assert job.kpoints["KPOINTS_OPT"] == LinePath(
        line=21, path=(((0.0, 0.0, 0.0), (0.5, 0.0, 0.5)),), label=(("G", "X"),)
    )


def test_plan_el_dos(nacl_velph_toml: Callable[[], dict]):
    """KPOINTS_OPT of el_dos is the mesh of [vasp.el_dos.kpoints_dense]."""
    data = nacl_velph_toml()
    job = plan_el_dos(parse_velph_toml(data))
    dos = data["vasp"]["el_dos"]
    assert job.directory == pathlib.Path("el_dos")
    assert job.kpoints == {
        "KPOINTS": KpointsMesh(mesh=tuple(dos["kpoints"]["mesh"])),
        "KPOINTS_OPT": KpointsMesh(mesh=tuple(dos["kpoints_dense"]["mesh"])),
    }
    assert job.incar == dos["incar"]


def test_el_bands_el_dos_generate_new_format(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """el_bands and el_dos are written in the directories of their names."""
    config = _with_el_bands_path(parse_velph_toml(nacl_velph_toml()))
    write_velph_toml(config, tmp_path / "velph.toml")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    for set_name in ("el_bands", "el_dos"):
        result = runner.invoke(cmd_root, [set_name, "generate"])
        assert result.exit_code == 0, result.output
        assert f'VASP input files were written in "{set_name}".' in result.output
    assert (tmp_path / "el_bands" / "KPOINTS_OPT").read_text().split("\n")[:6] == [
        "k points along high symmetry lines by velph",
        "21",
        "line mode",
        "fractional",
        "0.00000000 0.00000000 0.00000000 G",
        "0.50000000 0.00000000 0.50000000 X",
    ]
    assert (tmp_path / "el_dos" / "KPOINTS_OPT").read_text().split("\n")[2] == "Gamma"
