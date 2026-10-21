"""Tests of velph ph_bands generate."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import click
import pytest
from click.testing import CliRunner

from phelel.velph.cli.ph_bands.generate import plan_ph_bands
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.io import parse_velph_toml
from phelel.velph.config.schema import LinePath
from phelel.velph.config.toml_writer import write_velph_toml
from phelel.velph.jobs.project import Project
from phelel.velph.jobs.project_directory import ProjectDirectory

# phelel_params.hdf5 in the current directory, which the tests change to tmp_path.
PHELEL_PARAMS = ProjectDirectory().phelel_params


def test_plan_ph_bands(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """phelel_params.hdf5 is required, and QPOINTS is a band path."""
    pytest.importorskip("seekpath")
    project = Project.from_config(parse_velph_toml(nacl_velph_toml()))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(click.ClickException) as e:
        plan_ph_bands(project)
    assert e.value.message == (
        '"phelel/phelel_params.hdf5" not found.  It is copied to "ph_bands".\n'
        'Run "velph phelel differentiate" to make it.'
    )
    PHELEL_PARAMS.parent.mkdir()
    PHELEL_PARAMS.write_text("hdf5\n")
    job = plan_ph_bands(project)
    assert job.directory == pathlib.Path("ph_bands")
    assert isinstance(job.kpoints["QPOINTS"], LinePath)
    assert job.kpoints["QPOINTS"].path is not None
    assert job.copied_files == {"phelel_params.hdf5": PHELEL_PARAMS}


def test_ph_bands_generate_new_format(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """ph_bands is written in "ph_bands", and not without phelel_params.hdf5."""
    pytest.importorskip("seekpath")
    write_velph_toml(parse_velph_toml(nacl_velph_toml()), tmp_path / "velph.toml")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cmd_root, ["ph_bands", "generate"])
    assert result.exit_code == 1
    assert not (tmp_path / "ph_bands").exists()

    PHELEL_PARAMS.parent.mkdir()
    PHELEL_PARAMS.write_text("hdf5\n")
    result = runner.invoke(cmd_root, ["ph_bands", "generate"])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (tmp_path / "ph_bands").iterdir()) == [
        "INCAR",
        "KPOINTS",
        "POSCAR",
        "QPOINTS",
        "_job.sh",
        "phelel_params.hdf5",
    ]
