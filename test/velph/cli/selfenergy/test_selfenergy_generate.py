"""Tests of velph selfenergy, transport and ph_selfenergy generate."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import click
import pytest
from click.testing import CliRunner

from phelel.velph.cli.selfenergy.generate import plan_elph
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.io import parse_velph_toml
from phelel.velph.config.schema import KpointsMesh
from phelel.velph.config.toml_writer import write_velph_toml
from phelel.velph.jobs.project import Project
from phelel.velph.jobs.project_directory import ProjectDirectory

# phelel_params.hdf5 in the current directory, which the tests change to tmp_path.
PHELEL_PARAMS = ProjectDirectory().phelel_params


def test_plan_elph(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The job of transport is made from [vasp.transport] on the primitive cell."""
    data = nacl_velph_toml()
    project = Project.from_config(parse_velph_toml(data))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "POTCAR").write_text("potcar\n")
    PHELEL_PARAMS.parent.mkdir()
    PHELEL_PARAMS.write_text("hdf5\n")
    job = plan_elph(project, "transport")
    transport = data["vasp"]["transport"]
    assert job.directory == pathlib.Path("transport")
    assert len(job.cell) == 2
    assert job.incar == transport["incar"]
    assert job.kpoints == {
        "KPOINTS": KpointsMesh(mesh=tuple(transport["kpoints"]["mesh"])),
        "KPOINTS_ELPH": KpointsMesh(mesh=tuple(transport["kpoints_dense"]["mesh"])),
    }
    assert job.job_script is not None
    assert "#SBATCH --job-name=vasp-elph\n" in job.job_script
    assert job.copied_files == {
        "POTCAR": pathlib.Path("POTCAR"),
        "phelel_params.hdf5": PHELEL_PARAMS,
    }
    assert not (tmp_path / "transport").exists()


def test_plan_elph_without_phelel_params(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """phelel_params.hdf5 is required, except in a dry run.

    A dry run adds NELM = 0 and ELPH_RUN = False to the INCAR and does not copy
    phelel_params.hdf5.

    """
    data = nacl_velph_toml()
    project = Project.from_config(parse_velph_toml(data))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(click.ClickException) as e:
        plan_elph(project, "selfenergy")
    assert e.value.message == (
        '"phelel/phelel_params.hdf5" not found.  It is copied to "selfenergy".\n'
        'Run "velph phelel differentiate" to make it, or run this command with '
        "--dry-run."
    )
    job = plan_elph(project, "selfenergy", dry_run=True)
    assert job.incar == {
        **data["vasp"]["selfenergy"]["incar"],
        "nelm": 0,
        "elph_run": False,
    }
    assert job.copied_files == {}


def test_elph_generate_new_format(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The generate commands read velph.toml of the new format.

    Without phelel_params.hdf5, the command stops without making a directory.

    """
    write_velph_toml(parse_velph_toml(nacl_velph_toml()), tmp_path / "velph.toml")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cmd_root, ["ph_selfenergy", "generate"])
    assert result.exit_code == 1
    assert '"phelel/phelel_params.hdf5" not found.' in result.output
    assert not (tmp_path / "ph_selfenergy").exists()

    PHELEL_PARAMS.parent.mkdir()
    PHELEL_PARAMS.write_text("hdf5\n")
    for set_name in ("selfenergy", "transport", "ph_selfenergy"):
        result = runner.invoke(cmd_root, [set_name, "generate"])
        assert result.exit_code == 0, result.output
        directory = tmp_path / set_name
        assert sorted(p.name for p in directory.iterdir()) == [
            "INCAR",
            "KPOINTS",
            "KPOINTS_ELPH",
            "POSCAR",
            "_job.sh",
            "phelel_params.hdf5",
        ]
        incar = (directory / "INCAR").read_text()
        assert "ENCUT = 500" in incar and "NELM" not in incar
