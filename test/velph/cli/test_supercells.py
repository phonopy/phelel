"""Tests of phelel.velph.cli.supercells."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable

import click
import pytest
import tomli_w
from click.testing import CliRunner

from phelel.velph.cli.phelel.init import run_init
from phelel.velph.cli.supercells import get_code_section, plan_supercells
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.io import parse_velph_toml
from phelel.velph.config.schema import CodeSection


def test_plan_supercells(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The jobs are disp-000 for the perfect supercell and disp-001, ...

    The directory name is appended to job_name of the scheduler settings.

    """
    data = nacl_velph_toml()
    config = parse_velph_toml(data)
    monkeypatch.chdir(tmp_path)
    phe = run_init(config, current_directory=tmp_path)
    assert phe.supercells_with_displacements is not None
    jobs = plan_supercells(config, "phelel", phe)
    assert [job.directory for job in jobs] == [
        pathlib.Path(f"phelel/disp-{i:03d}")
        for i in range(len(phe.supercells_with_displacements) + 1)
    ]
    assert jobs[0].poscar is phe.supercell
    assert jobs[1].job_script is not None
    assert "#SBATCH --job-name=vasp-elph-001\n" in jobs[1].job_script
    assert jobs[1].incar == data["vasp"]["phelel"]["incar"]
    assert not (tmp_path / "phelel").exists()


def test_supercells_site_mixture_merge(
    site_mixture_velph_toml: Callable[[bool], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Supercells of the merge scheme are written with the unmerged atoms.

    The perfect supercell (disp-000) is the unmerged supercell, the INCAR files
    have the VCA tag, and the hint of the species rows is shown once.

    """
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(site_mixture_velph_toml(False)))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cmd_root, ["phelel", "init"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(cmd_root, ["phelel", "generate"])
    assert result.exit_code == 0, result.output

    directories = sorted(tmp_path.glob("phelel/disp-*"))
    assert len(directories) > 1
    for directory in directories:
        lines = (directory / "POSCAR").read_text().splitlines()
        assert lines[5].split() == ["Ge", "Sn", "Te"]
        assert lines[6].split() == ["8", "8", "8"]
        assert "VCA = 0.5 0.5 1.0" in (directory / "INCAR").read_text()
    assert result.output.count("VASP VCA hint:") == 1


def test_get_code_section(nacl_velph_toml: Callable[[], dict]):
    """Fields not given take the defaults of velph init, and errors name the file.

    The supercell has to be given.

    """
    config = parse_velph_toml(nacl_velph_toml())
    assert config.phelel is not None
    config = dataclasses.replace(
        config,
        phelel=CodeSection(supercell=config.phelel.supercell),
        phonopy=None,
    )
    section = get_code_section(config, "phelel", "velph.toml")
    assert (section.amplitude, section.diagonal, section.plusminus) == (
        0.03,
        False,
        True,
    )
    with pytest.raises(click.ClickException) as e:
        get_code_section(config, "phonopy", "velph.toml")
    assert e.value.message == (
        '[phonopy] is not found in "velph.toml".\n'
        'Add [phonopy] to "velph.toml", or run "velph init" again.'
    )
    config = dataclasses.replace(config, phelel=CodeSection(amplitude=0.02))
    with pytest.raises(click.ClickException) as e:
        get_code_section(config, "phelel", "velph.toml")
    assert e.value.message == (
        '[phelel] of "velph.toml" does not give the supercell.\n'
        'Give "supercell_dimension" or "supercell_matrix" in [phelel].'
    )


def test_generate_without_yaml(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Without the yaml file of init, generate stops before writing files."""
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(nacl_velph_toml()))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cmd_root, ["phonopy", "generate"])
    assert result.exit_code == 1
    assert (
        'Error: "phonopy/phonopy_disp.yaml" not found.\n'
        'Run "velph phonopy init" to make it.'
    ) in result.output
    assert not (tmp_path / "phonopy").exists()


def test_generate_missing_key_of_job_script(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A key missing for the job script stops before any supercell is written."""
    data = nacl_velph_toml()
    data["vasp"]["phelel"]["scheduler"] = {"scheduler_template": "#!/bin/bash\n{pe}\n"}
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(data))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cmd_root, ["phelel", "init"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(cmd_root, ["phelel", "generate"])
    assert result.exit_code == 1
    assert '"pe" used in the job script template is not given' in result.stderr
    assert list(tmp_path.glob("phelel/disp-*")) == []
