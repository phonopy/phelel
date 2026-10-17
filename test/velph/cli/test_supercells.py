"""Tests of phelel.velph.cli.supercells."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pytest
import tomli_w
from click.testing import CliRunner

from phelel.velph.cli.phelel.init import run_init
from phelel.velph.cli.phonopy.init import run_init as run_phonopy_init
from phelel.velph.cli.supercells import plan_supercells
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.io import parse_velph_toml
from phelel.velph.jobs.project import Project


def test_plan_supercells(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The jobs are disp-000 for the perfect supercell and disp-001, ...

    The directory name is appended to job_name of the scheduler settings.

    """
    data = nacl_velph_toml()
    project = Project.from_config(parse_velph_toml(data))
    monkeypatch.chdir(tmp_path)
    phe = run_init(project, current_directory=tmp_path)
    assert phe.supercells_with_displacements is not None
    jobs = plan_supercells(project, "phelel", phe)
    assert [job.directory for job in jobs] == [
        pathlib.Path(f"phelel/disp-{i:03d}")
        for i in range(len(phe.supercells_with_displacements) + 1)
    ]
    assert jobs[0].cell is phe.supercell
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


@pytest.mark.parametrize(
    "code,filenames",
    [
        ("phelel", ("BORN", "phelel_params.hdf5")),
        ("phonopy", ("BORN", "FORCE_SETS", "FORCE_CONSTANTS", "force_constants.hdf5")),
        ("phono3py", ("BORN", "FORCES_FC3", "FORCES_FC2", "fc3.hdf5", "fc2.hdf5")),
    ],
)
def test_generate_ignores_files_in_current_directory(
    code: str,
    filenames: tuple[str, ...],
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The generate commands of the codes read only <code>/<code>_disp.yaml.

    Files that phelel.load, phonopy.load and phono3py.load look for in the
    current directory, such as FORCE_SETS, are not read, also when they are
    broken.

    """
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(nacl_velph_toml()))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cmd_root, [code, "init"])
    assert result.exit_code == 0, result.output
    for filename in filenames:
        (tmp_path / filename).write_text("broken\n")
    result = runner.invoke(cmd_root, [code, "generate"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / code / "disp-001" / "POSCAR").exists()


def test_set_nac_params_without_vasprun(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
):
    """Without nac/vasprun.xml, the message says how to set the NAC parameters."""
    (tmp_path / "nac").mkdir()
    ph = run_phonopy_init(
        Project.from_config(parse_velph_toml(nacl_velph_toml())),
        current_directory=tmp_path,
    )
    assert ph.nac_params is None
    assert (
        f'"{tmp_path / "nac" / "vasprun.xml"}" is not found, so NAC parameters are '
        "not set.\n"
        f'Run VASP in "{tmp_path / "nac"}" and run this command again to set them.\n'
    ) in capsys.readouterr().out


def test_init_with_unparsable_nac_vasprun(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """nac/vasprun.xml that cannot be parsed stops the init command.

    Such a file is written, for example, while VASP is running in "nac".

    """
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(nacl_velph_toml()))
    (tmp_path / "nac").mkdir()
    (tmp_path / "nac" / "vasprun.xml").write_text("<modeling>\n")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cmd_root, ["phonopy", "init"])
    assert result.exit_code == 1, result.output
    assert result.stderr.startswith('Error: "nac/vasprun.xml" cannot be parsed: ')
    assert result.stderr.endswith('Check that VASP in "nac" has finished.\n')
    assert not (tmp_path / "phonopy" / "phonopy_disp.yaml").exists()
