"""Tests of velph nac generate."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pytest
import tomli_w
from click.testing import CliRunner
from phonopy.structure.atoms import PhonopyAtoms

from phelel.velph.cli.nac.generate import plan_nac
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.io import parse_velph_toml


def test_plan_nac(nacl_velph_toml: Callable[[], dict], tmp_path: pathlib.Path):
    """The job of nac is on the cell of [vasp.nac], here the primitive cell.

    job_name is that of [scheduler], and POTCAR is not copied when the current
    directory does not have it.

    """
    data = nacl_velph_toml()
    assert data["vasp"]["nac"]["cell"] == "primitive"
    config = parse_velph_toml(data)
    with pytest.MonkeyPatch.context() as m:
        m.chdir(tmp_path)
        job = plan_nac(config)
    assert job.directory == pathlib.Path("nac")
    assert isinstance(job.poscar, PhonopyAtoms)
    assert len(job.poscar) == 2
    assert job.incar == data["vasp"]["nac"]["incar"]
    assert job.scheduler is not None
    assert job.scheduler["job_name"] == "vasp-elph"
    assert job.copied_files == {}


def test_nac_generate_unit_cell(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """With cell = "unitcell", POSCAR of nac is the unit cell."""
    data = nacl_velph_toml()
    data["vasp"]["nac"]["cell"] = "unitcell"
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(data))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cmd_root, ["nac", "generate"])
    assert result.exit_code == 0, result.output
    poscar_lines = (tmp_path / "nac" / "POSCAR").read_text().split("\n")
    assert poscar_lines[6].split() == ["4", "4"]
