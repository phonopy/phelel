"""Tests of velph relax generate."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable

import click
import pytest
from click.testing import CliRunner
from phonopy.structure.atoms import PhonopyAtoms

from phelel.velph.cli.relax.generate import plan_relax
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.io import write_velph_toml
from phelel.velph.config.legacy import parse_legacy_velph_toml
from phelel.velph.config.schema import KpointsMesh, VaspSection


def test_plan_relax(
    nacl_velph_toml_v014: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The job of the first relax is made from [vasp.relax] and [scheduler]."""
    data = nacl_velph_toml_v014()
    config = parse_legacy_velph_toml(data)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "POTCAR").write_text("potcar\n")
    job = plan_relax(config, pathlib.Path("relax/iter1"))
    assert job.directory == pathlib.Path("relax/iter1")
    assert isinstance(job.poscar, PhonopyAtoms)
    assert len(job.poscar) == 8
    assert job.incar == data["vasp"]["relax"]["incar"]
    assert job.kpoints == {"KPOINTS": KpointsMesh(mesh=(12, 12, 12))}
    assert job.scheduler is not None
    assert job.scheduler["job_name"] == "vasp-elph-iter1"
    assert job.copied_files == {"POTCAR": pathlib.Path("POTCAR")}
    assert not (tmp_path / "relax").exists()


def test_plan_relax_contcar(
    nacl_velph_toml_v014: Callable[[], dict], tmp_path: pathlib.Path
):
    """CONTCAR of the previous relax is used as POSCAR when it exists."""
    config = parse_legacy_velph_toml(nacl_velph_toml_v014())
    prev_directory = tmp_path / "relax" / "iter1"
    directory = tmp_path / "relax" / "iter2"
    with pytest.raises(click.ClickException) as e:
        plan_relax(config, directory, prev_directory=prev_directory)
    assert e.value.message == (
        f'"{prev_directory / "CONTCAR"}" not found.  It is used as POSCAR of '
        f'"{directory}".'
    )
    prev_directory.mkdir(parents=True)
    (prev_directory / "CONTCAR").write_text("contcar\n")
    job = plan_relax(config, directory, prev_directory=prev_directory)
    assert job.poscar == prev_directory / "CONTCAR"
    assert job.scheduler is not None
    assert job.scheduler["job_name"] == "vasp-elph-iter2"


def test_relax_generate_new_format(
    nacl_velph_toml_v014: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Velph relax generate reads velph.toml of the new format.

    The INCAR has the tags of [vasp.relax.incar].  The second run uses CONTCAR
    of the first one, and it stops without making a directory when CONTCAR
    does not exist.

    """
    config = parse_legacy_velph_toml(nacl_velph_toml_v014())
    assert config.vasp is not None and config.vasp.input_sets is not None
    relax = config.vasp.input_sets["relax"]
    assert relax.incar is not None and "algo" not in relax.incar
    relax = dataclasses.replace(relax, incar={**relax.incar, "algo": "Fast"})
    config = dataclasses.replace(
        config,
        vasp=VaspSection(input_sets={**config.vasp.input_sets, "relax": relax}),
    )
    write_velph_toml(config, tmp_path / "velph.toml")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(cmd_root, ["relax", "generate"])
    assert result.exit_code == 0, result.output
    incar = (tmp_path / "relax" / "iter1" / "INCAR").read_text()
    assert "ALGO = Fast" in incar
    assert "ISIF = 3" in incar

    result = runner.invoke(cmd_root, ["relax", "generate"])
    assert result.exit_code == 1
    assert '"relax/iter1/CONTCAR" not found.' in result.output
    assert not (tmp_path / "relax" / "iter2").exists()

    (tmp_path / "relax" / "iter1" / "CONTCAR").write_text("contcar\n")
    result = runner.invoke(cmd_root, ["relax", "generate"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "relax" / "iter2" / "POSCAR").read_text() == "contcar\n"
    assert (
        "--job-name=vasp-elph-iter2"
        in (tmp_path / "relax" / "iter2" / "_job.sh").read_text()
    )
