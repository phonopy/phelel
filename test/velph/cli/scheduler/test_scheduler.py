"""Tests of velph scheduler."""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable

import pytest
import tomli_w
from click.testing import CliRunner

from phelel.velph.cli.velph_cmd_root import cmd_root

# _job.sh of [scheduler] of the nacl_velph_toml fixture, without the name of
# the calculation in job_name.
JOB_SCRIPT = (
    "#!/bin/bash\n"
    "#SBATCH --job-name=vasp-elph\n"
    "#SBATCH --nodes=1\n"
    "#SBATCH --time=96:00:00\n"
    "#SBATCH --output=ci_%j.log\n"
    "\n"
    "\n"
    "mpirun vasp_std\n"
    "\n"
)


def _run_scheduler(
    data: dict, args: list[str], tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(data))
    monkeypatch.chdir(tmp_path)
    return CliRunner().invoke(cmd_root, ["scheduler", *args])


def test_scheduler_all_input_sets(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """_job.sh of every input set is shown, and the errors are shown in place.

    The input sets after an error are shown, the exit code is 1, and no file
    is written.

    """
    data = nacl_velph_toml()
    data["vasp"]["nac"]["scheduler"] = {"scheduler_template": "{qos}\n"}
    result = _run_scheduler(data, [], tmp_path, monkeypatch)
    assert result.exit_code == 1
    # ["", "relax", text of relax, "nac", text of nac, ...], where each text but
    # the last ends with the blank line between the input sets.
    parts = re.split(r"^\[vasp\.(\w+)\]\n", result.stdout, flags=re.MULTILINE)
    assert parts[0] == ""
    assert parts[1::2] == [
        "relax",
        "nac",
        "phelel",
        "phonopy",
        "phono3py",
        "selfenergy",
        "transport",
        "ph_selfenergy",
        "el_bands",
        "el_dos",
        "ph_bands",
    ]
    texts = parts[2::2]
    assert texts[0] == JOB_SCRIPT + "\n"
    assert texts[1] == (
        'Error: "qos" used in the job script template is not given in '
        '[scheduler] or [vasp.<input set>.scheduler].  To write "{" or "}" in '
        '_job.sh, write "{{" or "}}" in scheduler_template.\n\n'
    )
    assert texts[2] == JOB_SCRIPT + "\n"
    assert texts[-1] == JOB_SCRIPT
    assert [p.name for p in tmp_path.iterdir()] == ["velph.toml"]


def test_scheduler_input_sets_given(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Only the input sets given are shown, in the order given."""
    result = _run_scheduler(
        nacl_velph_toml(), ["phelel", "relax"], tmp_path, monkeypatch
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "[vasp.phelel]\n" + JOB_SCRIPT + "\n[vasp.relax]\n" + JOB_SCRIPT
    )


def test_scheduler_unknown_input_set(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """An input set not in velph.toml stops the command before any output."""
    data = nacl_velph_toml()
    del data["vasp"]["nac"]
    result = _run_scheduler(data, ["relax", "nac"], tmp_path, monkeypatch)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert 'Error: [vasp.nac] is not found in "velph.toml".' in result.stderr


def test_scheduler_without_scheduler(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Without [scheduler], the input sets with scheduler settings are named."""
    data = nacl_velph_toml()
    data["vasp"]["nac"]["scheduler"] = data.pop("scheduler")
    result = _run_scheduler(data, [], tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    assert result.stdout == (
        '"velph.toml" has no [scheduler], so no _job.sh is written.\n'
        "[vasp.nac.scheduler] is not used without [scheduler].\n"
    )
