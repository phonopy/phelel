"""Tests of the velph init command."""

from __future__ import annotations

import pathlib
import re
import shutil
from typing import Any

import click
import pytest
from click.testing import CliRunner, Result
from phonopy.interface.calculator import read_crystal_structure

from phelel.velph.cli.init.build import build_velph_config
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.io import INIT_KEYS, load_template, load_velph_toml
from phelel.velph.config.schema import (
    InitSection,
    MaxNumAtoms,
    SupercellDimension,
    SupercellMatrix,
)

cwd = pathlib.Path(__file__).parent

# Options of velph init that [init] does not give.
_COMMAND_LINE_ONLY = ("force_create", "template_toml_filename", "toml_filename")


@pytest.fixture
def in_tmp_path(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Run in tmp_path with POSCAR_NaCl in it."""
    shutil.copy(cwd / "POSCAR_NaCl", tmp_path / "POSCAR_NaCl")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _velph_init(*args: str) -> Result:
    return CliRunner().invoke(cmd_root, ["init", *args])


def _nacl_cell():
    cell, _ = read_crystal_structure(cwd / "POSCAR_NaCl", interface_mode="vasp")
    return cell


def test_init_writes_velph_toml(in_tmp_path: pathlib.Path):
    """velph.toml is what build_velph_config gives for the options."""
    result = _velph_init("POSCAR_NaCl", "proj", "--dim", "2", "2", "2")
    assert result.exit_code == 0, result.output
    assert 'Created new folder "proj".' in result.output
    expected = build_velph_config(
        _nacl_cell(),
        options=InitSection(
            supercell=SupercellDimension(supercell_dimension=(2, 2, 2))
        ),
    )
    assert load_velph_toml(in_tmp_path / "proj" / "velph.toml") == expected


def test_init_with_template(in_tmp_path: pathlib.Path):
    """The template is passed to build_velph_config."""
    template = in_tmp_path / "template.toml"
    template.write_text(
        '[init]\ncodes = ["phelel"]\n\n[phelel]\nsupercell_dimension = [2, 2, 2]\n'
    )
    result = _velph_init("POSCAR_NaCl", "proj", "--template-toml", "template.toml")
    assert result.exit_code == 0, result.output
    assert 'Read velph init template "template.toml".' in result.output
    config = load_velph_toml(in_tmp_path / "proj" / "velph.toml")
    assert config == build_velph_config(_nacl_cell(), load_template(template))
    assert config.phelel is not None
    assert config.phonopy is None and config.phono3py is None


@pytest.mark.parametrize(
    "args,expected",
    [
        ([], InitSection()),
        (
            ["--code", "phonopy", "--code", "phelel", "--code", "phonopy"],
            InitSection(codes=("phonopy", "phelel")),
        ),
        (
            ["--dim", "1", "2", "3"],
            InitSection(supercell=SupercellDimension(supercell_dimension=(1, 2, 3))),
        ),
        (
            ["--supercell-matrix", "0", "1", "1", "1", "0", "1", "1", "1", "0"],
            InitSection(
                supercell=SupercellMatrix(
                    supercell_matrix=((0, 1, 1), (1, 0, 1), (1, 1, 0))
                )
            ),
        ),
        (
            ["--max-num-atoms", "100"],
            InitSection(supercell=MaxNumAtoms(max_num_atoms=100)),
        ),
        (["--amplitude", "0.02"], InitSection(amplitude=0.02)),
        (["--diagonal", "true"], InitSection(diagonal=True)),
        (["--plusminus"], InitSection(plusminus=True)),
        (["--auto"], InitSection(plusminus="auto")),
        (["--nosym"], InitSection(nosym=True)),
        (["--cell-for-relax", "primitive"], InitSection(cell_for_relax="primitive")),
        (["--cell-for-nac", "UNITCELL"], InitSection(cell_for_nac="unitcell")),
        (["--tolerance", "0.001"], InitSection(tolerance=0.001)),
        (["--symmetrize-cell"], InitSection(symmetrize_cell=True)),
        (["--no-find-primitive"], InitSection(find_primitive=False)),
        (
            ["--primitive-cell-choice", "reduced"],
            InitSection(primitive_cell_choice="reduced"),
        ),
        (["--use-grg"], InitSection(use_grg=True)),
        (["--kspacing", "0.2"], InitSection(kspacing=0.2)),
        (["--kspacing-dense", "0.08"], InitSection(kspacing_dense=0.08)),
        (["--magmom", "8*1"], InitSection(magmom="8*1")),
        (["--site-mixture", "1 1"], InitSection(site_mixture="1 1")),
        (["--split-site-mixture"], InitSection(split_site_mixture=True)),
    ],
)
def test_init_options(
    in_tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    expected: InitSection,
):
    """Each option is given to build_velph_config in InitSection."""
    given: dict[str, Any] = {}

    def _build(input_cell, template, options):
        given["template"] = template
        given["options"] = options
        raise click.ClickException("stopped")

    monkeypatch.setattr("phelel.velph.cli.init.cmd_init.build_velph_config", _build)
    result = _velph_init("POSCAR_NaCl", "proj", *args)
    assert result.exit_code == 1, result.output
    assert given == {"template": None, "options": expected}


@pytest.mark.parametrize(
    "args",
    [
        ["--dim", "2", "2", "2", "--max-num-atoms", "100"],
        ["--dim", "2", "2", "2", "--supercell-matrix", *(["1"] * 9)],
    ],
)
def test_init_supercell_options_given_together(
    in_tmp_path: pathlib.Path, args: list[str]
):
    """Only one of the supercell options can be given."""
    result = _velph_init("POSCAR_NaCl", "proj", *args)
    assert result.exit_code == 2
    assert "Give only one of --dim, " in result.output
    assert not (in_tmp_path / "proj").exists()


def test_init_error_does_not_create_folder(in_tmp_path: pathlib.Path):
    """The project folder is not created when velph init stops with an error."""
    result = _velph_init("POSCAR_NaCl", "proj", "--code", "phelel")
    assert result.exit_code == 1
    assert '"phelel" is in codes, but no supercell is given for it.' in result.output
    assert not (in_tmp_path / "proj").exists()


@pytest.mark.parametrize(
    "args,message",
    [
        (["POSCAR_X", "proj"], '"POSCAR_X" not found.'),
        (
            ["POSCAR_NaCl", "proj", "--template-toml", "x.toml"],
            '"x.toml" not found.',
        ),
    ],
)
def test_init_file_not_found(in_tmp_path: pathlib.Path, args: list[str], message: str):
    """A file not found is an error."""
    result = _velph_init(*args)
    assert result.exit_code == 1
    assert f"Error: {message}" in result.output
    assert not (in_tmp_path / "proj").exists()


def test_init_old_template(in_tmp_path: pathlib.Path):
    """A template of velph earlier than 0.15 is an error."""
    (in_tmp_path / "template.toml").write_text("[init.options]\nkspacing = 0.2\n")
    result = _velph_init("POSCAR_NaCl", "proj", "--template-toml", "template.toml")
    assert result.exit_code == 1
    assert "is a velph init template of velph earlier than 0.15." in result.output
    assert not (in_tmp_path / "proj").exists()


def test_init_existing_velph_toml(in_tmp_path: pathlib.Path):
    """An existing velph.toml is overwritten only with --force."""
    (in_tmp_path / "proj").mkdir()
    velph_toml = in_tmp_path / "proj" / "velph.toml"
    velph_toml.write_text("old\n")

    result = _velph_init("POSCAR_NaCl", "proj")
    assert result.exit_code == 1
    assert 'Folder "proj" already exists.  Use --force' in result.output
    assert velph_toml.read_text() == "old\n"

    result = _velph_init("POSCAR_NaCl", ".", "--toml-filename", "proj/velph.toml")
    assert result.exit_code == 1
    assert '"proj/velph.toml" exists.  Use --force to overwrite it.' in result.output
    assert velph_toml.read_text() == "old\n"

    result = _velph_init("POSCAR_NaCl", "proj", "--force")
    assert result.exit_code == 0, result.output
    assert '"proj/velph.toml" exists, but will be overwritten.' in result.output
    assert load_velph_toml(velph_toml).version is not None


def test_init_project_folder_is_a_file(in_tmp_path: pathlib.Path):
    """A file of the name of the project folder is an error, also with --force."""
    (in_tmp_path / "proj").write_text("file\n")
    result = _velph_init("POSCAR_NaCl", "proj", "--force")
    assert result.exit_code == 1
    assert '"proj" is not a folder.' in result.output


def test_init_in_current_directory(in_tmp_path: pathlib.Path):
    """The current directory can be the project folder."""
    result = _velph_init("POSCAR_NaCl", ".", "--toml-filename", "a.toml")
    assert result.exit_code == 0, result.output
    assert 'Project directory: ".".' in result.output
    assert load_velph_toml(in_tmp_path / "a.toml").version is not None


def test_init_potcar(in_tmp_path: pathlib.Path):
    """POTCAR in the current directory is copied to the project folder."""
    potcar = "\n".join(
        [
            "   TITEL  = PAW_PBE Na 08Apr2002",
            "   ENMAX  =  101.0; ENMIN  =   75.0 eV",
            "   TITEL  = PAW_PBE Cl 06Sep2000",
            "   ENMAX  =  262.5; ENMIN  =  196.9 eV",
        ]
    )
    (in_tmp_path / "POTCAR").write_text(potcar)
    result = _velph_init("POSCAR_NaCl", "proj")
    assert result.exit_code == 0, result.output
    assert "  PAW_PBE Cl 06Sep2000" in result.output
    assert 'Max ENMAX in "POTCAR" is 262.5.' in result.output
    assert (in_tmp_path / "proj" / "POTCAR").read_text() == potcar


def test_init_help_keys():
    """The help of each option gives its key of [init].

    The key in parentheses is the click parameter name, and the keys are those
    of [init].

    """
    keys = []
    for param in cmd_root.commands["init"].params:
        if not isinstance(param, click.Option) or param.name in _COMMAND_LINE_ONLY:
            continue
        if param.name == "help":
            continue
        assert param.help is not None
        match = re.search(r"\((\w+): ", param.help)
        assert match is not None, param.name
        assert match.group(1) == param.name
        keys.append(param.name)
    assert sorted(keys) == sorted(INIT_KEYS)
