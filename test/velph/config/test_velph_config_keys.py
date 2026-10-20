"""Tests that the names kept in several places of velph agree.

The keys of [init], of the sections of velph.toml and of [phelel], [phonopy]
and [phono3py] are written in the dataclasses of schema.py, in the tables of
the loader (io.py) and in the options of velph init.  These tests fail when
one of them is changed without the others.

"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import pytest
from click.testing import CliRunner

from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.io import _TOP_KEYS, INIT_KEYS, SUPERCELL_KEYS, parse_init
from phelel.velph.config.schema import (
    CODE_KEYS,
    CodeSection,
    InitSection,
    VelphConfig,
)

# Options of velph init that [init] does not give.
_NOT_INIT_OPTIONS = {
    "cell_filename",
    "project_folder",
    "force_create",
    "template_toml_filename",
    "toml_filename",
}

# A value of each key of [init] that parse_init accepts.  Only one of
# SUPERCELL_KEYS is given, because they cannot be given together.
_INIT_VALUES: dict[str, Any] = {
    "codes": ["phelel"],
    "supercell_dimension": [2, 2, 2],
    "amplitude": 0.02,
    "diagonal": True,
    "plusminus": "auto",
    "nosym": True,
    "cell_for_relax": "primitive",
    "cell_for_nac": "unitcell",
    "tolerance": 1e-4,
    "symmetrize_cell": True,
    "find_primitive": False,
    "primitive_cell_choice": "reduced",
    "use_grg": True,
    "kspacing": 0.2,
    "kspacing_dense": 0.1,
    "magmom": "1",
    "site_mixture": "1",
    "split_site_mixture": True,
}


def _field_names(cls: type) -> set[str]:
    return {field.name for field in dataclasses.fields(cls)}


def test_options_of_velph_init_are_init_keys():
    """The options of velph init, except those for the files, are INIT_KEYS."""
    command = cmd_root.commands["init"]
    names = {parameter.name for parameter in command.params} - {"help"}
    assert names - _NOT_INIT_OPTIONS == set(INIT_KEYS)
    assert _NOT_INIT_OPTIONS <= names


def test_velph_init_gives_all_init_keys(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Velph init gives the value of every key of [init] to _init_section.

    A key missing in the dict of cmd_init is an option that velph init ignores.

    """
    given = []

    def _init_section(options: dict[str, Any]) -> InitSection:
        given.append(options)
        return InitSection()

    monkeypatch.setattr("phelel.velph.cli.init.cmd_init._init_section", _init_section)
    monkeypatch.chdir(tmp_path)
    CliRunner().invoke(cmd_root, ["init", "POSCAR", "project"])
    assert len(given) == 1
    assert set(given[0]) == set(INIT_KEYS)


def test_parse_init_reads_all_keys():
    """parse_init gives every field of InitSection from the keys of [init].

    The test needs a value in _INIT_VALUES for each key of INIT_KEYS, except
    the other keys of the supercell.

    """
    keys = [key for key in INIT_KEYS if key not in SUPERCELL_KEYS]
    init = parse_init(
        {"supercell_dimension": [2, 2, 2]} | {key: _INIT_VALUES[key] for key in keys}
    )
    assert [
        field.name
        for field in dataclasses.fields(InitSection)
        if getattr(init, field.name) is None
    ] == []


def test_top_keys_are_fields_of_velph_config():
    """The sections of velph.toml and of a template are fields of VelphConfig.

    [velph] gives the field version.  Every field is a section of velph.toml
    or of a template.

    """
    fields = {"velph": "version"}
    names = _field_names(VelphConfig)
    keys = {key for kind_keys in _TOP_KEYS.values() for key in kind_keys}
    assert {fields.get(key, key) for key in keys} == names


def test_code_keys_are_fields_of_code_section():
    """The keys of each code and the supercell are the fields of CodeSection.

    A field allowed in none of the codes, or a key that is not a field, makes
    the test fail.  Which code allows which field, such as number_of_snapshots
    for phonopy and phono3py only, is not checked, because only CODE_KEYS
    says which code allows which field.

    """
    names = _field_names(CodeSection)
    for keys in CODE_KEYS.values():
        assert set(keys) <= names
    keys = {key for code_keys in CODE_KEYS.values() for key in code_keys}
    assert keys | {"supercell"} == names
