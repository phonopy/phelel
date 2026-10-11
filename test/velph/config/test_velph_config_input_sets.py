"""Tests of the table of the input sets."""

from __future__ import annotations

import doctest
from collections.abc import Callable

import pytest

from phelel.velph.config import input_sets
from phelel.velph.config.input_sets import INPUT_SETS, input_set_fields
from phelel.velph.config.legacy import parse_legacy_velph_toml
from phelel.velph.config.schema import CODE_KEYS


def test_input_set_fields():
    """The fields of an input set are made from its row of INPUT_SETS.

    These are the fields common to all input sets, the other k-point blocks,
    and "cell" for the input sets that choose their cell.  relax and selfenergy
    are checked by the examples in the docstring of input_set_fields.

    """
    common = ("incar", "incar_unset", "kpoints", "scheduler")
    assert input_set_fields("phelel") == common
    assert input_set_fields("el_bands") == common + ("kpoints_opt",)
    assert input_set_fields("el_dos") == common + ("kpoints_dense",)
    assert input_set_fields("ph_bands") == common + ("qpoints",)
    with pytest.raises(KeyError):
        input_set_fields("relx")


def test_docstring_examples():
    """The examples in the docstrings of input_sets.py give what they show."""
    result = doctest.testmod(input_sets)
    assert result.attempted > 0
    assert result.failed == 0


def test_parse_legacy_velph_toml_meets_input_sets(
    nacl_velph_toml_v014: Callable[[], dict],
):
    """Velph init of velph 0.14 writes the k-point blocks of INPUT_SETS."""
    config = parse_legacy_velph_toml(nacl_velph_toml_v014())
    assert config.vasp is not None and config.vasp.input_sets is not None
    assert set(config.vasp.input_sets) == set(INPUT_SETS)
    for set_name, entry in INPUT_SETS.items():
        input_set = config.vasp.input_sets[set_name]
        for block in entry.kpoints_files.values():
            assert getattr(input_set, block) is not None, (set_name, block)


def test_tables_are_read_only():
    """The table of the input sets and that of the code sections cannot be changed."""
    with pytest.raises(TypeError):
        INPUT_SETS["relax"] = INPUT_SETS["nac"]  # type: ignore[index]
    with pytest.raises(TypeError):
        INPUT_SETS["relax"].kpoints_files["KPOINTS"] = "kpoints_dense"  # type: ignore[index]
    with pytest.raises(TypeError):
        CODE_KEYS["phelel"] = ()  # type: ignore[index]
