"""Tests of the fixtures of velph tests."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from phelel.velph.config.io import parse_velph_toml
from phelel.velph.config.legacy import parse_legacy_velph_toml
from phelel.velph.config.schema import SupercellDimension, VelphConfig

CODES = ("phelel", "phonopy", "phono3py")
CELLS = ("unitcell", "primitive_cell")


def test_nacl_velph_toml_equals_v014(
    nacl_velph_toml: Callable[[], dict], nacl_velph_toml_v014: Callable[[], dict]
):
    """nacl_velph_toml gives the settings of the velph.toml of velph 0.14.

    Both are written by velph init for the conventional unit cell of NaCl with
    --dim 2 2 2, and the file of velph 0.14 is read by the legacy reader.  They
    differ in three ways.  [velph] version differs.  velph 0.14 did not write
    nosym, which is false in nacl_velph_toml.  velph 0.14 wrote the masses with
    six decimals.

    """
    config = parse_velph_toml(nacl_velph_toml())
    config_v014 = parse_legacy_velph_toml(nacl_velph_toml_v014())
    for code in CODES:
        section = getattr(config, code)
        assert section.supercell == SupercellDimension(supercell_dimension=(2, 2, 2))
        assert section.nosym is False
        assert getattr(config_v014, code) == dataclasses.replace(section, nosym=None)
    for key in CELLS:
        cell, cell_v014 = getattr(config, key), getattr(config_v014, key)
        assert _without_masses(cell) == _without_masses(cell_v014)
        np.testing.assert_allclose(_masses(cell), _masses(cell_v014), rtol=0, atol=1e-6)
    assert _other_sections(config) == _other_sections(config_v014)


def _without_masses(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Return cell dict whose points have no mass."""
    points = [
        {key: value for key, value in point.items() if key != "mass"}
        for point in cell["points"]
    ]
    return {**cell, "points": points}


def _masses(cell: Mapping[str, Any]) -> list[float]:
    """Return masses of the points of cell dict."""
    return [point["mass"] for point in cell["points"]]


def _other_sections(config: VelphConfig) -> VelphConfig:
    """Return config without [velph] version, the code sections and the cells."""
    return dataclasses.replace(
        config, version=None, **dict.fromkeys(CODES + CELLS, None)
    )
