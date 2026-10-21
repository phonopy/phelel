"""Tests of the parser of phelel settings."""

from __future__ import annotations

import pathlib

from phelel.cui.settings import PhelelConfParser


def test_finufft_eps_from_conf_file(tmp_path: pathlib.Path):
    """FINUFFT_EPS in a configuration file is read as a float."""
    filename = tmp_path / "phelel.conf"
    filename.write_text("FINUFFT_EPS = 1e-8\n", encoding="utf-8")
    settings = PhelelConfParser(filename=filename).settings
    assert isinstance(settings.finufft_eps, float)
    assert settings.finufft_eps == 1e-8
