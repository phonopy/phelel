"""Tests of the parser of phelel settings."""

from __future__ import annotations

import pathlib

import pytest

from phelel.cui.phelel_argparse import get_parser
from phelel.cui.settings import PhelelConfParser


def test_finufft_eps_from_conf_file(tmp_path: pathlib.Path):
    """FINUFFT_EPS in a configuration file is read as a float."""
    filename = tmp_path / "phelel.conf"
    filename.write_text("FINUFFT_EPS = 1e-8\n", encoding="utf-8")
    settings = PhelelConfParser(filename=filename).settings
    assert isinstance(settings.finufft_eps, float)
    assert settings.finufft_eps == 1e-8


@pytest.mark.parametrize("value", ["native", "NATIVE", "native-box", "finufft"])
def test_nufft_from_conf_file(tmp_path: pathlib.Path, value: str):
    """NUFFT in a configuration file is read in lower case."""
    filename = tmp_path / "phelel.conf"
    filename.write_text(f"NUFFT = {value}\n", encoding="utf-8")
    settings = PhelelConfParser(filename=filename).settings
    assert settings.nufft == value.lower()


def test_nufft_from_args():
    """--nufft is read, and nufft is None without it."""
    parser = get_parser("run")
    settings = PhelelConfParser(args=parser.parse_args(["--nufft", "native"])).settings
    assert settings.nufft == "native"
    assert PhelelConfParser(args=parser.parse_args([])).settings.nufft is None


def test_nufft_from_conf_file_with_wrong_value(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    """A value other than finufft, native and native-box stops with a message."""
    filename = tmp_path / "phelel.conf"
    filename.write_text("NUFFT = nfft\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        PhelelConfParser(filename=filename)
    out = capsys.readouterr().out
    assert out == (
        'nufft has to be "finufft", "native" or "native-box", not "nfft".\n'
        "Please check the setting tags and options.\n"
    )


@pytest.mark.parametrize("nufft", ["native", "native-box"])
def test_finufft_eps_with_native_nufft(nufft: str, capsys: pytest.CaptureFixture):
    """--finufft-eps with --nufft native or native-box stops with a message."""
    args = get_parser("run").parse_args(["--nufft", nufft, "--finufft-eps", "1e-8"])
    with pytest.raises(SystemExit):
        PhelelConfParser(args=args)
    out = capsys.readouterr().out
    assert out == (
        f"FINUFFT_EPS (--finufft-eps) cannot be given with NUFFT = {nufft} "
        f"(--nufft {nufft}).\n"
        "Please check the setting tags and options.\n"
    )
