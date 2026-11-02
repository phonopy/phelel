"""Tests of the parser of phelel settings."""

from __future__ import annotations

import pathlib

import pytest

from phelel.cui.phelel_argparse import get_parser
from phelel.cui.settings import PhelelConfParser


def test_finufft_eps_from_conf_file(tmp_path: pathlib.Path):
    """FINUFFT_EPS in a configuration file is read as a float."""
    filename = tmp_path / "phelel.conf"
    filename.write_text(
        "DV_EVALUATION_METHOD = finufft\nFINUFFT_EPS = 1e-8\n", encoding="utf-8"
    )
    settings = PhelelConfParser(filename=filename).settings
    assert isinstance(settings.finufft_eps, float)
    assert settings.finufft_eps == 1e-8


@pytest.mark.parametrize("value", ["shortest", "SHORTEST", "box", "finufft"])
def test_dv_evaluation_method_from_conf_file(tmp_path: pathlib.Path, value: str):
    """DV_EVALUATION_METHOD in a configuration file is read in lower case."""
    filename = tmp_path / "phelel.conf"
    filename.write_text(f"DV_EVALUATION_METHOD = {value}\n", encoding="utf-8")
    settings = PhelelConfParser(filename=filename).settings
    assert settings.dv_evaluation_method == value.lower()


def test_dv_evaluation_method_from_args():
    """--dv-evaluation-method is read, and it is None without the option."""
    parser = get_parser("run")
    settings = PhelelConfParser(
        args=parser.parse_args(["--dv-evaluation-method", "shortest"])
    ).settings
    assert settings.dv_evaluation_method == "shortest"
    assert (
        PhelelConfParser(args=parser.parse_args([])).settings.dv_evaluation_method
        is None
    )


def test_dv_evaluation_method_from_conf_file_with_wrong_value(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    """A value other than shortest, box and finufft stops with a message."""
    filename = tmp_path / "phelel.conf"
    filename.write_text("DV_EVALUATION_METHOD = nfft\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        PhelelConfParser(filename=filename)
    out = capsys.readouterr().out
    assert out == (
        'DV_EVALUATION_METHOD has to be "shortest", "box" or "finufft", not "nfft".\n'
        "Please check the setting tags and options.\n"
    )


@pytest.mark.parametrize("dv_evaluation_method", [None, "shortest", "box"])
def test_finufft_eps_without_finufft(
    dv_evaluation_method: str | None, capsys: pytest.CaptureFixture
):
    """--finufft-eps without --dv-evaluation-method finufft stops with a message.

    Without --dv-evaluation-method, "shortest" is used.

    """
    option = (
        []
        if dv_evaluation_method is None
        else ["--dv-evaluation-method", dv_evaluation_method]
    )
    args = get_parser("run").parse_args([*option, "--finufft-eps", "1e-8"])
    with pytest.raises(SystemExit):
        PhelelConfParser(args=args)
    out = capsys.readouterr().out
    assert out == (
        "FINUFFT_EPS (--finufft-eps) can be given only with "
        "DV_EVALUATION_METHOD = finufft (--dv-evaluation-method finufft).\n"
        "Please check the setting tags and options.\n"
    )
