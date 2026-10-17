"""Tests of the root group of the velph commands."""

from __future__ import annotations

import click
from click.testing import CliRunner

from phelel.velph.cli.velph_cmd_root import _VelphGroup
from phelel.velph.errors import VelphError


def _make_root(exception: Exception) -> click.Group:
    """Return a root group with a command in a subgroup that raises exception.

    The depth is that of the velph commands, such as "velph relax generate".

    """

    @click.group(cls=_VelphGroup)
    def root():
        pass

    @root.group()
    def sub():
        pass

    @sub.command()
    def run():
        raise exception

    return root


def test_velph_error_in_command_of_subgroup():
    """VelphError raised in a command of a subgroup is shown as an error.

    The root group converts it, because click runs the subcommands inside
    Group.invoke of the root group.  This test fails when a change of click
    makes the subcommands run outside it.

    """
    result = CliRunner().invoke(_make_root(VelphError("Message.")), ["sub", "run"])
    assert result.exit_code == 1
    assert result.stderr == "Error: Message.\n"


def test_other_error_in_command_of_subgroup():
    """An exception other than VelphError is not converted."""
    result = CliRunner().invoke(_make_root(RuntimeError("Bug.")), ["sub", "run"])
    assert isinstance(result.exception, RuntimeError)
    assert result.stderr == ""
