"""Tests of the root group of the velph commands."""

from __future__ import annotations

import click
from click.testing import CliRunner

from phelel.velph.cli.utils import toml_filename_option
from phelel.velph.cli.velph_cmd_root import _VelphGroup, cmd_root
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


def _commands(group: click.Group, path: tuple[str, ...] = ()):
    """Yield the names and the commands under group, depth first."""
    for name, command in group.commands.items():
        yield (*path, name), command
        if isinstance(command, click.Group):
            yield from _commands(command, (*path, name))


def _toml_filename_parameter(command: click.Command) -> click.Parameter | None:
    for parameter in command.params:
        if parameter.name == "toml_filename":
            return parameter
    return None


def _same_option(a: click.Parameter, b: click.Parameter) -> bool:
    return (a.opts, a.default, getattr(a, "help", None)) == (
        b.opts,
        b.default,
        getattr(b, "help", None),
    )


def _commands_with_other_toml_filename(excluded: set[tuple[str, ...]]) -> list[str]:
    """Return the commands whose --toml-filename is not toml_filename_option.

    The commands in excluded, given by their names under velph, are skipped.

    """

    @toml_filename_option
    @click.command()
    def reference(toml_filename: str):
        pass

    expected = _toml_filename_parameter(reference)
    assert expected is not None
    differ = []
    for path, command in _commands(cmd_root):
        parameter = _toml_filename_parameter(command)
        if path in excluded or parameter is None:
            continue
        if not _same_option(parameter, expected):
            differ.append(" ".join(path))
    return differ


def test_toml_filename_option():
    """The commands that read velph.toml have the same option --toml-filename.

    velph init writes velph.toml, and its option of the same name differs.

    """
    assert _commands_with_other_toml_filename({("init",)}) == []
    assert _commands_with_other_toml_filename(set()) == ["init"]
