"""velph command line tool / velph-convert."""

from __future__ import annotations

import click

from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.legacy import convert_legacy_velph_toml


#
# velph convert
#
@cmd_root.command("convert")
@click.argument("old_filename", type=click.Path(exists=True, dir_okay=False))
@click.argument("new_filename", type=click.Path(dir_okay=False))
@click.help_option("-h", "--help")
def cmd_convert(old_filename: str, new_filename: str):
    """Write velph.toml of velph 0.14 in the format of velph 0.15.

    OLD_FILENAME is the velph.toml written by velph 0.14, which is not changed.
    NEW_FILENAME is the velph.toml to be written, which must not exist.  The
    command prints what was removed and which kspacing was replaced by mesh.

    """
    try:
        messages = convert_legacy_velph_toml(old_filename, new_filename)
    except OSError as e:
        raise click.ClickException(str(e)) from e
    for message in messages:
        click.echo(message)
    click.echo(f'"{new_filename}" was written from "{old_filename}".')
