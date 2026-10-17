"""velph command line tool / velph-scheduler."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import cast

import click

from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.input_sets import InputSetName
from phelel.velph.config.schema import InputSetSection
from phelel.velph.errors import VelphError
from phelel.velph.jobs.vasp_job import job_script, read_velph_toml


#
# velph scheduler
#
@cmd_root.command("scheduler")
@click.argument("set_names", nargs=-1, metavar="[INPUT_SET]...")
@click.option(
    "--toml-filename",
    "toml_filename",
    type=click.Path(),
    default="velph.toml",
    show_default=True,
)
@click.help_option("-h", "--help")
def cmd_scheduler(set_names: tuple[str, ...], toml_filename: str):
    """Show the job script _job.sh of each input set.

    INPUT_SET is the name of an input set in [vasp] of velph.toml, such as
    relax or phelel.  Without INPUT_SET, _job.sh of every input set is shown.
    No file is written.  _job.sh is made as the generate commands make it,
    except that job_name is shown without the name of the calculation, such as
    "-iter1" of relax and "-001" of the supercells.  The exit code is 1 when
    _job.sh of an input set cannot be made.

    """
    config = read_velph_toml(toml_filename)
    input_sets: Mapping[str, InputSetSection] = {}
    if config.vasp is not None and config.vasp.input_sets is not None:
        input_sets = config.vasp.input_sets
    for set_name in set_names:
        if set_name not in input_sets:
            raise click.ClickException(
                f'[vasp.{set_name}] is not found in "{toml_filename}".'
            )
    if config.scheduler is None:
        click.echo(f'"{toml_filename}" has no [scheduler], so no _job.sh is written.')
        for set_name, input_set in input_sets.items():
            if input_set.scheduler is not None:
                click.echo(
                    f"[vasp.{set_name}.scheduler] is not used without [scheduler]."
                )
        return
    if not input_sets:
        click.echo(f'"{toml_filename}" has no input set in [vasp].')
        return

    failed = False
    for i, set_name in enumerate(set_names or input_sets):
        if i > 0:
            click.echo("")
        click.echo(f"[vasp.{set_name}]")
        try:
            text = job_script(config, cast(InputSetName, set_name))
        except VelphError as e:
            click.echo(f"Error: {e}")
            failed = True
            continue
        assert text is not None
        click.echo(text, nl=not text.endswith("\n"))
    if failed:
        sys.exit(1)
