"""velph command line tool / velph-scheduler."""

from __future__ import annotations

import sys
from typing import cast

import click

from phelel.velph.cli.utils import toml_filename_option
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.schema import InputSetName
from phelel.velph.errors import VelphError
from phelel.velph.jobs.project import read_velph_toml
from phelel.velph.jobs.vasp_job import job_script


#
# velph scheduler
#
@cmd_root.command("scheduler")
@click.argument("set_names", nargs=-1, metavar="[INPUT_SET]...")
@toml_filename_option
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
    project = read_velph_toml(toml_filename)
    input_sets = project.input_sets
    for set_name in set_names:
        if set_name not in input_sets:
            raise click.ClickException(
                f'[vasp.{set_name}] is not found in "{toml_filename}".'
            )
    # The names given are names of input sets, as checked above.
    names = cast(tuple[InputSetName, ...], set_names) or tuple(input_sets)
    if project.config.scheduler is None:
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
    for i, set_name in enumerate(names):
        if i > 0:
            click.echo("")
        click.echo(f"[vasp.{set_name}]")
        try:
            text = job_script(project, set_name)
        except VelphError as e:
            click.echo(f"Error: {e}")
            failed = True
            continue
        assert text is not None
        click.echo(text, nl=not text.endswith("\n"))
    if failed:
        sys.exit(1)
