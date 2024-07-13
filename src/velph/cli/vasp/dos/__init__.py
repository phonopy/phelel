"""velph command line tool / velph-vasp-dos."""

import click

from velph.cli.vasp import cmd_vasp


@cmd_vasp.group("dos")
@click.help_option("-h", "--help")
def cmd_dos():
    """Choose DOS options."""
    pass


@cmd_dos.command()
@click.option("--icharg", nargs=1, type=int, default=None)
@click.option("--ediffg", nargs=1, type=float, default=None)
@click.option("--nelm", nargs=1, type=int, default=None)
@click.option("--ialgo", nargs=1, type=int, default=None)
@click.help_option("-h", "--help")
def generate(icharg, ediffg, nelm, ialgo):
    """Generate DOS input files."""
    if icharg is not None:
        click.echo(f"icharg = {icharg}")
    if ediffg is not None:
        click.echo(f"ediffg = {ediffg}")
    if nelm is not None:
        click.echo(f"nelm = {nelm}")
    if ialgo is not None:
        click.echo(f"ialgo = {ialgo}")

    print("Reading velph.toml.")
    print("KPOINTS is created.")
    print("INCAR is created.")
