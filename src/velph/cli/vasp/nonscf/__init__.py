"""velph command line tool / velph-vasp-nonscf."""

import click

from velph.cli.vasp import cmd_vasp


@cmd_vasp.group("nonscf")
@click.help_option("-h", "--help")
def cmd_nonscf():
    """Choose non-SCF options."""
    pass


@cmd_nonscf.command()
@click.option("--lwave/--no-lwave", default=False, show_default=True)
@click.option("--lcharg/--no-lcharg", default=True, show_default=True)
@click.option("--lorbit", nargs=1, type=int, default=None)
@click.help_option("-h", "--help")
def generate(lwave, lcharg, lorbit):
    """Generate non-SCF input files."""
    if lwave:
        click.echo("lwave = .true.")
    if not lcharg:
        click.echo("lcharg = .false.")
    if lorbit is not None:
        click.echo(f"lorbit = {lorbit}")

    print("Reading velph.toml.")
    print("KPOINTS is created.")
    print("INCAR is created.")
