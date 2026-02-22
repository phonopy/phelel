"""velph command line tool module."""

import click


@click.group()
@click.help_option("-h", "--help")
def cmd_root():
    """Command-line utility to help VASP el-ph calculation."""
    pass
