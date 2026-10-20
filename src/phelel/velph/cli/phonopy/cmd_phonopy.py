"""velph command line tool / velph-phonopy."""

from __future__ import annotations

import click

from phelel.velph.cli.phonopy.init import run_init
from phelel.velph.cli.supercells import write_supercell_input_files
from phelel.velph.cli.utils import toml_filename_option
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.jobs.project import read_velph_toml
from phelel.velph.jobs.project_directory import ProjectDirectory


@cmd_root.group("phonopy")
@click.help_option("-h", "--help")
def cmd_phonopy():
    """Supercells of phonopy.

    Make the supercells of phonopy and write their VASP input files.

    """
    pass


#
# velph phonopy init
#
@cmd_phonopy.command("init")
@toml_filename_option
@click.help_option("-h", "--help")
def cmd_init(toml_filename: str):
    """Generate displacements and write phonopy_disp.yaml."""
    ph = run_init(read_velph_toml(toml_filename))

    phonopy_yaml_filename = ProjectDirectory().disp_yaml("phonopy")
    phonopy_yaml_filename.parent.mkdir(parents=True, exist_ok=True)
    ph.save(phonopy_yaml_filename)

    click.echo(f'"{phonopy_yaml_filename}" was written.')
    click.echo('Run "velph phonopy generate" to write the VASP input files.')


#
# velph phonopy generate
#
@cmd_phonopy.command("generate")
@toml_filename_option
@click.help_option("-h", "--help")
def cmd_generate(toml_filename: str):
    """Generate phonopy input files."""
    write_supercell_input_files(toml_filename, "phonopy")
