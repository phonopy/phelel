"""Implementation of velph-transport-generate."""

import pathlib

from phelel.velph.cli.selfenergy.generate import write_selfenergy_input_files


def write_input_files(toml_filepath: pathlib.Path, dry_run: bool):
    """Generate transport input files."""
    write_selfenergy_input_files(toml_filepath, dry_run, calc_type="transport")
