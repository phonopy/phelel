"""Implementation of velph-ph_selfenergy-generate."""

import pathlib

from phelel.velph.cli.selfenergy.generate import write_selfenergy_input_files


def write_input_files(toml_filepath: pathlib.Path, dry_run: bool):
    """Generate ph_selfenergy input files."""
    write_selfenergy_input_files(toml_filepath, dry_run, "ph_selfenergy")
