"""Utilities for cli."""

from __future__ import annotations

import pathlib
import xml.parsers.expat
from collections.abc import Sequence

import click
from phonopy.interface.vasp import VasprunxmlExpat, get_vasp_vca_hint_lines
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon


def echo_vasp_vca_hint(cell: PhonopyAtoms) -> None:
    """Show POSCAR species rows, POTCAR order and VCA tag of site mixture."""
    if cell.has_weighted_species:
        for line in get_vasp_vca_hint_lines(cell):
            click.echo(line)


def get_num_digits(sequence: Sequence, min_length: int = 3) -> int:
    """Return number of digits of sequence."""
    nd = len(str(len(sequence)))
    if nd < min_length:
        nd = min_length
    return nd


def get_nac_params(
    nac_cell: PhonopyAtoms,
    vasprun_path: pathlib.Path,
    primitive: PhonopyAtoms,
    is_symmetry: bool,
    symprec: float = 1e-5,
) -> dict:
    """Collect NAC parameters from vasprun.xml and return them.

    nac_cell is the cell of the NAC calculation, given by "cell" of [vasp.nac].
    click.ClickException is raised when vasprun.xml cannot be parsed, for
    example when VASP has not finished writing it.

    """
    with open(vasprun_path, "rb") as f:
        try:
            vasprun = VasprunxmlExpat(f)
            vasprun.parse()
        except xml.parsers.expat.ExpatError as e:
            raise click.ClickException(
                f'"{vasprun_path}" cannot be parsed: {e}.\n'
                f'Check that VASP in "{vasprun_path.parent}" has finished.'
            ) from e

    borns_, epsilon_ = symmetrize_borns_and_epsilon(
        vasprun.born,
        vasprun.epsilon,
        nac_cell,
        primitive=primitive,
        symprec=symprec,
        is_symmetry=is_symmetry,
    )

    nac_params = {
        "born": borns_,
        "factor": get_physical_units().Hartree * get_physical_units().Bohr,
        "dielectric": epsilon_,
    }
    return nac_params
