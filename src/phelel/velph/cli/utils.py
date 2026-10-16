"""Utilities for cli."""

from __future__ import annotations

import os
import pathlib
import xml.parsers.expat
from collections.abc import Mapping, Sequence
from typing import Any

import click
from phono3py.interface.calculator import (
    get_default_displacement_distance as get_phono3py_displacement_distance,
)
from phonopy.interface.calculator import (
    get_default_displacement_distance as get_phonopy_displacement_distance,
)
from phonopy.interface.vasp import (
    VasprunxmlExpat,
    get_vasp_vca_hint_lines,
    write_vasp,
)
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon

from phelel.velph.config.schema import CodeName


def get_default_amplitude(calc_type: CodeName) -> float:
    """Return default displacement distance of a calculation type.

    The default displacement distances of phonopy (for phonopy) and phono3py
    (for phono3py and phelel) for VASP are used.

    """
    if calc_type == "phonopy":
        return get_phonopy_displacement_distance("vasp")
    return get_phono3py_displacement_distance("vasp")


def write_poscar(
    directory: os.PathLike, cell: PhonopyAtoms, filename: str = "POSCAR"
) -> None:
    """Write POSCAR.

    A cell with weighted species of site mixture is written for VASP VCA. Atoms
    of one symbol and weight that are next to each other in the cell are written
    in one species row. The INCAR VCA tag is written by ``write_incar``.

    """
    write_vasp(
        pathlib.Path(directory) / filename, cell, for_vca=cell.has_weighted_species
    )


def echo_vasp_vca_hint(cell: PhonopyAtoms) -> None:
    """Show POSCAR species rows, POTCAR order and VCA tag of site mixture."""
    if cell.has_weighted_species:
        for line in get_vasp_vca_hint_lines(cell):
            click.echo(line)


_BRACES_HINT = (
    '  To write "{" or "}" in _job.sh, write "{{" or "}}" in scheduler_template.'
)


def make_job_script(scheduler: Mapping[str, Any]) -> str:
    """Return the job script _job.sh made from the scheduler settings of a job.

    The keys in braces in scheduler_template, such as {job_name}, are replaced
    by the values of the settings, as str.format does.  click.ClickException is
    raised when scheduler_template is not given, when a key that it uses is not
    given, and when str.format cannot read it, for example for a single "}".

    """
    if "scheduler_template" not in scheduler:
        raise click.ClickException(
            '"scheduler_template" has to be given in [scheduler] or '
            "[vasp.<input set>.scheduler]."
        )
    try:
        return scheduler["scheduler_template"].format(**scheduler)
    except KeyError as e:
        raise click.ClickException(
            f'"{e.args[0]}" used in the job script template is not given in '
            "[scheduler] or [vasp.<input set>.scheduler]." + _BRACES_HINT
        ) from e
    except (IndexError, ValueError) as e:
        raise click.ClickException(
            f"The job script template cannot be read: {e}." + _BRACES_HINT
        ) from e


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
) -> dict | None:
    """Collect NAC parameters from vasprun.xml and return them.

    nac_cell is the cell of the NAC calculation, given by "cell" of [vasp.nac].

    """
    with open(vasprun_path, "rb") as f:
        try:
            vasprun = VasprunxmlExpat(f)
            vasprun.parse()
        except xml.parsers.expat.ExpatError:
            click.echo(f'Parsing "{vasprun_path}" failed.')
            return None

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
