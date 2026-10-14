"""Utilities for cli."""

from __future__ import annotations

import os
import pathlib
import xml.parsers.expat
from collections.abc import Sequence

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
from phelel.velph.utils.scheduler import (
    get_custom_schedular_script,
    get_sge_scheduler_script,
    get_slurm_scheduler_script,
)


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


def write_launch_script(
    toml_scheduler_dict: dict, directory: os.PathLike, job_id: str | None = None
) -> None:
    """Write scheduler launch script.

    click.ClickException is raised when a setting that the job script template
    needs is missing.

    """
    sched_string = None
    try:
        if "scheduler_name" in toml_scheduler_dict:
            if toml_scheduler_dict["scheduler_name"] == "sge":
                sched_string = get_sge_scheduler_script(
                    toml_scheduler_dict, job_id=job_id
                )
            elif toml_scheduler_dict["scheduler_name"] == "slurm":
                sched_string = get_slurm_scheduler_script(
                    toml_scheduler_dict, job_id=job_id
                )

        if sched_string is None:
            if "custom_template" in toml_scheduler_dict:
                raise RuntimeError(
                    'Key "custom_template" is obsoleted. Use "scheduler_template".'
                )

            if "scheduler_template" not in toml_scheduler_dict:
                raise click.ClickException(
                    '"scheduler_template" has to be given in [scheduler] or '
                    '[vasp.CALC_TYPE.scheduler], unless "scheduler_name" is '
                    '"sge" or "slurm".'
                )

            sched_string = get_custom_schedular_script(
                toml_scheduler_dict["scheduler_template"],
                toml_scheduler_dict,
                job_id=job_id,
            )
    except KeyError as e:
        raise click.ClickException(
            f'"{e.args[0]}" used in the job script template is not given in '
            "[scheduler] or [vasp.CALC_TYPE.scheduler]."
        ) from e

    if sched_string:
        with open(pathlib.Path(directory) / "_job.sh", "w") as w:
            w.write(sched_string)


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
