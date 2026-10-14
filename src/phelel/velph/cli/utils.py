"""Utilities for cli."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import xml.parsers.expat
from collections.abc import Iterator, Sequence
from enum import Enum
from typing import Any, Literal, cast

import click
from numpy.typing import NDArray
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

from phelel.velph.utils.scheduler import (
    get_custom_schedular_script,
    get_sge_scheduler_script,
    get_slurm_scheduler_script,
)


class CellChoice(Enum):
    """Cell choice for specific calculations."""

    UNSPECIFIED = "unspecified"
    UNITCELL = "unitcell"
    PRIMITIVE = "primitive"


class PrimitiveCellChoice(Enum):
    """Primitive cell choice."""

    STANDARDIZED = "standardized"
    REDUCED = "reduced"


def parse_cell_choice(value: Any, name: str) -> CellChoice | None:
    """Return CellChoice of "primitive" or "unitcell".

    value is compared in lower case. Otherwise None is returned with an error
    message, in which name tells where value is given.

    """
    choice = _parse_choice(value, name, (CellChoice.PRIMITIVE, CellChoice.UNITCELL))
    return cast(CellChoice | None, choice)


def parse_primitive_cell_choice(value: Any, name: str) -> PrimitiveCellChoice | None:
    """Return PrimitiveCellChoice of "standardized" or "reduced".

    See parse_cell_choice.

    """
    choice = _parse_choice(value, name, tuple(PrimitiveCellChoice))
    return cast(PrimitiveCellChoice | None, choice)


def _parse_choice(value: Any, name: str, choices: tuple[Enum, ...]) -> Enum | None:
    if isinstance(value, str):
        for choice in choices:
            if value.lower() == choice.value:
                return choice
    allowed = " or ".join(f'"{choice.value}"' for choice in choices)
    click.echo(f'{name} has to be {allowed}, not "{value}".', err=True)
    return None


@dataclasses.dataclass(frozen=True)
class DefaultCellChoices:
    """Default cell choices."""

    nac: CellChoice = CellChoice.PRIMITIVE
    relax: CellChoice = CellChoice.UNITCELL


SupercellCalcType = Literal["phelel", "phonopy", "phono3py"]


@dataclasses.dataclass(frozen=True)
class DisplacementOptions:
    """Options for generating displacements.

    amplitude=None means the default of each calculation type given by
    get_default_amplitude.

    """

    amplitude: float | None = None
    diagonal: bool = False
    max_num_atoms: int | None = None
    number_of_snapshots: int | None = None
    plusminus: bool | Literal["auto"] = True
    supercell_dimension: tuple[int, int, int] | None = None
    supercell_matrix: tuple[int, int, int, int, int, int, int, int, int] | None = None


def get_default_amplitude(calc_type: SupercellCalcType) -> float:
    """Return default displacement distance of a calculation type.

    The default displacement distances of phonopy (for phonopy) and phono3py
    (for phono3py and phelel) for VASP are used.

    """
    if calc_type == "phonopy":
        return get_phonopy_displacement_distance("vasp")
    return get_phono3py_displacement_distance("vasp")


@dataclasses.dataclass(frozen=True)
class VelphInitParams:
    """Basic init parameters of velph.

    amplitude, diagonal, plusminus, phelel_nosym can be parameters for [phelel].
    These are treated specially.

    Except for amplitude, diagonal, plusminus, phelel_nosym, None is equivalent
    to its default value.

    """

    cell_for_nac: CellChoice = CellChoice.UNSPECIFIED
    cell_for_relax: CellChoice = CellChoice.UNSPECIFIED
    find_primitive: bool = True
    displacement_options: DisplacementOptions = DisplacementOptions()
    kspacing: float = 0.1
    kspacing_dense: float = 0.05
    magmom: str | None = None
    site_mixture: str | None = None
    split_site_mixture: bool = False
    phelel_displacement_options: DisplacementOptions | None = None
    phonopy_displacement_options: DisplacementOptions | None = None
    phono3py_displacement_options: DisplacementOptions | None = None
    phelel_nosym: bool = False
    primitive_cell_choice: PrimitiveCellChoice = PrimitiveCellChoice.STANDARDIZED
    symmetrize_cell: bool = False
    tolerance: float = 1e-5
    use_grg: bool = False

    def __contains__(self, key) -> bool:  # noqa: D105
        return hasattr(self, key)

    def __getitem__(self, key) -> Any:  # noqa: D105
        return getattr(self, key)


@dataclasses.dataclass(frozen=True)
class VelphInitOptions:
    """Options for velph-init command.

    This is shared in velph-init options and [init.options] in template-toml.

    """

    amplitude: float | None = None
    cell_for_nac: Literal["primitive", "unitcell"] | None = None
    cell_for_relax: Literal["primitive", "unitcell"] | None = None
    diagonal: bool | None = None
    find_primitive: bool | None = None
    kspacing: float | None = None
    kspacing_dense: float | None = None
    magmom: str | None = None
    max_num_atoms: int | None = None
    site_mixture: str | None = None
    split_site_mixture: bool | None = None
    phelel_nosym: bool | None = None
    plusminus: bool | Literal["auto"] | None = None
    primitive_cell_choice: Literal["standardized", "reduced"] | None = None
    supercell_dimension: tuple[int, int, int] | None = None
    supercell_matrix: tuple[int, int, int, int, int, int, int, int, int] | None = None
    symmetrize_cell: bool | None = None
    tolerance: float | None = None
    use_grg: bool | None = None

    def __getitem__(self, key) -> Any:  # noqa: D105
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:  # noqa: D105
        return (f.name for f in dataclasses.fields(self))

    def items(self) -> Iterator[tuple[str, Any]]:  # noqa: D102
        return ((f.name, getattr(self, f.name)) for f in dataclasses.fields(self))


@dataclasses.dataclass(frozen=True)
class VelphFilePaths:
    """File paths or pointers used in velph-init."""

    cell_filepath: os.PathLike
    velph_template_filepath: os.PathLike | None = None


@dataclasses.dataclass(frozen=True)
class KpointsData:
    """K-points data for VASP calculations."""

    mesh: NDArray | None = None
    D_diag: NDArray | None = None
    shift: NDArray | None = None
    kspacing: float | None = None
    line: int | None = None
    path: (
        tuple[tuple[tuple[float, float, float], tuple[float, float, float]]] | None
    ) = None
    label: tuple[tuple[str, str]] | None = None


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
