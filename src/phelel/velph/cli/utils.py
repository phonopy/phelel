"""Utilities for cli."""

from __future__ import annotations

import copy
import dataclasses
import os
import pathlib
import xml.parsers.expat
from collections.abc import Sequence
from enum import Enum
from typing import Any, Iterator, Literal, cast

import click
import numpy as np
from numpy.typing import NDArray
from phono3py.interface.calculator import (
    get_default_displacement_distance as get_phono3py_displacement_distance,
)
from phonopy.interface.calculator import (
    get_default_displacement_distance as get_phonopy_displacement_distance,
)
from phonopy.interface.vasp import VasprunxmlExpat
from phonopy.phonon.grid import BZGrid
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms, parse_cell_dict
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from spglib import SpglibDataset, SpglibMagneticDataset

from phelel.velph.utils.scheduler import (
    get_custom_schedular_script,
    get_sge_scheduler_script,
    get_slurm_scheduler_script,
)
from phelel.velph.utils.structure import get_symmetry_dataset
from phelel.velph.utils.vasp import VaspIncar, VaspKpoints


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


# VASP INCAR tags that make VASP generate k-points without KPOINTS files.
VASP_KSPACING_TAGS = ("kspacing", "elph_kspacing")


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


def get_displacement_options(
    calc_dict: dict,
    calc_type: SupercellCalcType,
    number_of_snapshots: int | None = None,
) -> DisplacementOptions:
    """Return displacement options in a velph.toml section.

    calc_dict is [phelel], [phonopy], or [phono3py] of velph.toml given by
    calc_type. A missing key gives the default that velph init writes:
    get_default_amplitude for amplitude and the defaults of
    DisplacementOptions for diagonal and plusminus. With number_of_snapshots
    (random displacements), diagonal and plusminus are False.

    """
    amplitude = calc_dict.get("amplitude", get_default_amplitude(calc_type))
    if number_of_snapshots:
        return DisplacementOptions(
            amplitude=amplitude,
            diagonal=False,
            plusminus=False,
            number_of_snapshots=number_of_snapshots,
        )
    return DisplacementOptions(
        amplitude=amplitude,
        diagonal=calc_dict.get("diagonal", DisplacementOptions.diagonal),
        plusminus=calc_dict.get("plusminus", DisplacementOptions.plusminus),
    )


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


def write_incar(
    toml_incar_dict: dict,
    directory: os.PathLike,
    cell: PhonopyAtoms | None = None,
    incar_filename: str | os.PathLike = "INCAR",
) -> None:
    """Write INCAR file.

    kspacing and elph_kspacing are not accepted, because velph always writes
    KPOINTS files.

    """
    for key in toml_incar_dict:
        if key.lower() in VASP_KSPACING_TAGS:
            raise click.ClickException(
                f'INCAR tag "{key}" cannot be used in velph, because velph '
                "always writes KPOINTS files. Remove it from velph.toml and give "
                "the k-point mesh in the kpoints blocks."
            )
    incar_dict = copy.deepcopy(toml_incar_dict)
    if cell is not None and cell.magnetic_moments is not None:
        incar_dict["magmom"] = cell.magnetic_moments.tolist()
    VaspIncar.write(pathlib.Path(directory) / incar_filename, incar_dict)


def write_kpoints_mesh_mode(
    toml_incar_dict: dict,
    directory: os.PathLike,
    tag: str,
    toml_kpoints_dict: dict,
    kpoints_filename="KPOINTS",
) -> None:
    """Write KPOINTS file in mesh mode."""
    try:
        VaspKpoints.write_mesh_mode(
            pathlib.Path(directory) / kpoints_filename, toml_kpoints_dict
        )
    except KeyError:
        click.echo(f'Invalid setting of [{tag}]. "{kpoints_filename}" was not made.')


def write_kpoints_line_mode(
    cell: PhonopyAtoms,
    directory: os.PathLike,
    tag: str,
    toml_kpoints_dict: dict,
    kpoints_filename: str | os.PathLike = "KPOINTS",
) -> None:
    """Write KPOINTS file in line mode."""
    VaspKpoints.write_line_mode(
        pathlib.Path(directory) / kpoints_filename, cell, toml_kpoints_dict
    )


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


def get_scheduler_dict(toml_dict: dict, calc_type: str) -> dict:
    """Collect and return scheduler information.

    This function extracts scheduler dict from ``toml_dict``.

    Scheduler information is written in [scheduler] section. This information
    can be overwritten in ``each calc_type`` by writing parameters in
    [vasp.calc_type.scheduler] section.

    Parameters
    ----------
    toml_dict : dict
        Dictionary after parsing velph.toml.
    calc_type : str
        This can be written in either way "supercell" or "supercell.phonon".
        This string is splitted by period (".") and the list of strings is used
        as keys of ``toml_dict["vasp"]["key1"]["key2"]``.

    """
    scheduler_dict = copy.deepcopy(toml_dict["scheduler"])
    calc_type_keys = calc_type.split(".")
    tmp_dict = toml_dict["vasp"]
    for key in calc_type_keys:
        tmp_dict = tmp_dict[key]
    if "scheduler" in tmp_dict:
        scheduler_dict.update(tmp_dict["scheduler"])
    return scheduler_dict


def assert_kpoints_mesh_symmetry(
    toml_dict: dict, kpoints_dict: dict, primitive: PhonopyAtoms
):
    """Check if mesh grid respects crystallographic point group or not."""
    if "kspacing" in kpoints_dict:
        symmetry_dataset = kspacing_to_mesh(kpoints_dict, primitive)
        if "symmetry" in toml_dict:
            if isinstance(symmetry_dataset, SpglibDataset):
                if "spacegroup_type" in toml_dict["symmetry"]:
                    assert (
                        symmetry_dataset.international
                        == toml_dict["symmetry"]["spacegroup_type"]
                    )
            else:
                if "uni_number" in toml_dict["symmetry"]:
                    assert (
                        symmetry_dataset.uni_number
                        == toml_dict["symmetry"]["uni_number"]
                    )


def choose_cell_in_dict(
    toml_dict: dict,
    toml_filename: pathlib.Path,
    calc_type: Literal["relax", "nac"],
) -> PhonopyAtoms | None:
    """Return unit cell, primitive cell, or Niggli reduced cell from toml_dict.

    Unit cell and primitive cell have to exist in velph.toml. But Niggli reduced
    cell is optional.

    """
    if "cell" in toml_dict["vasp"][calc_type]:
        cell_choice = parse_cell_choice(
            toml_dict["vasp"][calc_type]["cell"],
            f"cell in [vasp.{calc_type}] of {toml_filename}",
        )
        if cell_choice is None:
            return None
        if cell_choice is CellChoice.UNITCELL:
            cell = parse_cell_dict(toml_dict["unitcell"])
        else:
            cell = parse_cell_dict(toml_dict["primitive_cell"])
    else:
        if dataclasses.asdict(DefaultCellChoices())[calc_type] is CellChoice.PRIMITIVE:
            cell = parse_cell_dict(toml_dict["primitive_cell"])
            click.echo(f"Primitive cell was used for {calc_type}.")
        elif dataclasses.asdict(DefaultCellChoices())[calc_type] is CellChoice.UNITCELL:
            cell = parse_cell_dict(toml_dict["unitcell"])
            click.echo(f"Unitcell was used for {calc_type}.")
        else:
            raise RuntimeError("This should not happen.")

    return cell


def get_num_digits(sequence: Sequence, min_length: int = 3) -> int:
    """Return number of digits of sequence."""
    nd = len(str(len(sequence)))
    if nd < min_length:
        nd = min_length
    return nd


def kspacing_to_mesh(
    kpoints_dict: dict, unitcell: PhonopyAtoms, use_grg: bool = True
) -> SpglibDataset | SpglibMagneticDataset:
    """Update kpoints_dict by mesh corresponding to kspacing.

    The mesh numbers follow the VASP INCAR tag KSPACING,
    max(1, ceiling(|b_i| 2 pi / kspacing)).

    Parameters
    ----------
    kpoints_dict : dict
        E.g., {"kspacing": 0.5}
    lattice : array_like
        Basis vectors in row vectors.
        shape=(3, 3)

    Returns
    -------
    SpglibDataset, SpglibMagneticDataset
        Symmetry dataset of spglib.

    """
    kspacing = kpoints_dict["kspacing"]
    symmetry_dataset = get_symmetry_dataset(unitcell)
    gm = BZGrid(
        2 * np.pi / kspacing,
        lattice=unitcell.cell,
        symmetry_dataset=symmetry_dataset,
        use_grg=use_grg,
        rounding="ceiling",
    )
    if gm.grid_matrix is None:
        kpoints_dict["mesh"] = gm.D_diag.tolist()
    else:
        kpoints_dict["mesh"] = gm.grid_matrix.tolist()
    return symmetry_dataset


def check_fft(toml_filename: str, calculation_name: str) -> None:
    """Show [NGX, NGY, NGZ] in vasprun.xml."""
    vasprun_path = pathlib.Path(calculation_name) / "vasprun.xml"
    if vasprun_path.is_file():
        with open(vasprun_path, "rb") as f:
            vasprun = VasprunxmlExpat(f)
            vasprun.parse()
            click.echo(
                f'Modify [phelel] section in "{toml_filename}" '
                f'as "fft_mesh = {vasprun.fft_grid}"'
            )
    else:
        click.echo(f'"{vasprun_path}" not found.')
        click.echo("For estimating FFT mesh numbers, prepare dry-run by")
        click.echo(f"velph elph generate -d -c {calculation_name}")


def get_nac_params(
    toml_dict: dict,
    vasprun_path: pathlib.Path,
    primitive: PhonopyAtoms | None,
    convcell: PhonopyAtoms,
    is_symmetry: bool,
    symprec: float = 1e-5,
) -> dict | None:
    """Collect NAC parameters from vasprun.xml and return them."""
    nac_cell = convcell
    if "cell" in toml_dict["vasp"].get("nac", {}):
        cell_choice = parse_cell_choice(
            toml_dict["vasp"]["nac"]["cell"], "cell in [vasp.nac]"
        )
        if cell_choice is None:
            return None
        if cell_choice is CellChoice.PRIMITIVE:
            nac_cell = primitive

    with open(vasprun_path, "rb") as f:
        try:
            vasprun = VasprunxmlExpat(f)
            vasprun.parse()
        except xml.parsers.expat.ExpatError:
            click.echo(f'Parsing "{vasprun_path}" failed.')
            return None

    assert nac_cell is not None
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
