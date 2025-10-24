"""Utilities for cli."""

from __future__ import annotations

import copy
import dataclasses
import os
import pathlib
import xml.parsers.expat
from collections.abc import Sequence
from enum import Enum
from typing import Any, Iterator, Literal

import click
import h5py
import numpy as np
import spglib
from numpy.typing import NDArray
from phono3py.phonon.grid import BZGrid
from phonopy.interface.vasp import VasprunxmlExpat, sort_positions_by_symbols
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms, atom_data, parse_cell_dict
from phonopy.structure.cells import (
    get_primitive,
    get_primitive_matrix_by_centring,
    get_reduced_bases,
)
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from spglib import SpglibDataset, SpglibMagneticDataset

from phelel.velph.utils.scheduler import (
    get_custom_schedular_script,
    get_sge_scheduler_script,
    get_slurm_scheduler_script,
)
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


@dataclasses.dataclass(frozen=True)
class DefaultCellChoices:
    """Default cell choices."""

    nac: CellChoice = CellChoice.PRIMITIVE
    relax: CellChoice = CellChoice.UNITCELL


@dataclasses.dataclass(frozen=True)
class DisplacementOptions:
    """Options for generating displacements."""

    amplitude: float = 0.03
    diagonal: bool = False
    max_num_atoms: int | None = None
    number_of_snapshots: int | None = None
    plusminus: bool | Literal["auto"] = True
    supercell_dimension: tuple[int, int, int] | None = None
    supercell_matrix: tuple[int, int, int, int, int, int, int, int, int] | None = None


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
    phelel_nosym: bool | None = None
    plusminus: bool | None = None
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


def write_incar(
    toml_incar_dict: dict,
    directory: os.PathLike,
    cell: PhonopyAtoms | None = None,
    incar_filename: str | os.PathLike = "INCAR",
) -> None:
    """Write INCAR file."""
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
    kspacing_name="kspacing",
) -> None:
    """Write KPOINTS file in mesh mode."""
    if toml_incar_dict.get(kspacing_name) is None:
        try:
            VaspKpoints.write_mesh_mode(
                pathlib.Path(directory) / kpoints_filename, toml_kpoints_dict
            )
        except KeyError:
            click.echo(
                f'Invalid setting of [{tag}]. "{kpoints_filename}" was not made.'
            )
    else:
        click.echo(
            f'"{kpoints_filename}" was not made because of '
            f'"{kspacing_name}" tag in INCAR setting.'
        )


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
    """Write scheduler launch script."""
    sched_string = None
    if "scheduler_name" in toml_scheduler_dict:
        if toml_scheduler_dict["scheduler_name"] == "sge":
            sched_string = get_sge_scheduler_script(toml_scheduler_dict, job_id=job_id)
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
            click.echo(
                '"scheduler_template" has to be specified in scheduler setting.',
                err=True,
            )

        sched_string = get_custom_schedular_script(
            toml_scheduler_dict["scheduler_template"],
            toml_scheduler_dict,
            job_id=job_id,
        )

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
        if "unitcell" in toml_dict["vasp"][calc_type]["cell"]:
            cell = parse_cell_dict(toml_dict["unitcell"])
        elif "primitive" in toml_dict["vasp"][calc_type]["cell"]:
            cell = parse_cell_dict(toml_dict["primitive_cell"])
        else:
            msg = (
                f"[vasp.{calc_type}] cell in {toml_filename} has to be either "
                '"unitcell" or "primitive_cell"'
            )
            click.echo(msg, err=True)
            return None
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


def get_reclat_from_vaspout(fp_vaspout: h5py.File):
    """Return reciprocal space basis vectors.

    Returns
    -------
    reclat : np.ndarray
        Reciprocal basis vectors in row vectors.
        shape=(3, 3)

    """
    # Basis vectors in direct space in column vectors
    lattice = np.transpose(
        fp_vaspout["input"]["poscar"]["lattice_vectors"][:]  # type: ignore
        * fp_vaspout["input"]["poscar"]["scale"][()]  # type: ignore
    )
    # Basis vectors in reciprocal space in row vectors
    reclat = 2 * np.pi * np.linalg.inv(lattice)
    return reclat


def get_distances_along_BZ_path(nk_total, n_segments, nk_per_seg, k_cart):
    """Measure distances of points from origin along paths.

    Returns
    -------
    distances : np.ndarray
        Distances of points from origin along BZ-paths.
        shape=(nk_total,)

    """
    distances = np.zeros(nk_total)
    count = 0
    for _ in range(n_segments):
        for i_pts in range(nk_per_seg):
            # Treatment of jump between equivalent points on BZ boundary
            if i_pts == 0:
                delta_dist = 0
            else:
                delta_dist = np.linalg.norm(k_cart[count, :] - k_cart[count - 1, :])
            distances[count] = distances[count - 1] + delta_dist
            count += 1
    return distances


def get_special_points(
    labels: list[str],
    distances: list[float],
    n_segments: int,
    nk_per_seg: int,
    nk_total: int,
) -> tuple[list, list]:
    """Plot special points at vertical lines and labels."""
    # Left most
    points = []
    labels_at_points = []

    labels_at_points.append(labels[0][0])
    points.append(distances[0])

    count = 0
    for i_seg in range(n_segments):
        for _ in range(nk_per_seg):
            count += 1
        if count != nk_total:
            points.append(distances[count])
            if labels[i_seg * 2 + 1] == labels[i_seg * 2 + 2]:
                labels_at_points.append(labels[i_seg * 2 + 1][0])
            else:
                labels_at_points.append(
                    f"{labels[i_seg * 2 + 1][0]}|{labels[i_seg * 2 + 2][0]}"
                )

    labels_at_points.append(labels[-1][0])
    points.append(distances[-1])

    return points, labels_at_points


def get_symmetry_dataset(
    cell: PhonopyAtoms, tolerance: float = 1e-5
) -> SpglibDataset | SpglibMagneticDataset:
    """Return spglib symmetry dataset."""
    if cell.magnetic_moments is None:
        dataset = spglib.get_symmetry_dataset(cell.totuple(), symprec=tolerance)
    else:
        dataset = spglib.get_magnetic_symmetry_dataset(
            cell.totuple(), symprec=tolerance
        )

    assert dataset is not None

    return dataset


def get_primitive_cell(
    cell: PhonopyAtoms,
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
    tolerance: float = 1e-5,
) -> tuple[PhonopyAtoms, NDArray]:
    """Return primitive cell and transformation matrix.

    This primitive cell is generated from the input cell without
    rigid rotation in contrast to `_get_standardized_unitcell`.

    """
    tmat = sym_dataset.transformation_matrix
    pmat = get_primitive_matrix_from_dataset(sym_dataset)
    total_tmat = np.array(np.linalg.inv(tmat) @ pmat, dtype="double", order="C")

    return (
        get_primitive(cell, primitive_matrix=total_tmat, symprec=tolerance),
        total_tmat,
    )


def get_primitive_matrix_from_dataset(
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
) -> NDArray:
    """Return primitive matrix from symmetry dataset."""
    if isinstance(sym_dataset, SpglibDataset):
        centring = sym_dataset.international[0]
    else:
        centring = spglib.get_spacegroup_type(
            sym_dataset.hall_number
        ).international_short[0]
    return get_primitive_matrix_by_centring(centring)


def get_reduced_cell(
    cell: PhonopyAtoms, method="niggli", tolerance: float = 1e-5
) -> PhonopyAtoms:
    """Return a reduced cell of input cell."""
    reduced_lattice = get_reduced_bases(cell.cell, method=method, tolerance=tolerance)
    assert reduced_lattice is not None
    reduced_positions = cell.scaled_positions @ (
        cell.cell @ np.linalg.inv(reduced_lattice)
    )
    reduced_positions = reduced_positions - np.rint(reduced_positions)
    reduced_positions[:, :] += np.where(reduced_positions < 0, 1, 0)
    reduced_cell = cell.copy()
    reduced_cell.cell = reduced_lattice
    reduced_cell.scaled_positions = reduced_positions
    return reduced_cell


def generate_standardized_cells(
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
    tolerance: float = 1e-5,
) -> tuple[PhonopyAtoms, PhonopyAtoms, NDArray]:
    """Return standardized unit cell and primitive cell."""
    if isinstance(sym_dataset, SpglibDataset):
        click.echo(
            "Crystal structure was standardized based on space-group-type "
            f"{sym_dataset.international}."
        )
    else:
        click.echo(
            "Crystal structure was standardized based on magnetic-space-group-type "
            f"UNI No.{sym_dataset.uni_number}."
        )
    convcell = get_standardized_unitcell(sym_dataset)
    pmat = get_primitive_matrix_from_dataset(sym_dataset)
    if (np.abs(pmat - np.eye(3)) < 1e-8).all():
        primitive = convcell
    else:
        primitive = get_primitive(convcell, primitive_matrix=pmat, symprec=tolerance)

    return convcell, primitive, pmat


def get_standardized_unitcell(
    dataset: SpglibDataset | SpglibMagneticDataset,
) -> PhonopyAtoms:
    """Return conventional unit cell.

    This conventional unit cell can include rigid rotation with respect to
    input unit cell for which symmetry was analized.

    Parameters
    ----------
    cell : PhonopyAtoms
        Input cell.
    dataset : SpgliDataset
        Symmetry dataset of spglib.

    Returns
    -------
    PhonopyAtoms
        Convetional unit cell.

    """
    std_positions = dataset.std_positions
    std_types = dataset.std_types
    _, _, _, perm = sort_positions_by_symbols(std_types, std_positions)
    if isinstance(dataset, SpglibDataset):
        return PhonopyAtoms(
            cell=dataset.std_lattice,
            scaled_positions=std_positions[perm],
            symbols=[atom_data[n][1] for n in std_types[perm]],
        )
    else:
        return PhonopyAtoms(
            cell=dataset.std_lattice,
            scaled_positions=std_positions[perm],
            symbols=[atom_data[n][1] for n in std_types[perm]],
            magnetic_moments=dataset.std_tensors[perm],
        )


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

    Parameters
    ----------
    kpoints_dict : dict
        E.g., {"kspacing": 0.5}
    lattice : array_like
        Basis vectors in row vectors.
        shape=(3, 3)

    Returns
    -------
    SpglibDataset
        Symmetry dataset of spglib.

    """
    kspacing = kpoints_dict["kspacing"]
    symmetry_dataset = get_symmetry_dataset(unitcell)
    gm = BZGrid(
        2 * np.pi / kspacing,
        lattice=unitcell.cell,
        symmetry_dataset=symmetry_dataset,
        use_grg=use_grg,
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
    with open(vasprun_path, "rb") as f:
        try:
            vasprun = VasprunxmlExpat(f)
            vasprun.parse()
        except xml.parsers.expat.ExpatError:
            click.echo(f'Parsing "{vasprun_path}" failed.')
            return None

    nac_cell = convcell
    try:
        if "primitive" in toml_dict["vasp"]["nac"]["cell"]:
            nac_cell = primitive
    except KeyError:
        pass

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
