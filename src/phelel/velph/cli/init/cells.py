"""Cells, symmetry and supercells made by velph init.

make_cells makes the unit cell and the primitive cell of velph.toml from the
input cell, and the section [symmetry].  resolve_supercell makes the
supercell matrix of a code from the supercell given to velph init.

"""

from __future__ import annotations

import dataclasses

import click
import numpy as np
import spglib
from numpy.typing import NDArray
from phonopy.interface.vasp import get_vasp_structure_lines
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import (
    apply_site_mixture,
    estimate_supercell_matrix,
    generate_standardized_cells,
    get_supercell,
    group_by_key,
)
from spglib import SpglibDataset, SpglibMagneticDataset

from phelel.velph.config.schema import (
    CodeName,
    MaxNumAtoms,
    PrimitiveCellChoice,
    SupercellDimension,
    SupercellMatrix,
    SupercellSpec,
    SymmetrySection,
)
from phelel.velph.jobs.project import get_supercell_matrix
from phelel.velph.utils.structure import (
    get_primitive_cell,
    get_reduced_cell,
    get_symmetry_dataset,
)
from phelel.velph.utils.vasp_input import expand_incar_value


@dataclasses.dataclass(frozen=True)
class InitCells:
    """Cells made by velph init.

    Attributes
    ----------
    unitcell : PhonopyAtoms
        [unitcell] of velph.toml, from which the supercells are made.
    primitive : PhonopyAtoms
        [primitive_cell] of velph.toml.
    sym_dataset : SpglibDataset or SpglibMagneticDataset
        Symmetry dataset of the input cell with the magnetic moments and the
        weights of site mixture given to velph init.
    symmetry : SymmetrySection
        [symmetry] of velph.toml.

    """

    unitcell: PhonopyAtoms
    primitive: PhonopyAtoms
    sym_dataset: SpglibDataset | SpglibMagneticDataset
    symmetry: SymmetrySection


def make_cells(
    input_cell: PhonopyAtoms,
    *,
    tolerance: float,
    symmetrize_cell: bool,
    find_primitive: bool,
    primitive_cell_choice: PrimitiveCellChoice,
    magmom: str | None,
    site_mixture: str | None,
) -> InitCells:
    """Return the cells of velph.toml made from input_cell.

    magmom is the value of the INCAR tag MAGMOM, such as "2*1 2*-1", and
    site_mixture gives the weights of the atoms, such as "0.5 0.5 1".  They
    cannot be given together.  input_cell is not modified.

    The unit cell is the standardized cell of spglib with symmetrize_cell, and
    input_cell otherwise.  The primitive cell is the unit cell without
    find_primitive.  With find_primitive, it is the standardized primitive cell
    of spglib with symmetrize_cell; otherwise it is input_cell when input_cell
    is primitive, and else the primitive cell made from input_cell by the
    transformation matrix of the symmetry dataset, which keeps the Cartesian
    orientation of input_cell.  With primitive_cell_choice "reduced", the
    primitive cell is then reduced.

    click.ClickException is raised when magmom and site_mixture are given
    together or when site_mixture does not fit input_cell.

    """
    cell = input_cell.copy()
    if magmom is not None and site_mixture is not None:
        raise click.ClickException("--site-mixture cannot be combined with --magmom.")
    if magmom is not None:
        cell.magnetic_moments = expand_incar_value(magmom.strip())
    if site_mixture is not None:
        # The co-located atoms are kept as the atoms of the input cell.  With
        # the merge scheme, phonopy merges them into sites.
        weights = [float(x) for x in site_mixture.split()]
        try:
            cell = apply_site_mixture(cell, weights, symprec=tolerance)
        except ValueError as e:
            raise click.ClickException(str(e)) from e

    sym_dataset = get_symmetry_dataset(cell, tolerance=tolerance)
    _echo_space_group(sym_dataset)
    if symmetrize_cell:
        unitcell, primitive = _get_standardized_cells(
            cell, sym_dataset, tolerance, find_primitive
        )
    else:
        unitcell = cell
        primitive = _get_primitive_of_input_cell(
            cell, sym_dataset, tolerance, find_primitive
        )
    click.echo("Supercell is generated with respect to the cell below.")
    click.echo("-" * 80)
    click.echo(str(unitcell))
    click.echo("-" * 80)
    if primitive_cell_choice == "reduced":
        primitive = get_reduced_cell(primitive, tolerance=tolerance)

    return InitCells(
        unitcell=unitcell,
        primitive=primitive,
        sym_dataset=sym_dataset,
        symmetry=_get_symmetry_section(sym_dataset, tolerance, unitcell, primitive),
    )


def resolve_supercell(
    code: CodeName,
    supercell: SupercellSpec,
    cells: InitCells,
    *,
    symmetrize_cell: bool,
    tolerance: float,
) -> SupercellDimension | SupercellMatrix:
    """Return the supercell matrix of code in the form written in velph.toml.

    MaxNumAtoms gives the largest supercell of the standardized cell that keeps
    the lattice symmetry and has at most max_num_atoms atoms.  It needs
    symmetrize_cell, because the supercell is made from the unit cell, which is
    then the standardized cell.  A diagonal supercell matrix is returned as
    SupercellDimension.

    The point group of the supercell is that of the rotations that spglib finds
    for it with tolerance.  When it has fewer rotations than that of the unit
    cell, a warning is shown.

    click.ClickException is raised for MaxNumAtoms without symmetrize_cell.

    """
    if isinstance(supercell, MaxNumAtoms):
        if not symmetrize_cell:
            raise click.ClickException(
                f'The supercell of [{code}] is given by "max_num_atoms" '
                '(--max-num-atoms), which needs "symmetrize_cell = true" '
                "(--symmetrize-cell)."
            )
        smat = np.diag(
            estimate_supercell_matrix(
                cells.sym_dataset, max_num_atoms=supercell.max_num_atoms
            )
        )
    else:
        smat = get_supercell_matrix(supercell)
    smat = np.array(smat, dtype=int)
    _echo_if_point_group_lowered(code, smat, cells.unitcell, tolerance)
    if np.array_equal(smat, np.diag(np.diag(smat))):
        return SupercellDimension(
            supercell_dimension=tuple(int(n) for n in np.diag(smat))  # type: ignore[arg-type]
        )
    return SupercellMatrix(
        supercell_matrix=tuple(tuple(int(n) for n in row) for row in smat)  # type: ignore[arg-type]
    )


def _echo_space_group(sym_dataset: SpglibDataset | SpglibMagneticDataset) -> None:
    if isinstance(sym_dataset, SpglibDataset):
        click.echo(f"Space-group: {sym_dataset.international}")
    else:
        spg_type = spglib.get_spacegroup_type(sym_dataset.hall_number)
        assert spg_type is not None
        click.echo(f"Magnetic-space-group type-{sym_dataset.msg_type}")
        click.echo(f"  Uni-number: {sym_dataset.uni_number}")
        click.echo(f"  Reference space-group-type: {spg_type.international_short}")


def _get_standardized_cells(
    cell: PhonopyAtoms,
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
    tolerance: float,
    find_primitive: bool,
) -> tuple[PhonopyAtoms, PhonopyAtoms]:
    """Return the standardized unit cell and the primitive cell of spglib."""
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
    unitcell, primitive, tmat = generate_standardized_cells(
        cell, sym_dataset, symprec=tolerance
    )
    _echo_if_species_rows_differ(cell, unitcell)
    if not find_primitive:
        return unitcell, unitcell
    if len(primitive) != len(unitcell):
        click.echo(
            "Transformation matrix from conventional unit cell to primitive cell is"
        )
        _echo_matrix(tmat)
    return unitcell, primitive


def _get_primitive_of_input_cell(
    cell: PhonopyAtoms,
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
    tolerance: float,
    find_primitive: bool,
) -> PhonopyAtoms:
    """Return the primitive cell when the unit cell is the input cell."""
    primitive, tmat = get_primitive_cell(cell, sym_dataset, tolerance=tolerance)
    if find_primitive:
        if len(primitive) == len(cell):
            click.echo(
                "Input cell was already a primitive cell, and it is used as the "
                "primitive cell."
            )
            return cell
        click.echo("Found a primitive cell whose transformation matrix is")
        _echo_matrix(tmat)
        return primitive
    if len(primitive) != len(cell):
        click.echo(
            "Input cell is not a primitive cell from the symmetry point of view."
        )
        click.echo("But velph will consider the input cell as the primitive cell.")
        click.echo(
            "For reference, below is a potential primitive cell found in the input "
            "cell."
        )
        click.echo("-" * 70)
        click.echo("\n".join(get_vasp_structure_lines(primitive)).strip())
        click.echo("-" * 70)
    return cell


def _get_symmetry_section(
    sym_dataset: SpglibDataset | SpglibMagneticDataset,
    tolerance: float,
    unitcell: PhonopyAtoms,
    primitive: PhonopyAtoms,
) -> SymmetrySection:
    """Return [symmetry].

    primitive_matrix is given only when the primitive cell has fewer atoms
    than the unit cell.

    """
    if isinstance(sym_dataset, SpglibDataset):
        spacegroup_type, uni_number = sym_dataset.international, None
    else:
        spacegroup_type, uni_number = None, int(sym_dataset.uni_number)
    primitive_matrix = None
    if len(unitcell) != len(primitive):
        pmat = (primitive.cell @ np.linalg.inv(unitcell.cell)).T
        primitive_matrix = tuple(tuple(float(x) for x in row) for row in pmat)
    return SymmetrySection(
        tolerance=tolerance,
        spacegroup_type=spacegroup_type,
        uni_number=uni_number,
        primitive_matrix=primitive_matrix,  # type: ignore[arg-type]
    )


def _echo_if_point_group_lowered(
    code: CodeName, smat: NDArray, unitcell: PhonopyAtoms, tolerance: float
) -> None:
    """Show a warning when the supercell has a lower point group than unitcell.

    The point group is the set of the rotations that spglib finds.

    """
    num_rotations = _count_rotations(unitcell, tolerance)
    supercell = get_supercell(unitcell, smat, symprec=tolerance)
    num_supercell_rotations = _count_rotations(supercell, tolerance)
    if num_supercell_rotations >= num_rotations:
        return
    click.echo(
        f"Warning: the supercell of [{code}] made with the supercell matrix "
        f"{smat.tolist()} has {num_supercell_rotations} rotations, fewer than "
        f"the {num_rotations} of the unit cell.  --symmetrize-cell or a "
        "supercell_matrix that keeps the point group gives a supercell with the "
        "rotations of the unit cell."
    )


def _count_rotations(cell: PhonopyAtoms, tolerance: float) -> int:
    rotations = get_symmetry_dataset(cell, tolerance=tolerance).rotations
    return len({tuple(r.ravel()) for r in rotations})


def _echo_matrix(matrix: NDArray) -> None:
    for v in matrix:
        click.echo(f"  [{v[0]:6.3f} {v[1]:6.3f} {v[2]:6.3f}]")


def _get_species_rows(cell: PhonopyAtoms) -> tuple[list[str], list[int]]:
    """Return labels and numbers of atoms of species rows of POSCAR.

    A species row is a group of atoms of one species, i.e., of one symbol and
    weight of site mixture, that are next to each other in the cell.  The label
    has the weight, e.g. "Ge(0.5)", for a weighted species.

    """
    counts, _, _ = group_by_key(cell.species_ids, consecutive=True)
    starts = np.cumsum([0] + counts[:-1])
    weights = cell.mixture_weights
    labels = []
    for i in starts:
        if weights is None or weights[i] == 1.0:
            labels.append(cell.symbols[i])
        else:
            labels.append(f"{cell.symbols[i]}({weights[i]:g})")
    return labels, counts


def _echo_if_species_rows_differ(input_cell: PhonopyAtoms, unitcell: PhonopyAtoms):
    """Show species rows when standardization changes number or order of atoms."""
    input_rows = _get_species_rows(input_cell)
    rows = _get_species_rows(unitcell)
    if input_rows == rows:
        return
    click.echo("Number or order of atoms was changed by the standardization.")
    for name, (labels, counts) in (
        ("Input structure", input_rows),
        ("Unit cell", rows),
    ):
        click.echo(
            f"  {name + ':':17s}{' '.join(labels)} / {' '.join(str(n) for n in counts)}"
        )
