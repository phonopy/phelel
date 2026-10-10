"""Dataclasses that hold the contents of velph.toml and of velph init templates.

A velph.toml and a velph init template are read into the same dataclasses,
VelphConfig and the sections in it.  Every field of a template is optional.
A field that is not given is None.

Two kinds of fields can hold a value that velph init converts before it
writes velph.toml.  A supercell can be given by the maximum number of atoms
(MaxNumAtoms), and a k-point mesh can be given by a spacing (Spacing).  velph
init converts them to a supercell matrix and to a mesh, so a velph.toml has
only Dimension or Matrix as a supercell and only Mesh as a k-point mesh.

velph init reads settings from several layers: the defaults, the template
and the command line.  The metadata of each field says how the field is
merged when two layers give it (see MERGE_KEY and merge.py).

"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Literal

Code = Literal["phelel", "phonopy", "phono3py"]
CellChoice = Literal["unitcell", "primitive"]
PrimitiveCellChoice = Literal["standardized", "reduced"]
SiteMixtureScheme = Literal["merge", "split"]
Plusminus = bool | Literal["auto"]

IntVector3 = tuple[int, int, int]
IntMatrix3 = tuple[IntVector3, IntVector3, IntVector3]
FloatVector3 = tuple[float, float, float]
FloatMatrix3 = tuple[FloatVector3, FloatVector3, FloatVector3]

CODES: tuple[Code, ...] = ("phelel", "phonopy", "phono3py")
CELL_CHOICES: tuple[CellChoice, ...] = ("unitcell", "primitive")
PRIMITIVE_CELL_CHOICES: tuple[PrimitiveCellChoice, ...] = ("standardized", "reduced")
SITE_MIXTURE_SCHEMES: tuple[SiteMixtureScheme, ...] = ("merge", "split")

# INCAR tags with which VASP generates the k-points itself and does not read
# the k-point files that velph writes.
VASP_KSPACING_TAGS = ("kspacing", "elph_kspacing")

# How a field is merged when two layers give it.  The rule is set by
# dataclasses.field(metadata={MERGE_KEY: rule}).  A field without MERGE_KEY,
# such as a supercell or a k-point block, is replaced as a whole by the value
# of the upper layer.
MERGE_KEY = "merge"
# A section such as [phelel].  Each of its fields is merged by its own rule.
RECURSIVE = "recursive"
# A table whose keys are not fixed in the schema, such as INCAR tags or
# scheduler settings.  A key of the upper layer replaces the same key of the
# lower layer, and the other keys of both layers are kept.
KEYS = "keys"
# The input sets in [vasp].  An input set given in both layers is merged as a
# section, and an input set given in one layer is kept.
SECTIONS = "sections"


def _recursive() -> Any:
    return dataclasses.field(default=None, metadata={MERGE_KEY: RECURSIVE})


def _keys() -> Any:
    return dataclasses.field(default=None, metadata={MERGE_KEY: KEYS})


def _sections() -> Any:
    return dataclasses.field(default=None, metadata={MERGE_KEY: SECTIONS})


@dataclasses.dataclass(frozen=True)
class MaxNumAtoms:
    """Supercell given by the maximum number of atoms.

    velph init estimates the supercell matrix from it.  It is accepted only in
    a template.

    """

    max_num_atoms: int


@dataclasses.dataclass(frozen=True)
class Dimension:
    """Supercell matrix given by its three diagonal elements."""

    supercell_dimension: IntVector3


@dataclasses.dataclass(frozen=True)
class Matrix:
    """Supercell matrix."""

    supercell_matrix: IntMatrix3


SupercellSpec = MaxNumAtoms | Dimension | Matrix


@dataclasses.dataclass(frozen=True)
class Mesh:
    """k-point mesh given by three integers.

    A 3x3 matrix of integers gives a generalized regular grid instead.

    """

    mesh: IntVector3 | IntMatrix3
    shift: FloatVector3 | None = None


@dataclasses.dataclass(frozen=True)
class Spacing:
    """k-point mesh given by a spacing in 1/Angstrom.

    velph init computes the mesh from it.  It is accepted only in a template.

    """

    kspacing: float
    shift: FloatVector3 | None = None


KpointsSpec = Mesh | Spacing


@dataclasses.dataclass(frozen=True)
class LinePath:
    """Band path written in line mode.

    line is the number of points on each segment of path.  label gives the
    names of the two ends of each segment.

    """

    line: int
    path: tuple[tuple[FloatVector3, FloatVector3], ...] | None = None
    label: tuple[tuple[str, str], ...] | None = None


@dataclasses.dataclass(frozen=True)
class CodeSection:
    """Section [phelel], [phonopy] or [phono3py].

    CODE_KEYS gives the keys allowed in each of these sections.

    """

    supercell: SupercellSpec | None = None
    amplitude: float | None = None
    diagonal: bool | None = None
    plusminus: Plusminus | None = None
    nosym: bool | None = None
    number_of_snapshots: int | None = None
    site_mixture_scheme: SiteMixtureScheme | None = None


# Keys allowed in [phelel], [phonopy] and [phono3py], besides the three
# supercell keys.
CODE_KEYS: dict[Code, tuple[str, ...]] = {
    "phelel": ("amplitude", "diagonal", "plusminus", "nosym", "site_mixture_scheme"),
    "phonopy": (
        "amplitude",
        "diagonal",
        "plusminus",
        "nosym",
        "number_of_snapshots",
        "site_mixture_scheme",
    ),
    "phono3py": (
        "amplitude",
        "diagonal",
        "plusminus",
        "nosym",
        "number_of_snapshots",
    ),
}


@dataclasses.dataclass(frozen=True)
class InputSetSection:
    """Input set: section [vasp.<input set>], such as [vasp.relax].

    An input set holds the VASP input settings of one kind of calculation:
    its INCAR tags, k-point blocks, cell choice and scheduler settings.
    incar holds only the INCAR tags of this input set.  The generate commands
    add the tags of [vasp.incar] that incar does not give, except the tags
    listed in incar_unset.  INPUT_SET_FIELDS gives the fields allowed in each
    input set.

    """

    cell: CellChoice | None = None
    incar: dict[str, Any] | None = _keys()
    incar_unset: tuple[str, ...] | None = None
    kpoints: KpointsSpec | None = None
    kpoints_dense: KpointsSpec | None = None
    kpoints_opt: LinePath | None = None
    qpoints: LinePath | None = None
    scheduler: dict[str, Any] | None = _keys()


_INPUT_SET_COMMON = ("incar", "incar_unset", "kpoints", "scheduler")

# Fields of InputSetSection allowed in each input set.
INPUT_SET_FIELDS: dict[str, tuple[str, ...]] = {
    "relax": _INPUT_SET_COMMON + ("cell",),
    "nac": _INPUT_SET_COMMON + ("cell",),
    "phelel": _INPUT_SET_COMMON,
    "phonopy": _INPUT_SET_COMMON,
    "phono3py": _INPUT_SET_COMMON,
    "selfenergy": _INPUT_SET_COMMON + ("kpoints_dense",),
    "transport": _INPUT_SET_COMMON + ("kpoints_dense",),
    "ph_selfenergy": _INPUT_SET_COMMON + ("kpoints_dense",),
    "el_bands": _INPUT_SET_COMMON + ("kpoints_opt",),
    "el_dos": _INPUT_SET_COMMON + ("kpoints_dense",),
    "ph_bands": _INPUT_SET_COMMON + ("qpoints",),
}


@dataclasses.dataclass(frozen=True)
class VaspSection:
    """Section [vasp], which holds [vasp.incar] and the input sets."""

    incar: dict[str, Any] | None = _keys()
    input_sets: dict[str, InputSetSection] | None = _sections()


@dataclasses.dataclass(frozen=True)
class InitSection:
    """Section [init] of a template, which gives velph init options in a file.

    The keys are the click parameter names of the velph init options.  For
    example, --dim is supercell_dimension, which is held in supercell.
    [init] is not written in velph.toml.

    """

    codes: tuple[Code, ...] | None = None
    supercell: SupercellSpec | None = None
    amplitude: float | None = None
    diagonal: bool | None = None
    plusminus: Plusminus | None = None
    nosym: bool | None = None
    cell_for_relax: CellChoice | None = None
    cell_for_nac: CellChoice | None = None
    tolerance: float | None = None
    symmetrize_cell: bool | None = None
    find_primitive: bool | None = None
    primitive_cell_choice: PrimitiveCellChoice | None = None
    use_grg: bool | None = None
    kspacing: float | None = None
    kspacing_dense: float | None = None
    magmom: str | None = None
    site_mixture: str | None = None
    split_site_mixture: bool | None = None


@dataclasses.dataclass(frozen=True)
class SymmetrySection:
    """Section [symmetry], written by velph init."""

    tolerance: float | None = None
    spacegroup_type: str | None = None
    uni_number: int | None = None
    primitive_matrix: FloatMatrix3 | None = None


@dataclasses.dataclass(frozen=True)
class VelphConfig:
    """Contents of a velph.toml or of a velph init template.

    version is the value of [velph] version.  unitcell and primitive_cell hold
    the cell dicts that get_cell_dict of phonopy writes and parse_cell_dict of
    phonopy reads.

    """

    version: str | None = None
    init: InitSection | None = _recursive()
    phelel: CodeSection | None = _recursive()
    phonopy: CodeSection | None = _recursive()
    phono3py: CodeSection | None = _recursive()
    vasp: VaspSection | None = _recursive()
    scheduler: dict[str, Any] | None = _keys()
    symmetry: SymmetrySection | None = _recursive()
    unitcell: Mapping[str, Any] | None = None
    primitive_cell: Mapping[str, Any] | None = None
