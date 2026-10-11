"""Dataclasses that hold the contents of velph.toml and of velph init templates.

A velph.toml and a velph init template are read into the same dataclasses,
VelphConfig and the sections in it.  Every field of a template is optional.
A field that is not given is None.

Two kinds of fields can hold a value that velph init converts before it
writes velph.toml.  A supercell can be given by the maximum number of atoms
(MaxNumAtoms), and a k-point mesh can be given by a spacing
(KpointsSpacing).  velph init converts them to a supercell matrix and to a
mesh, so a velph.toml has only SupercellDimension or SupercellMatrix as a
supercell and only KpointsMesh as a k-point mesh.

velph init reads settings from several layers: the defaults, the template
and the command line.  The metadata of each field says how the field is
merged when two layers give it (see MERGE_RULE_KEY and merge.py).

The dataclasses are frozen, and the tables in them, such as the INCAR tags
and the scheduler settings, are read-only copies (MappingProxyType) of the
dicts given.  A value in a table, such as a list of MAGMOM, is not copied.
copy.deepcopy and pickle do not work on these dataclasses.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, get_args

CodeName = Literal["phelel", "phonopy", "phono3py"]
CellChoice = Literal["unitcell", "primitive"]
PrimitiveCellChoice = Literal["standardized", "reduced"]
SiteMixtureScheme = Literal["merge", "split"]
Plusminus = bool | Literal["auto"]

IntVector3 = tuple[int, int, int]
IntMatrix3 = tuple[IntVector3, IntVector3, IntVector3]
FloatVector3 = tuple[float, float, float]
FloatMatrix3 = tuple[FloatVector3, FloatVector3, FloatVector3]

CODES: tuple[CodeName, ...] = get_args(CodeName)
CELL_CHOICES: tuple[CellChoice, ...] = get_args(CellChoice)
PRIMITIVE_CELL_CHOICES: tuple[PrimitiveCellChoice, ...] = get_args(PrimitiveCellChoice)
SITE_MIXTURE_SCHEMES: tuple[SiteMixtureScheme, ...] = get_args(SiteMixtureScheme)

# INCAR tags with which VASP generates the k-points itself and does not read
# the k-point files that velph writes.
VASP_KSPACING_TAGS = ("kspacing", "elph_kspacing")

# How a field is merged when two layers give it.  The rule is set by
# dataclasses.field(metadata={MERGE_RULE_KEY: rule}).  A field without MERGE_RULE_KEY,
# such as a supercell or a k-point block, is replaced as a whole by the value
# of the upper layer.
MERGE_RULE_KEY = "merge"
# A section such as [phelel].  Each of its fields is merged by its own rule.
MERGE_RECURSIVE = "recursive"
# A table whose keys are not fixed in the schema, such as INCAR tags or
# scheduler settings.  A key of the upper layer replaces the same key of the
# lower layer, and the other keys of both layers are kept.
MERGE_KEYS = "keys"
# The input sets in [vasp].  An input set given in both layers is merged as a
# section, and an input set given in one layer is kept.
MERGE_SECTIONS = "sections"


def _recursive() -> Any:
    return dataclasses.field(default=None, metadata={MERGE_RULE_KEY: MERGE_RECURSIVE})


def _keys() -> Any:
    return dataclasses.field(default=None, metadata={MERGE_RULE_KEY: MERGE_KEYS})


def _sections() -> Any:
    return dataclasses.field(default=None, metadata={MERGE_RULE_KEY: MERGE_SECTIONS})


def _make_read_only(section: Any, *names: str) -> None:
    """Replace the tables in fields of a frozen section by read-only copies."""
    for name in names:
        table = getattr(section, name)
        if table is not None:
            object.__setattr__(section, name, MappingProxyType(dict(table)))


@dataclasses.dataclass(frozen=True)
class MaxNumAtoms:
    """Supercell given by the maximum number of atoms.

    velph init estimates the supercell matrix from it.  It is accepted only in
    a template.

    """

    max_num_atoms: int


@dataclasses.dataclass(frozen=True)
class SupercellDimension:
    """Supercell matrix given by its three diagonal elements."""

    supercell_dimension: IntVector3


@dataclasses.dataclass(frozen=True)
class SupercellMatrix:
    """Supercell matrix."""

    supercell_matrix: IntMatrix3


SupercellSpec = MaxNumAtoms | SupercellDimension | SupercellMatrix


@dataclasses.dataclass(frozen=True)
class KpointsMesh:
    """k-point mesh given by three integers.

    A 3x3 matrix of integers gives a generalized regular grid instead.

    """

    mesh: IntVector3 | IntMatrix3
    shift: FloatVector3 | None = None


@dataclasses.dataclass(frozen=True)
class KpointsSpacing:
    """k-point mesh given by a spacing in 1/Angstrom.

    velph init computes the mesh from it.  It is accepted only in a template.

    """

    kspacing: float
    shift: FloatVector3 | None = None


KpointsSpec = KpointsMesh | KpointsSpacing


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
CODE_KEYS: Mapping[CodeName, tuple[str, ...]] = MappingProxyType(
    {
        "phelel": (
            "amplitude",
            "diagonal",
            "plusminus",
            "nosym",
            "site_mixture_scheme",
        ),
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
)


@dataclasses.dataclass(frozen=True)
class InputSetSection:
    """Input set: section [vasp.<input set>], such as [vasp.relax].

    An input set holds the VASP input settings of one kind of calculation:
    its INCAR tags, k-point blocks, cell choice and scheduler settings.
    incar holds only the INCAR tags of this input set.  The generate commands
    add the tags of [vasp.incar] that incar does not give, except the tags
    listed in incar_unset.  input_set_fields in input_sets.py gives the fields
    allowed in each input set.

    """

    cell: CellChoice | None = None
    incar: Mapping[str, Any] | None = _keys()
    incar_unset: tuple[str, ...] | None = None
    kpoints: KpointsSpec | None = None
    kpoints_dense: KpointsSpec | None = None
    kpoints_opt: LinePath | None = None
    qpoints: LinePath | None = None
    scheduler: Mapping[str, Any] | None = _keys()

    def __post_init__(self) -> None:
        """Make incar and scheduler read-only."""
        _make_read_only(self, "incar", "scheduler")


@dataclasses.dataclass(frozen=True)
class VaspSection:
    """Section [vasp], which holds [vasp.incar] and the input sets."""

    incar: Mapping[str, Any] | None = _keys()
    input_sets: Mapping[str, InputSetSection] | None = _sections()

    def __post_init__(self) -> None:
        """Make incar and input_sets read-only."""
        _make_read_only(self, "incar", "input_sets")


@dataclasses.dataclass(frozen=True)
class InitSection:
    """Section [init] of a template, which gives velph init options in a file.

    The keys are the click parameter names of the velph init options.  For
    example, --dim is supercell_dimension, which is held in supercell.
    [init] is not written in velph.toml.

    """

    codes: tuple[CodeName, ...] | None = None
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
    scheduler: Mapping[str, Any] | None = _keys()
    symmetry: SymmetrySection | None = _recursive()
    unitcell: Mapping[str, Any] | None = None
    primitive_cell: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Make scheduler read-only."""
        _make_read_only(self, "scheduler")
