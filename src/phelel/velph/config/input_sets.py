"""Input sets of velph.toml and what the generate commands write for each.

An input set is a section [vasp.<input set>] of velph.toml, such as
[vasp.relax].  It holds the VASP input settings of one kind of calculation.
The generate command of an input set writes its VASP input files in the
directory named after the input set, for example "relax/iter1" for relax and
"phelel/disp-001" for phelel.

INPUT_SETS is the table of the input sets, and InputSetName of schema.py gives
their names.  input_set_fields gives the fields
allowed in each input set from it.  The table is read-only
(MappingProxyType), and so is kpoints_files of each row.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from phelel.velph.config.schema import CodeName, InputSetName

# Cell on which the VASP calculations of an input set are run (see
# InputSetEntry.cell).
InputSetCell = Literal["cell", "supercells", "primitive"]

# Names of the k-point files that VASP reads.
KpointsFileName = Literal["KPOINTS", "KPOINTS_ELPH", "KPOINTS_OPT", "QPOINTS"]

# Fields of InputSetSection that give the contents of a k-point file.
KpointsBlock = Literal["kpoints", "kpoints_dense", "kpoints_opt", "qpoints"]


@dataclasses.dataclass(frozen=True)
class InputSetEntry:
    """Row of INPUT_SETS for one input set.

    Attributes
    ----------
    cell : InputSetCell
        Cell on which the VASP calculations of the input set are run.
        "cell": the cell given by "cell" of the input set, the unit cell or
        the primitive cell.  "supercells": the supercells with displacements
        made by the code of the same name as the input set.  "primitive":
        [primitive_cell] of velph.toml.
    kpoints_files : Mapping[KpointsFileName, KpointsBlock]
        Name of each k-point file written in the job directory, mapped to the
        k-point block of the input set that gives its contents, for example
        {"KPOINTS": "kpoints", "KPOINTS_ELPH": "kpoints_dense"}.  It is made
        read-only when the row is made.
    required_code : CodeName or None
        Section of velph.toml, [phelel], [phonopy] or [phono3py], that velph
        init needs in order to write the input set.  None means that velph
        init always writes the input set.

    """

    cell: InputSetCell
    kpoints_files: Mapping[KpointsFileName, KpointsBlock]
    required_code: CodeName | None

    def __post_init__(self) -> None:
        """Make kpoints_files read-only."""
        object.__setattr__(
            self, "kpoints_files", MappingProxyType(dict(self.kpoints_files))
        )


INPUT_SETS: Mapping[InputSetName, InputSetEntry] = MappingProxyType(
    {
        "relax": InputSetEntry("cell", {"KPOINTS": "kpoints"}, None),
        "nac": InputSetEntry("cell", {"KPOINTS": "kpoints"}, None),
        "phelel": InputSetEntry("supercells", {"KPOINTS": "kpoints"}, "phelel"),
        "phonopy": InputSetEntry("supercells", {"KPOINTS": "kpoints"}, "phonopy"),
        "phono3py": InputSetEntry("supercells", {"KPOINTS": "kpoints"}, "phono3py"),
        "selfenergy": InputSetEntry(
            "primitive",
            {"KPOINTS": "kpoints", "KPOINTS_ELPH": "kpoints_dense"},
            "phelel",
        ),
        "transport": InputSetEntry(
            "primitive",
            {"KPOINTS": "kpoints", "KPOINTS_ELPH": "kpoints_dense"},
            "phelel",
        ),
        "ph_selfenergy": InputSetEntry(
            "primitive",
            {"KPOINTS": "kpoints", "KPOINTS_ELPH": "kpoints_dense"},
            "phelel",
        ),
        "el_bands": InputSetEntry(
            "primitive", {"KPOINTS": "kpoints", "KPOINTS_OPT": "kpoints_opt"}, None
        ),
        "el_dos": InputSetEntry(
            "primitive", {"KPOINTS": "kpoints", "KPOINTS_OPT": "kpoints_dense"}, None
        ),
        "ph_bands": InputSetEntry(
            "primitive", {"KPOINTS": "kpoints", "QPOINTS": "qpoints"}, "phelel"
        ),
    }
)

# Fields that every input set allows.
_INPUT_SET_COMMON = ("incar", "incar_unset", "kpoints", "scheduler")


def input_set_fields(set_name: InputSetName) -> tuple[str, ...]:
    """Return the fields of InputSetSection allowed in an input set.

    These are the fields common to all input sets, the k-point blocks of its
    k-point files other than kpoints, and "cell" when the input set chooses
    its cell, in this order.  The order is that of the error messages of the
    loader: the allowed keys of an unknown-key error are listed in this order,
    and the missing fields of a velph.toml are checked in this order.

    KeyError is raised when set_name is not the name of an input set.

    Examples
    --------
    >>> input_set_fields("relax")
    ('incar', 'incar_unset', 'kpoints', 'scheduler', 'cell')
    >>> input_set_fields("selfenergy")
    ('incar', 'incar_unset', 'kpoints', 'scheduler', 'kpoints_dense')

    """
    entry = INPUT_SETS[set_name]
    # kpoints, the block of KPOINTS, is one of the common fields.
    blocks = tuple(
        block for block in entry.kpoints_files.values() if block != "kpoints"
    )
    cell = ("cell",) if entry.cell == "cell" else ()
    return _INPUT_SET_COMMON + blocks + cell
