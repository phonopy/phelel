"""Read velph.toml and velph init templates.

parse_template and parse_velph_toml read a parsed TOML document into
VelphConfig, and load_template and load_velph_toml read a file.  load_toml
reads a TOML file into a dict.  The two kinds of files are read in different
ways:

- every field of a template is optional.  [init] is accepted, and so are
  max_num_atoms and kspacing in a k-point block, which velph init converts
  to a supercell matrix and to a mesh, and [vasp.incar] and incar_unset,
  from which velph init makes the INCAR tags of each input set.
- a velph.toml has to give [velph] version, the cells and the fields that
  the generate commands read.  max_num_atoms, kspacing in a k-point block,
  [vasp.incar] and incar_unset are errors.

An unknown key is an error in both kinds of files.

A velph.toml without [velph] version and a template that has keys of velph
earlier than 0.15, such as [init.options], are errors, which check_old_forms
of old_forms.py reports before the file is read.  velph.toml is written by
toml_writer.py.

"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import IO, Any, Literal, NoReturn, cast

from phonopy.structure.atoms import parse_cell_dict

from phelel.velph.config.input_sets import INPUT_SETS, input_set_fields
from phelel.velph.config.old_forms import (
    DIGEST_TAIL,
    SCHEDULER_NAME_REMOVED,
    check_old_forms,
)
from phelel.velph.config.schema import (
    CELL_CHOICES,
    CODE_KEYS,
    CODES,
    PRIMITIVE_CELL_CHOICES,
    SITE_MIXTURE_SCHEMES,
    VASP_KSPACING_TAGS,
    CodeSection,
    InitSection,
    InputSetName,
    InputSetSection,
    KpointsMesh,
    KpointsSpacing,
    KpointsSpec,
    LinePath,
    MaxNumAtoms,
    SupercellDimension,
    SupercellMatrix,
    SupercellSpec,
    SymmetrySection,
    VaspSection,
    VelphConfig,
    VelphFileKind,
)
from phelel.velph.errors import VelphConfigError

SUPERCELL_KEYS = ("max_num_atoms", "supercell_dimension", "supercell_matrix")

# Optional fields of an input set in a velph.toml.  The other fields allowed
# in the input set (input_set_fields) are required in a velph.toml.
_OPTIONAL_INPUT_SET_FIELDS = ("incar", "incar_unset", "scheduler")

_TOP_KEYS: Mapping[VelphFileKind, tuple[str, ...]] = MappingProxyType(
    {
        "velph_toml": (
            "velph",
            "phelel",
            "phonopy",
            "phono3py",
            "vasp",
            "scheduler",
            "symmetry",
            "unitcell",
            "primitive_cell",
        ),
        "template": ("init", "phelel", "phonopy", "phono3py", "vasp", "scheduler"),
    }
)


def _init_keys() -> tuple[str, ...]:
    """Return the keys of [init], made from the fields of InitSection.

    The keys are the fields in their order, except that the field supercell is
    given by one of SUPERCELL_KEYS, which take its place.

    """
    keys: list[str] = []
    for field in dataclasses.fields(InitSection):
        if field.name == "supercell":
            keys.extend(SUPERCELL_KEYS)
        else:
            keys.append(field.name)
    return tuple(keys)


# Keys of [init], which are the click parameter names of the velph init
# options that [init] can give.
INIT_KEYS = _init_keys()


#
# Reading
#
def load_toml(filename: str | os.PathLike) -> dict[str, Any]:
    """Read a TOML file into a dict.

    VelphConfigError is raised when the file is not valid TOML.

    """
    with open(filename, "rb") as f:
        return _load_toml(f, str(filename))


def load_velph_toml(filename: str | os.PathLike) -> VelphConfig:
    """Read a velph.toml."""
    return parse_velph_toml(load_toml(filename), name=str(filename))


def load_template(source: str | os.PathLike | IO[bytes]) -> VelphConfig:
    """Read a velph init template from a file name or a binary file object."""
    if isinstance(source, (str, os.PathLike)):
        data = load_toml(source)
        name = str(source)
    else:
        name = getattr(source, "name", "template")
        data = _load_toml(source, name)
    return parse_template(data, name=name)


def parse_template(data: Mapping[str, Any], name: str = "template") -> VelphConfig:
    """Return VelphConfig of a velph init template parsed by tomllib.

    name is the name of the template used in error messages.

    """
    return _parse_config(data, "template", name)


def parse_velph_toml(data: Mapping[str, Any], name: str = "velph.toml") -> VelphConfig:
    """Return VelphConfig of a velph.toml parsed by tomllib.

    name is the name of the file used in error messages.

    """
    return _parse_config(data, "velph_toml", name)


def _parse_config(
    data: Mapping[str, Any], file_kind: VelphFileKind, name: str
) -> VelphConfig:
    """Return VelphConfig of data, with the name of the file in an error."""
    check_old_forms(data, file_kind, name)
    try:
        return _parse_sections(data, file_kind)
    except VelphConfigError as e:
        raise VelphConfigError(f'"{name}": {e}') from e


def _parse_sections(data: Mapping[str, Any], file_kind: VelphFileKind) -> VelphConfig:
    _check_keys(data, _TOP_KEYS[file_kind], "", top=True)

    version = None
    if "velph" in data:
        velph = _table(data["velph"], "velph")
        _check_keys(velph, ("version",), "velph")
        if "version" in velph:
            version = _str(velph["version"], "velph", "version")

    config = VelphConfig(
        version=version,
        init=parse_init(data["init"]) if "init" in data else None,
        phelel=_parse_code(data, "phelel", file_kind),
        phonopy=_parse_code(data, "phonopy", file_kind),
        phono3py=_parse_code(data, "phono3py", file_kind),
        vasp=_parse_vasp(data["vasp"], file_kind) if "vasp" in data else None,
        scheduler=_parse_scheduler(data["scheduler"], "scheduler")
        if "scheduler" in data
        else None,
        symmetry=_parse_symmetry(data["symmetry"]) if "symmetry" in data else None,
        unitcell=_parse_cell(data["unitcell"], "unitcell")
        if "unitcell" in data
        else None,
        primitive_cell=_parse_cell(data["primitive_cell"], "primitive_cell")
        if "primitive_cell" in data
        else None,
    )
    if file_kind == "velph_toml":
        _check_required(config)
    return config


def _load_toml(f: IO[bytes], name: str) -> dict[str, Any]:
    try:
        return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise VelphConfigError(f'Error in reading "{name}": {e}') from e


def _check_required(config: VelphConfig) -> None:
    if config.version is None:
        _fail("velph", '"version" is required.')
    for key in ("unitcell", "primitive_cell"):
        if getattr(config, key) is None:
            _fail(key, "This section is required.")
    for code in CODES:
        section = getattr(config, code)
        if section is not None and section.supercell is None:
            _fail(code, '"supercell_dimension" or "supercell_matrix" is required.')
    if config.vasp is not None and config.vasp.input_sets is not None:
        for set_name, input_set in config.vasp.input_sets.items():
            for field in input_set_fields(set_name):
                if field in _OPTIONAL_INPUT_SET_FIELDS:
                    continue
                if getattr(input_set, field) is None:
                    if field == "cell":
                        _fail(f"vasp.{set_name}", '"cell" is required.')
                    _fail(f"vasp.{set_name}.{field}", "This block is required.")


def parse_init(value: Any) -> InitSection:
    """Return InitSection of [init] of a template.

    The keys of [init] are the click parameter names of the velph init
    options, so velph init reads the options given on the command line by
    this function, too.

    """
    where = "init"
    table = _table(value, where)
    _check_keys(table, INIT_KEYS, where)
    codes = None
    if "codes" in table:
        codes = tuple(
            dict.fromkeys(
                _choice(code, CODES, where, "codes")
                for code in _list(table["codes"], where, "codes")
            )
        )
    return InitSection(
        codes=codes,  # type: ignore[arg-type]
        supercell=_parse_supercell(table, where, "template"),
        amplitude=_opt(table, "amplitude", _float, where),
        diagonal=_opt(table, "diagonal", _bool, where),
        plusminus=_opt(table, "plusminus", _plusminus, where),
        nosym=_opt(table, "nosym", _bool, where),
        cell_for_relax=_opt_choice(table, "cell_for_relax", CELL_CHOICES, where),
        cell_for_nac=_opt_choice(table, "cell_for_nac", CELL_CHOICES, where),
        tolerance=_opt(table, "tolerance", _float, where),
        symmetrize_cell=_opt(table, "symmetrize_cell", _bool, where),
        find_primitive=_opt(table, "find_primitive", _bool, where),
        primitive_cell_choice=_opt_choice(
            table, "primitive_cell_choice", PRIMITIVE_CELL_CHOICES, where
        ),
        use_grg=_opt(table, "use_grg", _bool, where),
        kspacing=_opt(table, "kspacing", _float, where),
        kspacing_dense=_opt(table, "kspacing_dense", _float, where),
        magmom=_opt(table, "magmom", _str, where),
        site_mixture=_opt(table, "site_mixture", _str, where),
        split_site_mixture=_opt(table, "split_site_mixture", _bool, where),
    )


def _parse_code(
    data: Mapping[str, Any], code: str, file_kind: VelphFileKind
) -> CodeSection | None:
    if code not in data:
        return None
    table = _table(data[code], code)
    _check_keys(table, SUPERCELL_KEYS + CODE_KEYS[code], code)  # type: ignore[index]
    return CodeSection(
        supercell=_parse_supercell(table, code, file_kind),
        amplitude=_opt(table, "amplitude", _float, code),
        diagonal=_opt(table, "diagonal", _bool, code),
        plusminus=_opt(table, "plusminus", _plusminus, code),
        nosym=_opt(table, "nosym", _bool, code),
        number_of_snapshots=_opt(table, "number_of_snapshots", _int, code),
        site_mixture_scheme=_opt_choice(
            table, "site_mixture_scheme", SITE_MIXTURE_SCHEMES, code
        ),
    )


def _parse_supercell(
    table: Mapping[str, Any], where: str, file_kind: VelphFileKind
) -> SupercellSpec | None:
    given = [key for key in SUPERCELL_KEYS if key in table]
    if len(given) > 1:
        _fail(where, f"Give only one of {_quoted(SUPERCELL_KEYS)}.")
    if not given:
        return None
    key = given[0]
    if key == "max_num_atoms":
        if file_kind == "velph_toml":
            _template_only_error(where, key)
        return MaxNumAtoms(max_num_atoms=_int(table[key], where, key))
    if key == "supercell_dimension":
        return SupercellDimension(
            supercell_dimension=_int_vector3(table[key], where, key)
        )
    return SupercellMatrix(
        supercell_matrix=_int_matrix3(table[key], where, key, flat=True)
    )


def _parse_vasp(value: Any, file_kind: VelphFileKind) -> VaspSection:
    table = _table(value, "vasp")
    _check_keys(table, ("incar", *INPUT_SETS), "vasp")
    if "incar" in table and file_kind == "velph_toml":
        _fail(
            "vasp.incar",
            "This table cannot be given in velph.toml.  Each input set of "
            "velph.toml holds all of its INCAR tags in [vasp.<input set>.incar].  "
            "Write the tags there, or give [vasp.incar] in a velph init template "
            'and run "velph init" again.',
        )
    incar = (
        _incar(table["incar"], "vasp.incar", file_kind) if "incar" in table else None
    )
    # The keys other than "incar" are names of input sets (_check_keys).
    input_sets = {
        cast(InputSetName, set_name): _parse_input_set(
            input_set_value, cast(InputSetName, set_name), file_kind
        )
        for set_name, input_set_value in table.items()
        if set_name != "incar"
    }
    return VaspSection(incar=incar, input_sets=input_sets or None)


def _parse_input_set(
    value: Any, set_name: InputSetName, file_kind: VelphFileKind
) -> InputSetSection:
    where = f"vasp.{set_name}"
    table = _table(value, where)
    _check_keys(table, input_set_fields(set_name), where)
    incar_unset = None
    if "incar_unset" in table and file_kind == "velph_toml":
        _fail(
            where,
            '"incar_unset" cannot be given in velph.toml.  It removes tags of '
            "[vasp.incar], which only a velph init template has.  Remove the tags "
            f'from [{where}.incar], or give "incar_unset" in a velph init template '
            'and run "velph init" again.',
        )
    if "incar_unset" in table:
        incar_unset = tuple(
            _str(tag, where, "incar_unset").lower()
            for tag in _list(table["incar_unset"], where, "incar_unset")
        )
    return InputSetSection(
        cell=_opt_choice(table, "cell", CELL_CHOICES, where),
        incar=_incar(table["incar"], f"{where}.incar", file_kind)
        if "incar" in table
        else None,
        incar_unset=incar_unset,
        kpoints=_kpoints(table["kpoints"], f"{where}.kpoints", file_kind)
        if "kpoints" in table
        else None,
        kpoints_dense=_kpoints(
            table["kpoints_dense"], f"{where}.kpoints_dense", file_kind
        )
        if "kpoints_dense" in table
        else None,
        kpoints_opt=_line_path(table["kpoints_opt"], f"{where}.kpoints_opt")
        if "kpoints_opt" in table
        else None,
        qpoints=_line_path(table["qpoints"], f"{where}.qpoints")
        if "qpoints" in table
        else None,
        scheduler=_parse_scheduler(table["scheduler"], f"{where}.scheduler")
        if "scheduler" in table
        else None,
    )


def _incar(value: Any, where: str, file_kind: VelphFileKind) -> dict[str, Any]:
    """Return the INCAR tags of a table, with the tag names in lower case."""
    table = _table(value, where)
    incar: dict[str, Any] = {}
    for key, tag_value in table.items():
        tag = key.lower()
        if tag in incar:
            _fail(where, f'INCAR tag "{tag}" is given twice in different cases.')
        if tag in VASP_KSPACING_TAGS:
            _fail(
                where,
                f'INCAR tag "{key}" cannot be used in velph, because velph '
                "writes the k-point files.  Give kspacing in a k-point block.",
            )
        if isinstance(tag_value, Mapping):
            if file_kind == "velph_toml":
                _fail(where, f'"{key}" has a table as value.  Remove "{key}".')
            _fail(
                where,
                f'"{key}" has a table as value.  A section does not inherit the '
                'tags of [vasp.incar] listed in its "incar_unset", for example '
                f'incar_unset = ["{tag}"] in [vasp.nac].',
            )
        incar[tag] = tag_value
    return incar


def _kpoints(value: Any, where: str, file_kind: VelphFileKind) -> KpointsSpec:
    table = _table(value, where)
    _check_keys(table, ("mesh", "kspacing", "shift"), where)
    shift = _float_vector3(table["shift"], where, "shift") if "shift" in table else None
    if "mesh" in table and "kspacing" in table:
        _fail(where, 'Give either "mesh" or "kspacing", not both.')
    if "kspacing" in table:
        if file_kind == "velph_toml":
            _template_only_error(where, "kspacing")
        return KpointsSpacing(
            kspacing=_float(table["kspacing"], where, "kspacing"), shift=shift
        )
    if "mesh" not in table:
        _fail(where, 'Give "mesh" or "kspacing".')
    mesh_value = table["mesh"]
    mesh: Any
    if _is_sequence(mesh_value) and mesh_value and _is_sequence(mesh_value[0]):
        mesh = _int_matrix3(mesh_value, where, "mesh")
    else:
        mesh = _int_vector3(mesh_value, where, "mesh")
    return KpointsMesh(mesh=mesh, shift=shift)


def _line_path(value: Any, where: str) -> LinePath:
    table = _table(value, where)
    _check_keys(table, ("line", "path", "label"), where)
    if "line" not in table:
        _fail(where, '"line" is required.')
    path = None
    if "path" in table:
        path = tuple(
            (
                _float_vector3(_pair(segment, where, "path")[0], where, "path"),
                _float_vector3(_pair(segment, where, "path")[1], where, "path"),
            )
            for segment in _list(table["path"], where, "path")
        )
    label = None
    if "label" in table:
        label = tuple(
            (
                _str(_pair(pair, where, "label")[0], where, "label"),
                _str(_pair(pair, where, "label")[1], where, "label"),
            )
            for pair in _list(table["label"], where, "label")
        )
    return LinePath(line=_int(table["line"], where, "line"), path=path, label=label)


def _parse_symmetry(value: Any) -> SymmetrySection:
    where = "symmetry"
    table = _table(value, where)
    _check_keys(
        table, ("tolerance", "spacegroup_type", "uni_number", "primitive_matrix"), where
    )
    primitive_matrix = None
    if "primitive_matrix" in table:
        rows = _list(table["primitive_matrix"], where, "primitive_matrix")
        if len(rows) != 3:
            _fail(where, '"primitive_matrix" has to be a 3x3 matrix.')
        primitive_matrix = tuple(
            _float_vector3(row, where, "primitive_matrix") for row in rows
        )
    return SymmetrySection(
        tolerance=_opt(table, "tolerance", _float, where),
        spacegroup_type=_opt(table, "spacegroup_type", _str, where),
        uni_number=_opt(table, "uni_number", _int, where),
        primitive_matrix=primitive_matrix,  # type: ignore[arg-type]
    )


def _parse_cell(value: Any, where: str) -> dict[str, Any]:
    """Return the cell dict after checking that phonopy reads it."""
    table = _table(value, where)
    try:
        cell = parse_cell_dict(cast(Any, dict(table)))
    except (TypeError, ValueError, KeyError) as e:
        _fail(where, f"Invalid crystal structure: {e}")
    if cell is None:
        _fail(where, "Invalid crystal structure: no points.")
    return dict(table)


def _free_table(value: Any, where: str) -> dict[str, Any]:
    table = _table(value, where)
    for key, item in table.items():
        if isinstance(item, Mapping):
            _fail(where, f'"{key}" has a table as value.')
    return dict(table)


def _parse_scheduler(value: Any, where: str) -> dict[str, Any]:
    """Return the scheduler settings of [scheduler] or [vasp.<input set>.scheduler].

    The keys are free, except that scheduler_template has to be a string and
    that scheduler_name and custom_template, which are removed, are an error.

    """
    table = _free_table(value, where)
    if "scheduler_name" in table:
        _fail(where, SCHEDULER_NAME_REMOVED + "  " + DIGEST_TAIL)
    if "custom_template" in table:
        _fail(where, '"custom_template" is removed.  Use "scheduler_template".')
    if "scheduler_template" in table:
        _str(table["scheduler_template"], where, "scheduler_template")
    return table


#
# Value checks
#
def _fail(where: str, message: str) -> NoReturn:
    if where:
        raise VelphConfigError(f"[{where}] {message}")
    raise VelphConfigError(message)


def _template_only_error(where: str, key: str) -> NoReturn:
    _fail(
        where,
        f'"{key}" cannot be given in velph.toml.  velph.toml holds the supercell '
        "matrices and k-point meshes that velph init computes.  Give "
        f'"{key}" in a velph init template and run "velph init" again.',
    )


def _quoted(keys: Sequence[str]) -> str:
    return ", ".join(f'"{key}"' for key in keys)


def _table(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(where, "This has to be a table.")
    return value


def _check_keys(
    table: Mapping[str, Any], allowed: Sequence[str], where: str, top: bool = False
) -> None:
    for key in table:
        if key not in allowed:
            if top:
                _fail("", f"Unknown section [{key}].  Allowed: {_quoted(allowed)}.")
            _fail(where, f'Unknown key "{key}".  Allowed: {_quoted(allowed)}.')


def _opt(table: Mapping[str, Any], key: str, check: Any, where: str) -> Any:
    if key not in table:
        return None
    return check(table[key], where, key)


def _opt_choice(
    table: Mapping[str, Any], key: str, choices: Sequence[str], where: str
) -> Any:
    if key not in table:
        return None
    return _choice(table[key], choices, where, key)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _list(value: Any, where: str, key: str) -> Sequence[Any]:
    if not _is_sequence(value):
        _fail(where, f'"{key}" has to be an array.')
    return value


def _pair(value: Any, where: str, key: str) -> Sequence[Any]:
    if not _is_sequence(value) or len(value) != 2:
        _fail(where, f'Each element of "{key}" has to be a pair.')
    return value


def _int(value: Any, where: str, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, f'"{key}" has to be an integer, not {value!r}.')
    return value


def _float(value: Any, where: str, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, f'"{key}" has to be a number, not {value!r}.')
    return float(value)


def _bool(value: Any, where: str, key: str) -> bool:
    if not isinstance(value, bool):
        _fail(where, f'"{key}" has to be true or false, not {value!r}.')
    return value


def _str(value: Any, where: str, key: str) -> str:
    if not isinstance(value, str):
        _fail(where, f'"{key}" has to be a string, not {value!r}.')
    return value


def _choice(value: Any, choices: Sequence[str], where: str, key: str) -> str:
    """Return value in lower case if it is one of choices.

    Otherwise VelphConfigError is raised.

    """
    if isinstance(value, str) and value.lower() in choices:
        return value.lower()
    _fail(where, f'"{key}" has to be one of {_quoted(choices)}, not {value!r}.')


def _plusminus(value: Any, where: str, key: str) -> bool | Literal["auto"]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    _fail(where, f'"{key}" has to be true, false or "auto", not {value!r}.')


def _int_vector3(value: Any, where: str, key: str) -> tuple[int, int, int]:
    if not _is_sequence(value) or len(value) != 3:
        _fail(where, f'"{key}" has to be three integers, not {value!r}.')
    return tuple(_int(v, where, key) for v in value)  # type: ignore[return-value]


def _int_matrix3(value: Any, where: str, key: str, flat: bool = False) -> Any:
    """Return a 3x3 matrix of integers.

    With flat=True, a list of nine integers is also accepted and is read row
    by row.

    """
    if flat and _is_sequence(value) and len(value) == 9:
        if all(not _is_sequence(v) for v in value):
            value = [value[0:3], value[3:6], value[6:9]]
    if not _is_sequence(value) or len(value) != 3:
        _fail(where, f'"{key}" has to be a 3x3 matrix of integers, not {value!r}.')
    return tuple(_int_vector3(row, where, key) for row in value)


def _float_vector3(value: Any, where: str, key: str) -> tuple[float, float, float]:
    if not _is_sequence(value) or len(value) != 3:
        _fail(where, f'"{key}" has to be three numbers, not {value!r}.')
    return tuple(_float(v, where, key) for v in value)  # type: ignore[return-value]
