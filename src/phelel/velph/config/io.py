"""Read and write velph.toml and velph init templates.

parse_template and parse_velph_toml read a parsed TOML document into
VelphConfig, and load_template and load_velph_toml read a file.  The two
kinds of files are read in different ways:

- every field of a template is optional.  [init] is accepted, and so are
  max_num_atoms and kspacing in a k-point block, which velph init converts
  to a supercell matrix and to a mesh.
- a velph.toml has to give [velph] version, the cells and the fields that
  the generate commands read.  max_num_atoms and kspacing in a k-point
  block are errors.

An unknown key is an error in both kinds of files.

A velph.toml without [velph] version was written by velph earlier than
0.15, and reading it is an error.  A template that has keys of velph earlier
than 0.15, such as [init.options], is also an error.  The error message
lists only the changes that concern the keys found in the file.

"""

from __future__ import annotations

import os
import re
import textwrap
from collections.abc import Iterator, Mapping, Sequence
from typing import IO, Any, Literal, NoReturn, cast

import tomli
import tomli_w
from phonopy.structure.atoms import parse_cell_dict

from phelel.velph.config.schema import (
    CELL_CHOICES,
    CODE_KEYS,
    CODES,
    INPUT_SET_FIELDS,
    PRIMITIVE_CELL_CHOICES,
    SITE_MIXTURE_SCHEMES,
    VASP_KSPACING_TAGS,
    CodeSection,
    Dimension,
    InitSection,
    InputSetSection,
    KpointsSpec,
    LinePath,
    Matrix,
    MaxNumAtoms,
    Mesh,
    Spacing,
    SupercellSpec,
    SymmetrySection,
    VaspSection,
    VelphConfig,
)

VelphFileKind = Literal["template", "velph_toml"]

SUPERCELL_KEYS = ("max_num_atoms", "supercell_dimension", "supercell_matrix")

# Optional fields of an input set in a velph.toml.  The other fields allowed
# in the input set (INPUT_SET_FIELDS) are required in a velph.toml.
_OPTIONAL_INPUT_SET_FIELDS = ("incar", "incar_unset", "scheduler")

_TOP_KEYS: dict[VelphFileKind, tuple[str, ...]] = {
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

_INIT_KEYS = (
    "codes",
    *SUPERCELL_KEYS,
    "amplitude",
    "diagonal",
    "plusminus",
    "nosym",
    "cell_for_relax",
    "cell_for_nac",
    "tolerance",
    "symmetrize_cell",
    "find_primitive",
    "primitive_cell_choice",
    "use_grg",
    "kspacing",
    "kspacing_dense",
    "magmom",
    "site_mixture",
    "split_site_mixture",
)


class VelphConfigError(ValueError):
    """Error in a velph.toml or a velph init template."""


#
# Reading
#
def load_velph_toml(filename: str | os.PathLike) -> VelphConfig:
    """Read a velph.toml."""
    with open(filename, "rb") as f:
        data = _load_toml(f, str(filename))
    return parse_velph_toml(data, name=str(filename))


def load_template(source: str | os.PathLike | IO[bytes]) -> VelphConfig:
    """Read a velph init template from a file name or a binary file object."""
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            data = _load_toml(f, str(source))
        name = str(source)
    else:
        name = getattr(source, "name", "template")
        data = _load_toml(source, name)
    return parse_template(data, name=name)


def parse_template(data: Mapping[str, Any], name: str = "template") -> VelphConfig:
    """Return VelphConfig of a velph init template parsed by tomli.

    name is the name of the template used in error messages.

    """
    return _parse_config(data, "template", name)


def parse_velph_toml(data: Mapping[str, Any], name: str = "velph.toml") -> VelphConfig:
    """Return VelphConfig of a velph.toml parsed by tomli.

    name is the name of the file used in error messages.

    """
    return _parse_config(data, "velph_toml", name)


def _parse_config(
    data: Mapping[str, Any], file_kind: VelphFileKind, name: str
) -> VelphConfig:
    _check_old_forms(data, file_kind, name)
    _check_keys(data, _TOP_KEYS[file_kind], "", top=True)

    version = None
    if "velph" in data:
        velph = _table(data["velph"], "velph")
        _check_keys(velph, ("version",), "velph")
        if "version" in velph:
            version = _str(velph["version"], "velph", "version")

    config = VelphConfig(
        version=version,
        init=_parse_init(data["init"]) if "init" in data else None,
        phelel=_parse_code(data, "phelel", file_kind),
        phonopy=_parse_code(data, "phonopy", file_kind),
        phono3py=_parse_code(data, "phono3py", file_kind),
        vasp=_parse_vasp(data["vasp"], file_kind) if "vasp" in data else None,
        scheduler=_free_table(data["scheduler"], "scheduler")
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
        return tomli.load(f)
    except tomli.TOMLDecodeError as e:
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
            for field in INPUT_SET_FIELDS[set_name]:
                if field in _OPTIONAL_INPUT_SET_FIELDS:
                    continue
                if getattr(input_set, field) is None:
                    if field == "cell":
                        _fail(f"vasp.{set_name}", '"cell" is required.')
                    _fail(f"vasp.{set_name}.{field}", "This block is required.")


def _parse_init(value: Any) -> InitSection:
    where = "init"
    table = _table(value, where)
    _check_keys(table, _INIT_KEYS, where)
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
            _intent_error(where, key)
        return MaxNumAtoms(max_num_atoms=_int(table[key], where, key))
    if key == "supercell_dimension":
        return Dimension(supercell_dimension=_int_vector3(table[key], where, key))
    return Matrix(supercell_matrix=_int_matrix3(table[key], where, key, flat=True))


def _parse_vasp(value: Any, file_kind: VelphFileKind) -> VaspSection:
    table = _table(value, "vasp")
    _check_keys(table, ("incar", *INPUT_SET_FIELDS), "vasp")
    incar = _incar(table["incar"], "vasp.incar") if "incar" in table else None
    input_sets = {
        set_name: _parse_input_set(input_set_value, set_name, file_kind)
        for set_name, input_set_value in table.items()
        if set_name != "incar"
    }
    return VaspSection(incar=incar, input_sets=input_sets or None)


def _parse_input_set(
    value: Any, set_name: str, file_kind: VelphFileKind
) -> InputSetSection:
    where = f"vasp.{set_name}"
    table = _table(value, where)
    _check_keys(table, INPUT_SET_FIELDS[set_name], where)
    incar_unset = None
    if "incar_unset" in table:
        incar_unset = tuple(
            _str(tag, where, "incar_unset").lower()
            for tag in _list(table["incar_unset"], where, "incar_unset")
        )
    return InputSetSection(
        cell=_opt_choice(table, "cell", CELL_CHOICES, where),
        incar=_incar(table["incar"], f"{where}.incar") if "incar" in table else None,
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
        scheduler=_free_table(table["scheduler"], f"{where}.scheduler")
        if "scheduler" in table
        else None,
    )


def _incar(value: Any, where: str) -> dict[str, Any]:
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
            _intent_error(where, "kspacing")
        return Spacing(
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
    return Mesh(mesh=mesh, shift=shift)


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


#
# Old forms
#
_DIGEST_EL_BANDS = (
    "[vasp.el_bands.bands] and [vasp.el_bands.dos] are now [vasp.el_bands] and "
    '[vasp.el_dos], with the directories "el_bands" and "el_dos".'
)
_DIGEST_FFT_MESH = (
    '[phelel] fft_mesh is removed.  "velph phelel differentiate" computes the '
    "FFT mesh from encut and prec in the INCARs of the electron-phonon "
    "calculations, or from --encut."
)
_DIGEST_KSPACING = (
    '"velph init" converts kspacing in a k-point block to mesh, and velph.toml '
    "has only mesh."
)
_DIGEST_EMPTY_TABLE = (
    "An INCAR tag set to {} is no longer used.  A section such as [vasp.nac] "
    'does not inherit the tags of [vasp.incar] listed in its "incar_unset".'
)
_DIGEST_INIT_OPTIONS = (
    "[init.options] is now [init].  Its keys are those of the velph init "
    "command line.  supercell_dimension, amplitude and the other supercell and "
    "displacement keys in [init] apply to every code in codes."
)
_DIGEST_TAIL = 'See "Changes in velph 0.15" in the documentation for all changes.'


def _check_old_forms(
    data: Mapping[str, Any], file_kind: VelphFileKind, name: str
) -> None:
    lines = list(_old_form_lines(data, file_kind))
    if file_kind == "velph_toml":
        velph = data.get("velph")
        if isinstance(velph, Mapping) and "version" in velph:
            return
        head = [
            f'"{name}" was written by velph earlier than 0.15 and cannot be read.',
            'Re-run "velph init" to make a new velph.toml.',
        ]
    else:
        if not lines:
            return
        head = [f'"{name}" is a velph init template of velph earlier than 0.15.']
    message = head
    if lines:
        message += ["", "Changes that concern this file:", ""]
        for line in lines:
            message += textwrap.wrap(
                line, width=76, initial_indent="  - ", subsequent_indent="    "
            )
    message += ["", _DIGEST_TAIL]
    raise VelphConfigError("\n".join(message))


def _old_form_lines(data: Mapping[str, Any], file_kind: VelphFileKind) -> Iterator[str]:
    vasp = data.get("vasp")
    if not isinstance(vasp, Mapping):
        vasp = {}
    el_bands = vasp.get("el_bands")
    if isinstance(el_bands, Mapping) and ({"bands", "dos"} & set(el_bands)):
        yield _DIGEST_EL_BANDS
    phelel = data.get("phelel")
    if isinstance(phelel, Mapping) and "fft_mesh" in phelel:
        yield _DIGEST_FFT_MESH
    input_set_tables = list(_old_input_set_tables(vasp))
    if file_kind == "velph_toml":
        if any(
            isinstance(input_set_table.get(block), Mapping)
            and "kspacing" in input_set_table[block]
            for input_set_table in input_set_tables
            for block in ("kpoints", "kpoints_dense")
        ):
            yield _DIGEST_KSPACING
    else:
        incar_tables = [vasp.get("incar")] + [
            input_set_table.get("incar") for input_set_table in input_set_tables
        ]
        if any(
            isinstance(incar, Mapping) and any(value == {} for value in incar.values())
            for incar in incar_tables
        ):
            yield _DIGEST_EMPTY_TABLE
        init = data.get("init")
        if isinstance(init, Mapping) and "options" in init:
            yield _DIGEST_INIT_OPTIONS


def _old_input_set_tables(vasp: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield the tables of the input sets in [vasp].

    The tables in [vasp.el_bands] of a velph.toml of velph earlier than 0.15
    are included.

    """
    for key, table in vasp.items():
        if key == "incar" or not isinstance(table, Mapping):
            continue
        yield table
        if key == "el_bands":
            for sub_table in table.values():
                if isinstance(sub_table, Mapping):
                    yield sub_table


#
# Value checks
#
def _fail(where: str, message: str) -> NoReturn:
    if where:
        raise VelphConfigError(f"[{where}] {message}")
    raise VelphConfigError(message)


def _intent_error(where: str, key: str) -> NoReturn:
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


#
# Writing
#
def write_velph_toml(config: VelphConfig, filename: str | os.PathLike) -> None:
    """Write a velph.toml after checking that it is read back as velph.toml."""
    text = dumps_config(config)
    parse_velph_toml(tomli.loads(text), name=str(filename))
    with open(filename, "w") as w:
        w.write(text)


def dumps_config(config: VelphConfig) -> str:
    """Return the TOML text of VelphConfig."""
    lines: list[str] = []
    _emit_table(lines, [], _config_to_document(config))
    return "\n".join(lines) + "\n"


def _config_to_document(config: VelphConfig) -> dict[str, Any]:
    document: dict[str, Any] = {}
    if config.version is not None:
        document["velph"] = {"version": config.version}
    if config.init is not None:
        document["init"] = _section_to_table(config.init)
    for code in CODES:
        section = getattr(config, code)
        if section is not None:
            document[code] = _section_to_table(section)
    if config.vasp is not None:
        vasp: dict[str, Any] = {}
        if config.vasp.incar is not None:
            vasp["incar"] = dict(config.vasp.incar)
        for set_name, input_set in (config.vasp.input_sets or {}).items():
            vasp[set_name] = _section_to_table(input_set)
        document["vasp"] = vasp
    if config.scheduler is not None:
        document["scheduler"] = dict(config.scheduler)
    if config.symmetry is not None:
        document["symmetry"] = _section_to_table(config.symmetry)
    for key in ("unitcell", "primitive_cell"):
        cell = getattr(config, key)
        if cell is not None:
            document[key] = dict(cell)
    return document


def _section_to_table(section: Any) -> dict[str, Any]:
    """Return the TOML table of a section, without the fields set to None."""
    table: dict[str, Any] = {}
    for name, value in vars(section).items():
        if value is None:
            continue
        if isinstance(value, (MaxNumAtoms, Dimension, Matrix)):
            table.update(_section_to_table(value))
        elif isinstance(value, (Mesh, Spacing, LinePath)):
            table[name] = _section_to_table(value)
        else:
            table[name] = value
    return table


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_LINE_WIDTH = 80


def _emit_table(lines: list[str], path: list[str], table: Mapping[str, Any]) -> None:
    values = {k: v for k, v in table.items() if not _is_table(v) and not _is_tables(v)}
    tables = {k: v for k, v in table.items() if _is_table(v)}
    arrays = {k: v for k, v in table.items() if _is_tables(v)}
    if path and (values or not (tables or arrays)):
        _blank_line(lines)
        lines.append(f"[{_dotted(path)}]")
    for key, value in values.items():
        lines += _key_value_lines(key, value)
    for key, sub_table in tables.items():
        _emit_table(lines, path + [key], sub_table)
    for key, items in arrays.items():
        for i, item in enumerate(items):
            _blank_line(lines)
            lines.append(f"[[{_dotted(path + [key])}]]  # {i + 1}")
            for item_key, item_value in item.items():
                lines += _key_value_lines(item_key, item_value)


def _blank_line(lines: list[str]) -> None:
    if lines and lines[-1] != "":
        lines.append("")


def _is_table(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_tables(value: Any) -> bool:
    return (
        _is_sequence(value)
        and len(value) > 0
        and all(isinstance(item, Mapping) for item in value)
    )


def _key(key: str) -> str:
    if _BARE_KEY.match(key):
        return key
    return tomli_w.dumps({key: 0}).split(" = ")[0]


def _dotted(path: list[str]) -> str:
    return ".".join(_key(key) for key in path)


def _key_value_lines(key: str, value: Any) -> list[str]:
    head = f"{_key(key)} = "
    if not _is_sequence(value):
        return (head + _scalar(value)).split("\n")
    if value and all(_is_sequence(row) for row in value):
        # Write a matrix, or a list of path segments, with one row per line.
        return [head + "["] + [f"  {_inline(row)}," for row in value] + ["]"]
    inline = head + _inline(value)
    if len(inline) <= _LINE_WIDTH:
        return [inline]
    lines = [head + "["]
    line = " "
    for element in value:
        item = f" {_inline(element)},"
        if len(line) + len(item) > _LINE_WIDTH:
            lines.append(line)
            line = " "
        line += item
    lines += [line, "]"]
    return lines


def _inline(value: Any) -> str:
    if _is_sequence(value):
        return "[" + ", ".join(_inline(v) for v in value) + "]"
    return _scalar(value)


def _scalar(value: Any) -> str:
    """Return the TOML text of a scalar, using tomli_w."""
    text = tomli_w.dumps({"x": value}, multiline_strings=True)
    return text[len("x = ") :].rstrip("\n")
