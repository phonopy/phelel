"""Writer of velph.toml.

write_velph_toml writes VelphConfig as a velph.toml, and dumps_config returns
its TOML text.  The sections are written in a fixed order, and a matrix or a
long list is written over several lines, so that velph.toml can be read and
edited by hand.

"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from typing import Any

import tomli_w

from phelel.velph.config.io import parse_velph_toml
from phelel.velph.config.schema import (
    CODES,
    KpointsMesh,
    KpointsSpacing,
    LinePath,
    MaxNumAtoms,
    SupercellDimension,
    SupercellMatrix,
    VelphConfig,
)


def write_velph_toml(config: VelphConfig, filename: str | os.PathLike) -> None:
    """Write a velph.toml after checking that it is read back as velph.toml."""
    text = dumps_config(config)
    parse_velph_toml(tomllib.loads(text), name=str(filename))
    with open(filename, "w") as w:
        w.write(text)


def dumps_config(config: VelphConfig) -> str:
    """Return the TOML text of VelphConfig."""
    lines: list[str] = []
    _emit_table(lines, [], _config_to_root_table(config))
    return "\n".join(lines) + "\n"


def _config_to_root_table(config: VelphConfig) -> dict[str, Any]:
    """Return the dict of the root table of the TOML text of config.

    The root table is the whole TOML text (see _emit_table).  Its keys are
    the sections of velph.toml, in the order below.

        {"velph": {"version": ...},
         "init": {...},
         "phelel": {...}, "phonopy": {...}, "phono3py": {...},
         "vasp": {"incar": {...}, "relax": {...}, "nac": {...}, ...},
         "scheduler": {...},
         "symmetry": {...},
         "unitcell": {...},
         "primitive_cell": {...}}

    _emit_table keeps the order of the keys, so the sections are written in
    this order.  A section that is None in config is not put in the dict,
    and the TOML text does not have it.

    The sections held as dataclasses in config, such as [phelel], the input
    sets of [vasp] and [symmetry], are made into dicts by _section_to_table.
    The fields set to None are not put in the dicts.  The keys of the
    supercell are put in the table of the code, for example
    supercell_dimension in [phelel].  A k-point block is put as a table in
    the input set, for example [vasp.relax.kpoints].

    The sections held as dicts in config, [vasp.incar], [scheduler],
    [unitcell] and [primitive_cell], are copied.  [vasp.incar] is in a
    template only, because resolve_incar removes it before velph.toml is
    written.  The input sets are in the order of config.vasp.input_sets.  In
    a config made by velph init, this is the order of INPUT_SETS.

    """
    root: dict[str, Any] = {}
    if config.version is not None:
        root["velph"] = {"version": config.version}
    if config.init is not None:
        root["init"] = _section_to_table(config.init)
    for code in CODES:
        section = getattr(config, code)
        if section is not None:
            root[code] = _section_to_table(section)
    if config.vasp is not None:
        vasp: dict[str, Any] = {}
        if config.vasp.incar is not None:
            vasp["incar"] = dict(config.vasp.incar)
        for set_name, input_set in (config.vasp.input_sets or {}).items():
            vasp[set_name] = _section_to_table(input_set)
        root["vasp"] = vasp
    if config.scheduler is not None:
        root["scheduler"] = dict(config.scheduler)
    if config.symmetry is not None:
        root["symmetry"] = _section_to_table(config.symmetry)
    for key in ("unitcell", "primitive_cell"):
        cell = getattr(config, key)
        if cell is not None:
            root[key] = dict(cell)
    return root


def _section_to_table(section: Any) -> dict[str, Any]:
    """Return the dict of the TOML table of a section.

    section is a dataclass of the schema, such as CodeSection of [phelel] or
    InputSetSection of [vasp.relax].  Each field that is not None is put in
    the dict, and the fields set to None are not.

    The fields below are dataclasses themselves, and this function calls
    itself on them.

    - The supercell, such as SupercellDimension in the field supercell of
      CodeSection.  Its dict has one key, such as supercell_dimension, and
      that key is put in the dict of the section without the field name
      supercell.  So the line supercell_dimension = [2, 2, 2] is written
      in [phelel].
    - A k-point block, such as KpointsMesh in the field kpoints of
      InputSetSection.  Its dict is put in the dict of the section under
      the field name, and it is written as the table [vasp.relax.kpoints].

    The fields of these dataclasses are not dataclasses, so the calls on
    them do not call this function again.

    """
    table: dict[str, Any] = {}
    for name, value in vars(section).items():
        if value is None:
            continue
        if isinstance(value, (MaxNumAtoms, SupercellDimension, SupercellMatrix)):
            table.update(_section_to_table(value))
        elif isinstance(value, (KpointsMesh, KpointsSpacing, LinePath)):
            table[name] = _section_to_table(value)
        else:
            table[name] = value
    return table


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_LINE_WIDTH = 80


def _emit_table(lines: list[str], path: list[str], table: Mapping[str, Any]) -> None:
    """Append the TOML lines of table to lines.

    In TOML, a table is a set of key = value lines under a header, such as
    [phelel] in the example below.  A table can hold another table.  The
    header of the inner table joins the keys with dots, such as
    [vasp.relax.incar].  An array of tables is a list of tables under one
    key.  Each element of the list starts with the key in double brackets,
    such as [[unitcell.points]].

        [phelel]
        supercell_dimension = [2, 2, 2]

        [vasp.relax.incar]
        encut = 500

        [[unitcell.points]]
        symbol = "Na"

        [[unitcell.points]]
        symbol = "Cl"

    tomllib reads a table as a dict, and an array of tables as a list of
    dicts.  The TOML text above is read as

        {"phelel": {"supercell_dimension": [2, 2, 2]},
         "vasp": {"relax": {"incar": {"encut": 500}}},
         "unitcell": {"points": [{"symbol": "Na"}, {"symbol": "Cl"}]}}

    This function writes a dict of this form as TOML text.  The argument
    table is the dict of one TOML table.  The function writes the header
    and the key = value lines of table, and then calls itself for each table
    inside table.  The call for an inner table writes all of that table,
    including the tables inside it, before it returns.  For example, the
    dict

        root = {"vasp": {"relax": {"cell": "unitcell",
                                   "incar": {"encut": 500}},
                         "nac": {"cell": "primitive"}}}

    is written by the calls below.  Each call is indented under the call
    that makes it, and the lines that it writes are shown under it.

        _emit_table(lines, [], root)
          _emit_table(lines, ["vasp"], root["vasp"])
            _emit_table(lines, ["vasp", "relax"], root["vasp"]["relax"])
              [vasp.relax]
              cell = "unitcell"
              _emit_table(lines, ["vasp", "relax", "incar"], ...)
                [vasp.relax.incar]
                encut = 500
            _emit_table(lines, ["vasp", "nac"], root["vasp"]["nac"])
              [vasp.nac]
              cell = "primitive"

    The calls for root and for root["vasp"] write no lines, because the
    root table has no header and [vasp] holds only tables (see Notes).  The
    lines are appended to lines in the order of the calls, so
    [vasp.relax.incar] is written before [vasp.nac].

    Parameters
    ----------
    lines : list[str]
        Lines of the TOML text written so far.  The lines of table are
        appended to this list.
    path : list[str]
        Keys of table in the whole TOML text.  When table is the table
        [vasp.relax], path is ["vasp", "relax"], and table is
        root["vasp"]["relax"], where root is the dict of the whole TOML
        text.  The header of table is made from path, by joining the keys
        with dots, because table does not hold its own name.  When table is
        root, path is [].  TOML calls root the root table.  The root table
        has no name, so no header is written for it.
    table : Mapping[str, Any]
        Table to write.  A value that is a Mapping is written as a table,
        and a non-empty list or tuple of Mappings is written as an array of
        tables.  Any other value is written as a key = value line.

    Notes
    -----
    The contents of table are written in the order below.  Within each
    item, the order of the keys in table is kept.

    1. The header [path] and the key = value lines of table.
    2. The tables in table.  Each one is written by calling this function
       with path + [key] and table[key], so that path in the call is the
       keys of table[key].
    3. The arrays of tables in table.  Each element is written under the
       header [[path.key]], followed by its number in a comment, for
       example [[unitcell.points]]  # 1.

    The key = value lines are written before the tables.  In TOML, a
    key = value line that follows the header of a table belongs to that
    table.  For example, a line written after [vasp.relax.incar] is read as
    an INCAR tag of the relax input set.

    The header [path] is written when table has key = value lines, and
    when table is empty.  An empty table is written as its header alone,
    and TOML reads that header as an empty table.  The root table has no
    header.  A table that holds only tables, such as [vasp], is written
    without its own header.  TOML makes the table [vasp] from the header
    [vasp.relax] of the table inside it.

    The elements of an array of tables are written by _key_value_lines.
    A table inside an element is written as an invalid line such as
    "extra = a = 1", and write_velph_toml stops with TOMLDecodeError when it
    reads the text back.  The arrays of tables in velph.toml are the points
    of [unitcell] and [primitive_cell], made by get_cell_dict of phonopy.
    Each point holds only values, such as symbol, coordinates and mass.

    """
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
    """Return the TOML text of a key.

    A key made of letters, digits, "_" and "-" is a bare key of TOML, and it
    is written as it is, for example encut.  Any other key has to be written
    in quotes, for example "a.b", because a.b without quotes is the key b in
    the table a.  The quotes and the escapes in such a key are written by
    tomli_w.  tomli_w writes {key: 0} as the line <key> = 0, and the text
    before the last " = " of that line is returned.  The last " = " is used,
    because the key itself can hold " = ", as in "a = b".

    """
    if _BARE_KEY.match(key):
        return key
    return tomli_w.dumps({key: 0}).rsplit(" = ", 1)[0]


def _dotted(path: list[str]) -> str:
    """Return the name of a table in its header, such as vasp.relax.

    path is the keys of the table (see _emit_table).  Each key is written by
    _key, and the keys are joined with dots.

    """
    return ".".join(_key(key) for key in path)


def _key_value_lines(key: str, value: Any) -> list[str]:
    """Return the TOML lines of key = value.

    Parameters
    ----------
    key : str
        Key.  A key made of letters, digits, "_" and "-" is written as it
        is.  Any other key is written in quotes by tomli_w, for example
        "a b" = 1.
    value : Any
        Value that is not a table.  It is a scalar, such as a number, a
        string or a bool, or a list or tuple of values.

    Returns
    -------
    list[str]
        Lines of the TOML text, in one of the forms below.

    Notes
    -----
    A scalar is written by tomli_w on one line, for example encut = 500.  A
    string that has line breaks, such as scheduler_template, is written as
    a multi-line string between triple quotes, and it takes more than one
    line.

    A list or tuple whose elements are all lists or tuples, such as a
    supercell matrix or the basis vectors of a cell, is written with one
    element per line.

        supercell_matrix = [
          [0, 2, 2],
          [2, 0, 2],
          [2, 2, 0],
        ]

    Any other list or tuple is written on one line when the line has at
    most _LINE_WIDTH characters, for example mesh = [6, 6, 6].  An empty
    list is written as key = [].  A longer list is written over several
    lines, and each line holds as many elements as fit in _LINE_WIDTH
    characters.

        magmom = [
          1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
          1.5, 1.5, 1.5,
        ]

    """
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
        # A line without an element is not written, also when one element
        # is longer than _LINE_WIDTH.
        if line.strip() and len(line) + len(item) > _LINE_WIDTH:
            lines.append(line)
            line = " "
        line += item
    lines += [line, "]"]
    return lines


def _inline(value: Any) -> str:
    """Return the TOML text of value on one line.

    A list or tuple is written in brackets, and each element is written by
    calling this function on it.  A list inside a list is therefore written
    in brackets inside the brackets, for example [[0, 2, 2], [2, 0, 2]].  A
    scalar is written by _scalar.

    """
    if _is_sequence(value):
        return "[" + ", ".join(_inline(v) for v in value) + "]"
    return _scalar(value)


def _scalar(value: Any) -> str:
    """Return the TOML text of a scalar, using tomli_w."""
    text = tomli_w.dumps({"x": value}, multiline_strings=True)
    return text[len("x = ") :].rstrip("\n")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))
