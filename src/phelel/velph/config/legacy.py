"""Reader of velph.toml written by velph earlier than 0.15.

load_velph_toml_any_version reads a velph.toml of the new format or of velph
0.14.  The velph commands read velph.toml through it, so that they also read
the velph.toml files written by velph 0.14.  A file of velph 0.14 is converted
to the new format and read into VelphConfig, so that the generate commands
write the same VASP input files as those of velph 0.14.

velph init of every version up to 0.14 writes [phelel] version, so a
velph.toml of velph earlier than 0.14 is read in the same way as one of velph
0.14.  The conversion is written for the format of velph 0.14, and it has not
been checked on files of earlier versions.  A key that the new format does not
have is reported by parse_velph_toml as an unknown key.  A key whose meaning
differs from that in velph 0.14 is read with the meaning of velph 0.14, and no
error is raised.

The conversion is as follows.

- [phelel] version becomes [velph] version.
- [phelel] fft_mesh, phonon_supercell_dimension and phonon_supercell_matrix,
  and [vasp.phelel.phonon] and [vasp.phono3py.phonon] are removed.
- [vasp.incar] is removed.  The generate commands of velph 0.14 do not read
  it, because velph init writes all INCAR tags in each input set.
- [vasp.el_bands.bands] and [vasp.el_bands.dos] become [vasp.el_bands] and
  [vasp.el_dos].
- [vasp.relax] and [vasp.nac] without "cell" get the cell that the generate
  commands of velph 0.14 use: the unit cell for relax and the primitive cell
  for nac.
- kspacing in [vasp.<input set>.kpoints] and [vasp.<input set>.kpoints_dense]
  is converted to mesh in the same way as the generate commands of velph 0.14
  do, that is, with use_grg=True, so the mesh is a generalized regular grid
  when the cell has a single lattice point.

convert_legacy_velph_toml writes the converted velph.toml in a new file.  The
command "velph convert OLD NEW" runs it.  [velph] version of the new file is the
version of velph that writes it, as in the velph.toml written by "velph init".

"""

from __future__ import annotations

import copy
import dataclasses
import os
import pathlib
from collections.abc import Iterator, Mapping
from typing import Any

from phonopy.structure.atoms import PhonopyAtoms, parse_cell_dict
from phonopy.structure.cells import (
    get_supercell,
    merge_weighted_species,
    shape_supercell_matrix,
)

from phelel import __version__
from phelel.velph.config.io import load_toml, parse_velph_toml
from phelel.velph.config.schema import CODES, VelphConfig
from phelel.velph.config.toml_writer import write_velph_toml
from phelel.velph.errors import VelphConfigError, VelphError
from phelel.velph.utils.mesh import kspacing_to_mesh

# Tolerance with which Phelel, Phonopy and Phono3py make the supercell.
# "velph phelel init", "velph phonopy init" and "velph phono3py init" of velph
# 0.14 do not give symprec to them, so their default is used.
_SUPERCELL_SYMPREC = 1e-5

# Keys of [phelel], [phonopy] and [phono3py] that the new format does not have.
_REMOVED_CODE_KEYS = (
    "fft_mesh",
    "phonon_supercell_dimension",
    "phonon_supercell_matrix",
)

# Cells that the generate commands of velph 0.14 use for relax and nac when
# "cell" is not given (DefaultCellChoices in velph/cli/utils.py of 0.14).
_DEFAULT_CELLS = {"relax": "unitcell", "nac": "primitive"}

# Input sets whose k-point meshes the generate commands of velph 0.14 compute
# on the primitive cell.  ph_bands is not included, because "velph ph_bands
# generate" does not convert kspacing.
_PRIMITIVE_CELL_INPUT_SETS = (
    "selfenergy",
    "transport",
    "ph_selfenergy",
    "el_bands",
    "el_dos",
)


def load_velph_toml_any_version(filename: str | os.PathLike) -> VelphConfig:
    """Read a velph.toml of the new format or of velph 0.14 into VelphConfig.

    A velph.toml with [velph] version is read in the new format.  Otherwise it
    is read as a velph.toml of velph 0.14.

    """
    data = load_toml(filename)
    velph = data.get("velph")
    if isinstance(velph, Mapping) and "version" in velph:
        return parse_velph_toml(data, name=str(filename))
    return parse_legacy_velph_toml(data, name=str(filename))


def parse_legacy_velph_toml(
    data: Mapping[str, Any], name: str = "velph.toml"
) -> VelphConfig:
    """Convert the contents of a velph.toml of velph 0.14 and read them.

    data is not modified.  An error in the converted contents is reported by
    parse_velph_toml with the section names of the new format, for example
    [vasp.el_dos] for [vasp.el_bands.dos].

    velph init of every version up to 0.14 writes [phelel] version, and
    VelphConfigError is raised when data does not have it.  data with [phelel]
    version of velph earlier than 0.14 is converted as a velph.toml of velph
    0.14 (see the docstring of this module).

    """
    phelel = data.get("phelel")
    if not (isinstance(phelel, Mapping) and "version" in phelel):
        raise VelphConfigError(
            f'"{name}" has no version.\n'
            "velph chooses how to read velph.toml by its version.\n"
            "velph 0.15 and later write [velph] version.\n"
            "velph 0.14 and earlier write [phelel] version.\n"
            'Run "velph init" to make a new velph.toml.'
        )
    converted = copy.deepcopy(dict(data))
    _convert_code_sections(converted)
    vasp = converted.get("vasp")
    if isinstance(vasp, dict):
        _convert_vasp_section(vasp)
        _convert_kspacing(converted)
    return parse_velph_toml(converted, name=name)


def convert_legacy_velph_toml(
    old_filename: str | os.PathLike, new_filename: str | os.PathLike
) -> list[str]:
    """Write the velph.toml of velph 0.14 old_filename in the new format.

    [velph] version of new_filename is the version of this velph, not the
    version in old_filename.  The messages on what was removed and on which
    kspacing was replaced by mesh are returned.  VelphError is raised when
    new_filename exists, and VelphConfigError when old_filename is in the new
    format or cannot be converted.  Nothing is written then.

    """
    if pathlib.Path(new_filename).exists():
        raise VelphError(
            f'"{new_filename}" exists.  Give the name of a file that does not exist.'
        )
    data = load_toml(old_filename)
    velph = data.get("velph")
    if isinstance(velph, Mapping) and "version" in velph:
        raise VelphConfigError(
            f'"{old_filename}" has [velph] version, so it is in the new format.'
        )
    config = parse_legacy_velph_toml(data, name=str(old_filename))
    write_velph_toml(dataclasses.replace(config, version=__version__), new_filename)
    return _conversion_messages(data)


def _conversion_messages(data: Mapping[str, Any]) -> list[str]:
    """Return what the conversion removes from data and where it replaces kspacing.

    The section names are those of velph 0.14.

    """
    messages = []
    vasp = data.get("vasp")
    if not isinstance(vasp, Mapping):
        vasp = {}
    if "incar" in vasp:
        messages.append("[vasp.incar] was removed.")
    for code in CODES:
        section = data.get(code)
        if not isinstance(section, Mapping):
            continue
        for key in _REMOVED_CODE_KEYS:
            if key in section:
                messages.append(f"{key} of [{code}] was removed.")
    for code in ("phelel", "phono3py"):
        input_set = vasp.get(code)
        if isinstance(input_set, Mapping) and "phonon" in input_set:
            messages.append(f"[vasp.{code}.phonon] was removed.")
    for name, input_set in _legacy_input_sets(vasp):
        for block in ("kpoints", "kpoints_dense"):
            kpoints = input_set.get(block)
            if isinstance(kpoints, Mapping) and "kspacing" in kpoints:
                messages.append(
                    f"kspacing of [vasp.{name}.{block}] was replaced by mesh."
                )
    return messages


def _legacy_input_sets(
    vasp: Mapping[str, Any],
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Yield the names and tables of the input sets of [vasp] of velph 0.14."""
    for set_name, table in vasp.items():
        if set_name == "incar" or not isinstance(table, Mapping):
            continue
        if set_name == "el_bands" and {"bands", "dos"} & set(table):
            for sub_name in ("bands", "dos"):
                if isinstance(table.get(sub_name), Mapping):
                    yield f"el_bands.{sub_name}", table[sub_name]
        else:
            yield set_name, table


def _convert_code_sections(data: dict[str, Any]) -> None:
    phelel = data.get("phelel")
    if isinstance(phelel, dict) and "version" in phelel:
        data["velph"] = {"version": phelel.pop("version")}
    for code in CODES:
        section = data.get(code)
        if not isinstance(section, dict):
            continue
        for key in _REMOVED_CODE_KEYS:
            section.pop(key, None)
        # velph init of velph 0.14 writes [phelel] with only version when no
        # supercell is given.
        if not section:
            del data[code]


def _convert_vasp_section(vasp: dict[str, Any]) -> None:
    vasp.pop("incar", None)
    for code in ("phelel", "phono3py"):
        input_set = vasp.get(code)
        if isinstance(input_set, dict):
            input_set.pop("phonon", None)
    el_bands = vasp.get("el_bands")
    if isinstance(el_bands, dict) and {"bands", "dos"} & set(el_bands):
        del vasp["el_bands"]
        if "bands" in el_bands:
            vasp["el_bands"] = el_bands["bands"]
        if "dos" in el_bands:
            vasp["el_dos"] = el_bands["dos"]
    for set_name, default_cell in _DEFAULT_CELLS.items():
        input_set = vasp.get(set_name)
        if not isinstance(input_set, dict):
            continue
        cell = input_set.get("cell", default_cell)
        input_set["cell"] = cell.lower() if isinstance(cell, str) else cell


def _convert_kspacing(data: dict[str, Any]) -> None:
    """Replace kspacing in the k-point blocks by mesh.

    Each cell is made only when an input set needs it, so that a velph.toml
    without kspacing is converted without reading the cells.

    """
    for set_name, input_set in data["vasp"].items():
        if not isinstance(input_set, dict):
            continue
        for block in ("kpoints", "kpoints_dense"):
            kpoints = input_set.get(block)
            if not isinstance(kpoints, dict) or "kspacing" not in kpoints:
                continue
            cell = _cell_of_input_set(data, set_name)
            if cell is None:
                continue
            kpoints["mesh"] = kspacing_to_mesh(
                kpoints.pop("kspacing"), cell, use_grg=True
            )


def _cell_of_input_set(data: Mapping[str, Any], set_name: str) -> PhonopyAtoms | None:
    """Return the cell on which the generate commands of velph 0.14 compute a mesh.

    None is returned when the generate commands do not convert kspacing of the
    input set, or when the cell cannot be made from data.  kspacing is then left
    as it is, and parse_velph_toml reports it.

    """
    if set_name in CODES:
        return _supercell(data, set_name)
    if set_name in _DEFAULT_CELLS:
        cell_choice = data["vasp"][set_name]["cell"]
        if cell_choice == "unitcell":
            return _cell(data, "unitcell")
        if cell_choice == "primitive":
            return _cell(data, "primitive_cell")
        return None
    if set_name in _PRIMITIVE_CELL_INPUT_SETS:
        return _cell(data, "primitive_cell")
    return None


def _cell(data: Mapping[str, Any], key: str) -> PhonopyAtoms | None:
    cell_dict = data.get(key)
    if not isinstance(cell_dict, Mapping):
        return None
    return parse_cell_dict(dict(cell_dict))  # type: ignore[arg-type]


def _supercell(data: Mapping[str, Any], code: str) -> PhonopyAtoms | None:
    """Return the supercell that "velph <code> init" of velph 0.14 makes.

    With the merge scheme of site mixture, Phelel and Phonopy merge the
    co-located atoms of the unit cell into sites before they make the
    supercell, and the generate commands compute the mesh on this supercell.
    phono3py does not support site mixture, so its supercell is made from the
    unit cell as it is.

    """
    section = data.get(code)
    unitcell = _cell(data, "unitcell")
    if not isinstance(section, Mapping) or unitcell is None:
        return None
    supercell_matrix = section.get(
        "supercell_dimension", section.get("supercell_matrix")
    )
    if supercell_matrix is None:
        return None
    site_mixture_scheme = section.get("site_mixture_scheme", "merge")
    if (
        code != "phono3py"
        and site_mixture_scheme == "merge"
        and unitcell.has_weighted_species
    ):
        unitcell, _ = merge_weighted_species(unitcell, symprec=_SUPERCELL_SYMPREC)
    return get_supercell(
        unitcell,
        shape_supercell_matrix(supercell_matrix),
        symprec=_SUPERCELL_SYMPREC,
    )
