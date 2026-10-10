"""Reader of velph.toml written by velph earlier than 0.15.

The generate commands read velph.toml through VelphConfig after they are
rewritten.  Until "velph init" writes the new format, the velph.toml files that
users have are in the format of velph 0.14.  load_legacy_velph_toml converts
such a file to the new format and reads it into VelphConfig, so that the
generate commands write the same VASP input files as before.

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
  do.  kspacing_to_mesh is used with its default use_grg=True, so the mesh is
  a generalized regular grid when the cell allows it.

This module is temporary.  It is removed when "velph init" writes the new
format.

"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from typing import Any

import tomli
from phonopy.structure.atoms import PhonopyAtoms, parse_cell_dict
from phonopy.structure.cells import (
    get_supercell,
    merge_weighted_species,
    shape_supercell_matrix,
)

from phelel.velph.cli.utils import kspacing_to_mesh
from phelel.velph.config.io import VelphConfigError, parse_velph_toml
from phelel.velph.config.schema import CODES, VelphConfig

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
# "cell" is not given (DefaultCellChoices in velph/cli/utils.py).
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


def load_legacy_velph_toml(filename: str | os.PathLike) -> VelphConfig:
    """Read a velph.toml of velph 0.14 into VelphConfig."""
    with open(filename, "rb") as f:
        try:
            data = tomli.load(f)
        except tomli.TOMLDecodeError as e:
            raise VelphConfigError(f'Error in reading "{filename}": {e}') from e
    return parse_legacy_velph_toml(data, name=str(filename))


def parse_legacy_velph_toml(
    data: Mapping[str, Any], name: str = "velph.toml"
) -> VelphConfig:
    """Convert the contents of a velph.toml of velph 0.14 and read them.

    data is not modified.  An error in the converted contents is reported by
    parse_velph_toml with the section names of the new format, for example
    [vasp.el_dos] for [vasp.el_bands.dos].

    """
    converted = copy.deepcopy(dict(data))
    _convert_code_sections(converted)
    vasp = converted.get("vasp")
    if isinstance(vasp, dict):
        _convert_vasp_section(vasp)
        _convert_kspacing(converted)
    return parse_velph_toml(converted, name=name)


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
            kspacing_to_mesh(kpoints, cell)
            del kpoints["kspacing"]


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
