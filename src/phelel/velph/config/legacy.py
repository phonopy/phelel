"""Reader of velph.toml written by velph earlier than 0.15.

load_velph_toml_any_version reads a velph.toml of the new format or of velph
0.14.  The velph commands read velph.toml through it, so that they also read
the velph.toml files written by velph 0.14.  A file of velph 0.14 is converted
to the new format and read into VelphConfig, so that the generate commands
write the same VASP input files as those of velph 0.14.

velph init of every version up to 0.14 writes [phelel] version, so a
velph.toml of velph earlier than 0.14 is read in the same way as one of velph
0.14.  The conversion is written for the format of velph 0.14 and for
[vasp.el_bands] of velph 0.10 and earlier.  It has been checked on one
velph.toml of velph 0.9.0, which had no other key of the earlier format than
[vasp.el_bands] and "scheduler_name".  A key that the new format does not
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
- [vasp.el_bands] of velph 0.10 and earlier, which has neither bands nor dos,
  is one input set for both the band structure and the DOS.  "velph el_bands
  generate" of these versions writes the band structure with kpoints_opt and
  the INCAR tags without lorbit and nedos, and the DOS with kpoints_dense and
  all the INCAR tags.  [vasp.el_bands] becomes [vasp.el_bands] without
  kpoints_dense and without lorbit and nedos in incar, and, when it has
  kpoints_dense, also [vasp.el_dos] without kpoints_opt.
- [vasp.relax] and [vasp.nac] without "cell" get the cell that the generate
  commands of velph 0.14 use: the unit cell for relax and the primitive cell
  for nac.
- kspacing in [vasp.<input set>.kpoints] and [vasp.<input set>.kpoints_dense]
  is converted to mesh in the same way as the generate commands of velph 0.14
  do, that is, with use_grg=True, so the mesh is a generalized regular grid
  when the cell has a single lattice point.
- scheduler_name in [scheduler] and [vasp.<input set>.scheduler] is removed.
  With scheduler_name = "sge" or "slurm", velph 0.14 writes the job script of
  its own template of SGE or Slurm and does not use scheduler_template.
  scheduler_template of [scheduler] becomes the template that velph 0.14 uses
  with [scheduler] alone, and an input set whose job script velph 0.14 writes
  from another template gets it as scheduler_template of
  [vasp.<input set>.scheduler].  The job scripts are then those of velph 0.14.

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

# INCAR tags that "velph el_bands generate" of velph 0.10 and earlier removes
# from [vasp.el_bands.incar] for the band structure.  The names are compared in
# lower case.
_BANDS_REMOVED_INCAR_TAGS = ("lorbit", "nedos")

# Job script templates of velph 0.14 for scheduler_name = "sge" and "slurm"
# (velph/utils/scheduler.py of velph 0.14).  The lines of _SGE_OPTIONAL_LINES
# are in the template of SGE only when the key is given.
_SGE_HEADER = """#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m n
#$ -N {job_name}
#$ -V
#$ -o _scheduler-stdout.txt
#$ -e _scheduler-stderr.txt
"""
_SGE_OPTIONAL_LINES = (("pe", "#$ -pe {pe}\n"), ("walltime", "#$ -l h_rt={walltime}\n"))
_JOB_COMMANDS = """
{prepend_text}

{mpirun_command} {vasp_binary}

{append_text}
"""
_SLURM_TEMPLATE = (
    """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --time=96:00:00      # Time limit hrs:min:sec
#SBATCH --output=ci_%j.log   # Standard output and error log
"""
    + _JOB_COMMANDS
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
    _convert_scheduler_name(converted)
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
    el_bands = vasp.get("el_bands")
    if isinstance(el_bands, Mapping) and not ({"bands", "dos"} & set(el_bands)):
        messages += _el_bands_of_velph_0_10_messages(el_bands)
    messages += _scheduler_name_messages(data, vasp)
    for name, input_set in _legacy_input_sets(vasp):
        for block in ("kpoints", "kpoints_dense"):
            kpoints = input_set.get(block)
            if isinstance(kpoints, Mapping) and "kspacing" in kpoints:
                messages.append(
                    f"kspacing of [vasp.{name}.{block}] was replaced by mesh."
                )
    return messages


def _el_bands_of_velph_0_10_messages(el_bands: Mapping[str, Any]) -> list[str]:
    """Return how [vasp.el_bands] of velph 0.10 and earlier is converted."""
    messages = []
    if "kpoints_dense" in el_bands:
        messages.append(
            "[vasp.el_bands] was split into [vasp.el_bands] with kpoints_opt and "
            "[vasp.el_dos] with kpoints_dense."
        )
    incar = el_bands.get("incar")
    if isinstance(incar, Mapping):
        removed = [
            tag for tag in incar if tag.lower().strip() in _BANDS_REMOVED_INCAR_TAGS
        ]
        if removed:
            verb = "was" if len(removed) == 1 else "were"
            messages.append(
                f"{' and '.join(removed)} {verb} removed from [vasp.el_bands.incar]."
            )
    return messages


def _scheduler_name_messages(
    data: Mapping[str, Any], vasp: Mapping[str, Any]
) -> list[str]:
    """Return the sections whose scheduler_name the conversion removes."""
    sections = []
    scheduler = data.get("scheduler")
    if isinstance(scheduler, Mapping) and "scheduler_name" in scheduler:
        sections.append(("scheduler", scheduler["scheduler_name"]))
    for name, input_set in _legacy_input_sets(vasp):
        own = input_set.get("scheduler")
        if isinstance(own, Mapping) and "scheduler_name" in own:
            sections.append((f"vasp.{name}.scheduler", own["scheduler_name"]))
    messages = [
        f'scheduler_name = "{value}" of [{section}] was removed.'
        for section, value in sections
    ]
    if isinstance(scheduler, Mapping) and any(
        value in ("sge", "slurm") for _, value in sections
    ):
        messages.append(
            "scheduler_template was written to give the job scripts that velph "
            "0.14 writes for scheduler_name."
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
    elif isinstance(el_bands, dict):
        _split_el_bands_of_velph_0_10(vasp, el_bands)
    for set_name, default_cell in _DEFAULT_CELLS.items():
        input_set = vasp.get(set_name)
        if not isinstance(input_set, dict):
            continue
        cell = input_set.get("cell", default_cell)
        input_set["cell"] = cell.lower() if isinstance(cell, str) else cell


def _split_el_bands_of_velph_0_10(
    vasp: dict[str, Any], el_bands: dict[str, Any]
) -> None:
    """Split [vasp.el_bands] of velph 0.10 and earlier into el_bands and el_dos.

    The input sets are those that "velph el_bands generate" of velph 0.10
    writes in el_bands/bands and el_bands/dos (see the docstring of this
    module).  el_dos is made only when el_bands has kpoints_dense, because
    the DOS is not written without it.

    """
    bands = {key: value for key, value in el_bands.items() if key != "kpoints_dense"}
    incar = bands.get("incar")
    if isinstance(incar, dict):
        bands["incar"] = {
            tag: value
            for tag, value in incar.items()
            if tag.lower().strip() not in _BANDS_REMOVED_INCAR_TAGS
        }
    vasp["el_bands"] = bands
    if "kpoints_dense" in el_bands:
        # A copy, so that el_bands and el_dos share no table, for example
        # kpoints, whose kspacing _convert_kspacing replaces in place.
        vasp["el_dos"] = copy.deepcopy(
            {key: value for key, value in el_bands.items() if key != "kpoints_opt"}
        )


def _template_of_velph_0_14(scheduler: Mapping[str, Any]) -> str | None:
    """Return the job script template that velph 0.14 uses with scheduler.

    scheduler is the settings of a job, [scheduler] together with
    [vasp.<input set>.scheduler].  None is returned when velph 0.14 has no
    template for it, and then velph 0.14 stops when it writes the job script.

    """
    name = scheduler.get("scheduler_name")
    if name == "sge":
        optional = "".join(
            line for key, line in _SGE_OPTIONAL_LINES if key in scheduler
        )
        return _SGE_HEADER + optional + _JOB_COMMANDS
    if name == "slurm":
        return _SLURM_TEMPLATE
    return scheduler.get("scheduler_template")


def _convert_scheduler_name(data: dict[str, Any]) -> None:
    """Replace scheduler_name by the scheduler_template of velph 0.14.

    See the docstring of this module.  Without [scheduler], velph 0.14 writes
    no job script, and scheduler_name of [vasp.<input set>.scheduler] is only
    removed.  An input set for which velph 0.14 has no template is left with
    the template of [scheduler], because no template stands for the error of
    velph 0.14.

    """
    vasp = data.get("vasp")
    own_schedulers = {
        set_name: input_set["scheduler"]
        for set_name, input_set in (vasp.items() if isinstance(vasp, dict) else ())
        if isinstance(input_set, dict) and isinstance(input_set.get("scheduler"), dict)
    }
    scheduler = data.get("scheduler")
    if not isinstance(scheduler, dict):
        for own in own_schedulers.values():
            own.pop("scheduler_name", None)
        return
    if "scheduler_name" not in scheduler and not any(
        "scheduler_name" in own for own in own_schedulers.values()
    ):
        return
    template = _template_of_velph_0_14(scheduler)
    for own in own_schedulers.values():
        own_template = _template_of_velph_0_14({**scheduler, **own})
        own.pop("scheduler_name", None)
        if own_template is not None and own_template != own.get(
            "scheduler_template", template
        ):
            own["scheduler_template"] = own_template
    scheduler.pop("scheduler_name", None)
    if template is not None:
        scheduler["scheduler_template"] = template


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
