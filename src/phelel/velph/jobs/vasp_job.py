"""VASP jobs: the input files of one VASP calculation in one directory.

A job is one VASP calculation in one directory, such as "relax/iter1" or
"phelel/disp-001".  The generate commands first make a VaspJob for each
directory from velph.toml, without writing anything.  write_vasp_job then
writes the files of a VaspJob.  An error found in making the jobs, such as in
the job script template, stops a command before anything is written.

"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
from collections.abc import Mapping
from typing import Any

from phonopy.interface.vasp import get_vasp_vca_weights, write_vasp
from phonopy.structure.atoms import PhonopyAtoms, parse_cell_dict

from phelel.velph.config.input_sets import INPUT_SETS, InputSetName, KpointsFileName
from phelel.velph.config.io import VelphConfigError
from phelel.velph.config.legacy import load_velph_toml_any_version
from phelel.velph.config.schema import (
    CellChoice,
    InputSetSection,
    KpointsMesh,
    LinePath,
    VelphConfig,
)
from phelel.velph.errors import VelphError
from phelel.velph.utils.kpath import get_seekpath_line_path
from phelel.velph.utils.vasp import VaspIncar, VaspKpoints


@dataclasses.dataclass(frozen=True)
class VaspJob:
    """Input files of one VASP calculation in one directory.

    poscar is the cell written as POSCAR, or the path of a file copied as
    POSCAR, such as CONTCAR of the previous relax.  incar is the INCAR as it is
    written.  kpoints maps the name of each k-point file to its mesh, or to
    its band path, which has to give "path".
    job_script is _job.sh as it is written, and _job.sh is not written when
    job_script is None.
    copied_files maps a file name in the directory to the file copied there,
    such as POTCAR.

    """

    directory: pathlib.Path
    poscar: PhonopyAtoms | pathlib.Path
    incar: Mapping[str, Any]
    kpoints: Mapping[KpointsFileName, KpointsMesh | LinePath]
    job_script: str | None = None
    copied_files: Mapping[str, pathlib.Path] = dataclasses.field(default_factory=dict)


def write_vasp_job(job: VaspJob) -> None:
    """Write the VASP input files of a job in its directory."""
    job.directory.mkdir(parents=True, exist_ok=True)
    if isinstance(job.poscar, pathlib.Path):
        shutil.copy2(job.poscar, job.directory / "POSCAR")
    else:
        write_poscar(job.directory, job.poscar)
    VaspIncar.write(job.directory / "INCAR", dict(job.incar))
    for filename, block in job.kpoints.items():
        kpoints: dict[str, Any]
        if isinstance(block, LinePath):
            assert block.path is not None
            kpoints = {"line": block.line, "path": block.path}
            if block.label is not None:
                kpoints["label"] = block.label
            VaspKpoints.write_line_mode(job.directory / filename, kpoints)
        else:
            kpoints = {"mesh": block.mesh}
            if block.shift is not None:
                kpoints["shift"] = block.shift
            VaspKpoints.write_mesh_mode(job.directory / filename, kpoints)
    for name, source in job.copied_files.items():
        shutil.copy2(source, job.directory / name)
    if job.job_script is not None:
        (job.directory / "_job.sh").write_text(job.job_script)


def read_velph_toml(
    toml_filename: str | os.PathLike, set_name: InputSetName | None = None
) -> VelphConfig:
    """Read velph.toml for a velph command.

    A velph.toml of the new format or of velph 0.14 is read.  VelphError is
    raised when the file does not exist.  VelphConfigError is raised when it
    has an error, or, with set_name, when it does not have [vasp.<set_name>].

    """
    if not pathlib.Path(toml_filename).exists():
        raise VelphError(f'"{toml_filename}" not found.')
    config = load_velph_toml_any_version(toml_filename)
    if set_name is None:
        return config
    if (
        config.vasp is None
        or config.vasp.input_sets is None
        or set_name not in config.vasp.input_sets
    ):
        raise VelphConfigError(
            f'[vasp.{set_name}] is not found in "{toml_filename}".\n'
            f'Add [vasp.{set_name}] to "{toml_filename}", or run "velph init" again.'
        )
    return config


def get_input_set(config: VelphConfig, set_name: InputSetName) -> InputSetSection:
    """Return the input set [vasp.<set_name>].

    The caller checks that velph.toml has the input set.

    """
    assert config.vasp is not None and config.vasp.input_sets is not None
    return config.vasp.input_sets[set_name]


def effective_scheduler(
    config: VelphConfig, set_name: InputSetName, job_id: str | None = None
) -> dict[str, Any] | None:
    """Return the scheduler settings of a job of an input set.

    These are the settings of [scheduler] together with those of
    [vasp.<set_name>.scheduler], which have priority.  With job_id,
    "-<job_id>" is appended to job_name, for example "vasp-elph-iter1".  None
    is returned when velph.toml has no [scheduler], and then _job.sh is not
    written.

    """
    if config.scheduler is None:
        return None
    scheduler = dict(config.scheduler)
    scheduler.update(get_input_set(config, set_name).scheduler or {})
    if job_id is not None and "job_name" in scheduler:
        scheduler["job_name"] = f"{scheduler['job_name']}-{job_id}"
    return scheduler


def job_script(
    config: VelphConfig, set_name: InputSetName, job_id: str | None = None
) -> str | None:
    """Return _job.sh of a job of an input set.

    The scheduler settings are those of effective_scheduler.  None is returned
    when velph.toml has no [scheduler].  VelphConfigError is raised by
    make_job_script when the job script template cannot be filled.

    """
    scheduler = effective_scheduler(config, set_name, job_id=job_id)
    if scheduler is None:
        return None
    return make_job_script(scheduler)


def add_cell_tags(incar: Mapping[str, Any], cell: PhonopyAtoms) -> dict[str, Any]:
    """Return INCAR tags with the tags that the cell needs.

    MAGMOM is added when the cell has magnetic moments.  VCA is added when the
    cell has the weighted species of site mixture.

    """
    incar_with_cell_tags = dict(incar)
    if cell.magnetic_moments is not None:
        incar_with_cell_tags["magmom"] = cell.magnetic_moments.tolist()
    if cell.has_weighted_species:
        incar_with_cell_tags["vca"] = get_vasp_vca_weights(cell)
    return incar_with_cell_tags


def get_cell(config: VelphConfig, cell_choice: CellChoice) -> PhonopyAtoms:
    """Return [unitcell] or [primitive_cell] of velph.toml."""
    cell_dict = config.unitcell if cell_choice == "unitcell" else config.primitive_cell
    assert cell_dict is not None
    cell = parse_cell_dict(dict(cell_dict))  # type: ignore[arg-type]
    assert cell is not None
    return cell


def get_kpoints(
    config: VelphConfig, set_name: InputSetName, cell: PhonopyAtoms | None = None
) -> dict[KpointsFileName, KpointsMesh | LinePath]:
    """Return the meshes and band paths of the k-point files of an input set.

    The k-point files and the k-point blocks that give them are those of
    INPUT_SETS.  The loader of velph.toml checks that the blocks exist and that
    a block other than a line block gives a mesh.  A line block without "path"
    gets the band path that seekpath gives for cell, which is then required.
    VelphError is raised when seekpath is not installed.

    """
    input_set = get_input_set(config, set_name)
    kpoints: dict[KpointsFileName, KpointsMesh | LinePath] = {}
    for filename, block in INPUT_SETS[set_name].kpoints_files.items():
        value = getattr(input_set, block)
        if isinstance(value, LinePath):
            if value.path is None:
                assert cell is not None
                value = _seekpath_line_path(
                    cell, value.line, f"vasp.{set_name}.{block}"
                )
        else:
            assert isinstance(value, KpointsMesh)
        kpoints[filename] = value
    return kpoints


def _seekpath_line_path(cell: PhonopyAtoms, line: int, where: str) -> LinePath:
    try:
        return get_seekpath_line_path(cell, line)
    except ModuleNotFoundError as e:
        if e.name != "seekpath":
            raise
        raise VelphError(
            f'[{where}] does not give "path", and seekpath, which makes the band '
            "path, is not installed.\n"
            f'Install seekpath, or give "path" and "label" in [{where}].'
        ) from e


def write_poscar(
    directory: os.PathLike, cell: PhonopyAtoms, filename: str = "POSCAR"
) -> None:
    """Write POSCAR.

    A cell with weighted species of site mixture is written for VASP VCA. Atoms
    of one symbol and weight that are next to each other in the cell are written
    in one species row. The INCAR tag VCA is added by add_cell_tags.

    """
    write_vasp(
        pathlib.Path(directory) / filename, cell, for_vca=cell.has_weighted_species
    )


_BRACES_HINT = (
    '  To write "{" or "}" in _job.sh, write "{{" or "}}" in scheduler_template.'
)


def make_job_script(scheduler: Mapping[str, Any]) -> str:
    """Return the job script _job.sh made from the scheduler settings of a job.

    The keys in braces in scheduler_template, such as {job_name}, are replaced
    by the values of the settings, as str.format does.  VelphConfigError is
    raised when scheduler_template is not given, when a key that it uses is not
    given, and when str.format cannot read it, for example for a single "}".

    """
    if "scheduler_template" not in scheduler:
        raise VelphConfigError(
            '"scheduler_template" has to be given in [scheduler] or '
            "[vasp.<input set>.scheduler]."
        )
    try:
        return scheduler["scheduler_template"].format(**scheduler)
    except KeyError as e:
        raise VelphConfigError(
            f'"{e.args[0]}" used in the job script template is not given in '
            "[scheduler] or [vasp.<input set>.scheduler]." + _BRACES_HINT
        ) from e
    except (IndexError, ValueError) as e:
        raise VelphConfigError(
            f"The job script template cannot be read: {e}." + _BRACES_HINT
        ) from e


def find_potcar() -> dict[str, pathlib.Path]:
    """Return POTCAR of the current directory as a file to copy, if it exists."""
    potcar = pathlib.Path("POTCAR")
    if potcar.exists():
        return {"POTCAR": potcar}
    return {}
