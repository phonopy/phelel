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
from phonopy.structure.atoms import PhonopyAtoms

from phelel.velph.config.input_sets import KpointsFileName
from phelel.velph.config.io import VelphConfigError
from phelel.velph.config.schema import InputSetName, KpointsMesh, LinePath
from phelel.velph.jobs.project import Project
from phelel.velph.utils.vasp import VaspIncar, VaspKpoints


@dataclasses.dataclass(frozen=True)
class VaspJob:
    """Input files of one VASP calculation in one directory.

    Attributes
    ----------
    directory : pathlib.Path
        Directory in which the input files are written, such as
        "relax/iter1".  It is made when it does not exist.
    cell : PhonopyAtoms
        Cell of the calculation.  It is written as POSCAR, unless
        poscar_source is given.
    incar : Mapping[str, Any]
        INCAR tags as they are written, with the tags that cell needs, such as
        MAGMOM (add_cell_tags).
    kpoints : Mapping[KpointsFileName, KpointsMesh | LinePath]
        Name of each k-point file mapped to its mesh, or to its band path,
        which has to give "path".
    job_script : str or None
        _job.sh as it is written.  _job.sh is not written when it is None.
    copied_files : Mapping[str, pathlib.Path]
        Name of a file in directory mapped to the file copied there, such as
        {"POTCAR": pathlib.Path("POTCAR")}.
    poscar_source : pathlib.Path or None
        File copied as POSCAR in place of cell, such as CONTCAR of the
        previous relax.  cell is then still used for the INCAR tags and for
        the hint of the species rows of site mixture.

    """

    directory: pathlib.Path
    cell: PhonopyAtoms
    incar: Mapping[str, Any]
    kpoints: Mapping[KpointsFileName, KpointsMesh | LinePath]
    job_script: str | None = None
    copied_files: Mapping[str, pathlib.Path] = dataclasses.field(default_factory=dict)
    poscar_source: pathlib.Path | None = None


def write_vasp_job(job: VaspJob) -> None:
    """Write the VASP input files of a job in its directory."""
    job.directory.mkdir(parents=True, exist_ok=True)
    if job.poscar_source is not None:
        shutil.copy2(job.poscar_source, job.directory / "POSCAR")
    else:
        write_poscar(job.directory, job.cell)
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


def effective_scheduler(
    project: Project, set_name: InputSetName, job_id: str | None = None
) -> dict[str, Any] | None:
    """Return the scheduler settings of a job of an input set.

    These are the settings of [scheduler] together with those of
    [vasp.<set_name>.scheduler], which have priority.  With job_id,
    "-<job_id>" is appended to job_name, for example "vasp-elph-iter1".  None
    is returned when velph.toml has no [scheduler], and then _job.sh is not
    written.

    """
    if project.config.scheduler is None:
        return None
    scheduler = dict(project.config.scheduler)
    scheduler.update(project.input_set(set_name).scheduler or {})
    if job_id is not None and "job_name" in scheduler:
        scheduler["job_name"] = f"{scheduler['job_name']}-{job_id}"
    return scheduler


def job_script(
    project: Project, set_name: InputSetName, job_id: str | None = None
) -> str | None:
    """Return _job.sh of a job of an input set.

    The scheduler settings are those of effective_scheduler.  None is returned
    when velph.toml has no [scheduler].  VelphConfigError is raised by
    make_job_script when the job script template cannot be filled.

    """
    scheduler = effective_scheduler(project, set_name, job_id=job_id)
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
