"""VASP jobs: the input files of one VASP calculation in one directory.

A job is one VASP calculation in one directory, such as "relax/iter1" or
"phelel/disp-001".  The generate commands first make a VaspJob for each
directory from velph.toml, without writing anything.  write_vasp_job then
writes the files of a VaspJob.

"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
from collections.abc import Mapping
from typing import Any

import click
from phonopy.interface.vasp import get_vasp_vca_weights
from phonopy.structure.atoms import PhonopyAtoms, parse_cell_dict

from phelel.velph.cli.utils import write_launch_script, write_poscar
from phelel.velph.config.input_sets import INPUT_SETS, InputSetName, KpointsFileName
from phelel.velph.config.io import VelphConfigError
from phelel.velph.config.legacy import load_velph_toml_any_version
from phelel.velph.config.schema import (
    CellChoice,
    InputSetSection,
    KpointsMesh,
    VelphConfig,
)
from phelel.velph.utils.vasp import VaspIncar, VaspKpoints


@dataclasses.dataclass(frozen=True)
class VaspJob:
    """Input files of one VASP calculation in one directory.

    poscar is the cell written as POSCAR, or the path of a file copied as
    POSCAR, such as CONTCAR of the previous relax.  incar is the INCAR as it is
    written.  kpoints maps the name of each k-point file to its mesh.
    scheduler is the scheduler settings of _job.sh, in which job_name is the
    one written.  _job.sh is not written when scheduler is None.
    copied_files maps a file name in the directory to the file copied there,
    such as POTCAR.

    """

    directory: pathlib.Path
    poscar: PhonopyAtoms | pathlib.Path
    incar: Mapping[str, Any]
    kpoints: Mapping[KpointsFileName, KpointsMesh]
    scheduler: Mapping[str, Any] | None = None
    copied_files: Mapping[str, pathlib.Path] = dataclasses.field(default_factory=dict)


def write_vasp_job(job: VaspJob) -> None:
    """Write the VASP input files of a job in its directory."""
    job.directory.mkdir(parents=True, exist_ok=True)
    if isinstance(job.poscar, pathlib.Path):
        shutil.copy2(job.poscar, job.directory / "POSCAR")
    else:
        write_poscar(job.directory, job.poscar)
    VaspIncar.write(job.directory / "INCAR", dict(job.incar))
    for filename, mesh in job.kpoints.items():
        kpoints: dict[str, Any] = {"mesh": mesh.mesh}
        if mesh.shift is not None:
            kpoints["shift"] = mesh.shift
        VaspKpoints.write_mesh_mode(job.directory / filename, kpoints)
    for name, source in job.copied_files.items():
        shutil.copy2(source, job.directory / name)
    if job.scheduler is not None:
        write_launch_script(dict(job.scheduler), job.directory)


def read_velph_toml(
    toml_filename: str | os.PathLike, set_name: InputSetName | None = None
) -> VelphConfig:
    """Read velph.toml for a velph command.

    A velph.toml of the new format or of velph 0.14 is read.
    click.ClickException is raised when the file does not exist, when it has an
    error, or, with set_name, when it does not have [vasp.<set_name>].

    """
    if not pathlib.Path(toml_filename).exists():
        raise click.ClickException(f'"{toml_filename}" not found.')
    try:
        config = load_velph_toml_any_version(toml_filename)
    except VelphConfigError as e:
        raise click.ClickException(str(e)) from e
    if set_name is None:
        return config
    if (
        config.vasp is None
        or config.vasp.input_sets is None
        or set_name not in config.vasp.input_sets
    ):
        raise click.ClickException(
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
    config: VelphConfig, set_name: InputSetName
) -> dict[KpointsFileName, KpointsMesh]:
    """Return the meshes of the k-point files of an input set.

    The k-point files and the k-point blocks that give their meshes are those
    of INPUT_SETS.  The loader of velph.toml checks that the blocks exist and
    that a block other than a line path gives a mesh.  A block given by a line
    path is not supported yet.

    """
    input_set = get_input_set(config, set_name)
    kpoints = {}
    for filename, block in INPUT_SETS[set_name].kpoints_files.items():
        mesh = getattr(input_set, block)
        assert isinstance(mesh, KpointsMesh)
        kpoints[filename] = mesh
    return kpoints


def find_potcar() -> dict[str, pathlib.Path]:
    """Return POTCAR of the current directory as a file to copy, if it exists."""
    potcar = pathlib.Path("POTCAR")
    if potcar.exists():
        return {"POTCAR": potcar}
    return {}
