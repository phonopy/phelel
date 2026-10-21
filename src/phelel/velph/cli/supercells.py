"""Supercells with displacements of phelel, phonopy and phono3py.

These are used by velph phelel init and generate, velph phonopy init and
generate, and velph phono3py init and generate.  The init commands make the
supercells from velph.toml and write <code>/<code>_disp.yaml.  The generate
commands read the yaml file and write the VASP input files of each supercell
in <code>/disp-*.  velph phelel init and velph phono3py init can take the
phonon supercells from phonopy/phonopy_disp.yaml written by velph phonopy init.

"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import click
import numpy as np
from phono3py import Phono3py
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.structure.cells import isclose

from phelel import Phelel
from phelel.interface.phelel_yaml import PhelelYaml
from phelel.velph.cli.utils import get_nac_params, write_jobs
from phelel.velph.config.schema import CodeName
from phelel.velph.jobs.project import Project, read_velph_toml
from phelel.velph.jobs.project_directory import ProjectDirectory
from phelel.velph.jobs.vasp_job import VaspJob, plan_input_set


def set_nac_params(
    ph: Phelel | Phonopy | Phono3py,
    project: Project,
    is_symmetry: bool,
    project_dir: ProjectDirectory,
) -> None:
    """Set NAC parameters read from nac/vasprun.xml, if the directory exists.

    The cell of the NAC calculation is given by "cell" of [vasp.nac].  NAC is
    not supported with the merge scheme of site mixture.  click.ClickException
    is raised when nac/vasprun.xml cannot be parsed.

    """
    nac_directory = project_dir.nac_directory
    if not nac_directory.exists():
        return
    if getattr(ph, "unmerged_unitcell", None) is not None:
        click.echo(
            f'NAC parameters are not read from "{nac_directory}", because NAC is '
            "not supported with the merge scheme of site mixture."
        )
        return
    vasprun_path = nac_directory / "vasprun.xml"
    if not vasprun_path.exists():
        click.echo(
            f'"{vasprun_path}" is not found, so NAC parameters are not set.\n'
            f'Run VASP in "{nac_directory}" and run this command again to set them.'
        )
        return
    click.echo(f'NAC parameters are read from "{vasprun_path}".')
    ph.nac_params = get_nac_params(
        project.cell_of("nac"), vasprun_path, project.primitive, is_symmetry
    )


def load_phonopy_yaml(
    filename: str | pathlib.Path, is_symmetry: bool = True
) -> Phonopy:
    """Return Phonopy with the cells, dataset and NAC params of phonopy_disp.yaml.

    Unlike phonopy.load, files in the current directory, such as FORCE_SETS and
    BORN, are not read.

    """
    return _make_phonopy(PhonopyYaml().read(filename), is_symmetry)


def load_phelel_yaml_as_phonopy(
    filename: str | pathlib.Path, is_symmetry: bool = True
) -> Phonopy:
    """Return Phonopy with the cells, dataset and NAC params of phelel_disp.yaml.

    The supercells with displacements are those of phelel.  The phonon
    supercell and the phonon dataset are not read.  Unlike phelel.load, files
    in the current directory, such as phelel_params.hdf5 and BORN, are not
    read.

    """
    return _make_phonopy(PhelelYaml().read(filename), is_symmetry)


def _make_phonopy(ph_yaml: PhonopyYaml, is_symmetry: bool) -> Phonopy:
    assert ph_yaml.unitcell is not None
    ph = Phonopy(
        ph_yaml.unitcell,
        supercell_matrix=ph_yaml.supercell_matrix,
        primitive_matrix=ph_yaml.primitive_matrix,
        is_symmetry=is_symmetry,
        calculator=ph_yaml.calculator,
        site_mixture_scheme=ph_yaml.site_mixture_scheme or "merge",
    )
    if ph_yaml.dataset is not None:
        ph.dataset = ph_yaml.dataset
    if ph_yaml.nac_params is not None:
        ph.nac_params = ph_yaml.nac_params
    return ph


def load_phono3py_yaml(filename: str | pathlib.Path) -> Phono3py:
    """Return Phono3py with the cells and datasets of phono3py_disp.yaml.

    Unlike phono3py.load, files in the current directory, such as fc3.hdf5,
    FORCES_FC3 and BORN, are not read.

    """
    ph3py_yaml = Phono3pyYaml()
    ph3py_yaml.read(filename)
    assert ph3py_yaml.unitcell is not None
    ph3py = Phono3py(
        ph3py_yaml.unitcell,
        supercell_matrix=ph3py_yaml.supercell_matrix,
        primitive_matrix=ph3py_yaml.primitive_matrix,
        phonon_supercell_matrix=ph3py_yaml.phonon_supercell_matrix,
        calculator=ph3py_yaml.calculator,
    )
    if ph3py_yaml.dataset is not None:
        ph3py.dataset = ph3py_yaml.dataset
    if ph3py_yaml.phonon_dataset is not None:
        ph3py.phonon_dataset = ph3py_yaml.phonon_dataset
    return ph3py


def read_phonopy_phonon(project: Project, project_dir: ProjectDirectory) -> Phonopy:
    """Return Phonopy of phonopy/phonopy_disp.yaml of the project.

    Its supercell matrix and dataset are the phonon supercell matrix and the
    phonon dataset of phelel and phono3py.  click.ClickException is raised when
    the file is not found, or when its unit cell or primitive matrix differs
    from that of velph.toml.

    """
    filename = project_dir.disp_yaml("phonopy")
    if not filename.exists():
        raise click.ClickException(
            f'"{filename}" is not found.\nRun "velph phonopy init" first.'
        )
    ph = load_phonopy_yaml(filename)
    ph_unitcell = ph.unitcell if ph.unmerged_unitcell is None else ph.unmerged_unitcell
    if not (
        isclose(ph_unitcell, project.unitcell)
        and np.allclose(ph.primitive_matrix, project.primitive_matrix)
    ):
        raise click.ClickException(
            f'The unit cell or the primitive cell of "{filename}" differs from that '
            f'of "{project.toml_filename}".\nRun "velph phonopy init" again.'
        )
    return ph


def plan_supercells(
    project: Project,
    code: CodeName,
    ph: Phelel | Phonopy | Phono3py,
    project_dir: ProjectDirectory | None = None,
) -> list[VaspJob]:
    """Return the VASP jobs of the supercells of a code.

    The jobs are the perfect supercell in <code>/disp-000 and the supercells
    with displacements in <code>/disp-001, ...  With the merge scheme of site
    mixture, the atoms of the supercells are those of the unmerged supercell.
    The number of the directory, such as "001", is appended to job_name of the
    scheduler settings.

    """
    if project_dir is None:
        project_dir = ProjectDirectory()
    assert ph.supercells_with_displacements is not None
    if isinstance(ph, (Phelel, Phonopy)) and ph.unmerged_supercell is not None:
        supercell = ph.unmerged_supercell
    else:
        supercell = ph.supercell
    cells = [supercell] + ph.supercells_with_displacements
    directories = project_dir.disp_directories(
        code, len(ph.supercells_with_displacements)
    )
    return [
        plan_input_set(
            project,
            code,
            project_dir=project_dir,
            directory=directory,
            cell=cell,
            job_id=ProjectDirectory.disp_id(directory),
        )
        for directory, cell in zip(directories, cells, strict=True)
    ]


# Readers of <code>/<code>_disp.yaml for the generate commands.
_DISP_YAML_LOADERS: dict[CodeName, Callable[[pathlib.Path], Phonopy | Phono3py]] = {
    "phelel": load_phelel_yaml_as_phonopy,
    "phonopy": load_phonopy_yaml,
    "phono3py": load_phono3py_yaml,
}


def write_supercell_input_files(
    toml_filename: str | pathlib.Path,
    code: CodeName,
    project_dir: ProjectDirectory | None = None,
) -> None:
    """Write the VASP inputs of the supercells of <code>/<code>_disp.yaml.

    click.ClickException is raised when <code>/<code>_disp.yaml, written by the
    init command of the code, does not exist.

    """
    if project_dir is None:
        project_dir = ProjectDirectory()
    yaml_filename = project_dir.disp_yaml(code)
    if not yaml_filename.exists():
        raise click.ClickException(
            f'"{yaml_filename}" not found.\nRun "velph {code} init" to make it.'
        )
    project = read_velph_toml(toml_filename, code)
    ph = _DISP_YAML_LOADERS[code](yaml_filename)
    write_jobs(plan_supercells(project, code, ph, project_dir=project_dir))
