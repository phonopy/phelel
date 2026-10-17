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

import click
import numpy as np
from numpy.typing import NDArray
from phono3py import Phono3py
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import isclose

from phelel import Phelel
from phelel.interface.phelel_yaml import PhelelYaml
from phelel.velph.cli.utils import get_nac_params, get_num_digits
from phelel.velph.config.defaults import get_default_config
from phelel.velph.config.merge import merge
from phelel.velph.config.schema import (
    CodeName,
    CodeSection,
    SupercellDimension,
    SupercellMatrix,
    VelphConfig,
)
from phelel.velph.jobs.vasp_job import (
    VaspJob,
    add_cell_tags,
    find_potcar,
    get_cell,
    get_input_set,
    get_kpoints,
    job_script,
)

PHONOPY_DISP_YAML = pathlib.Path("phonopy") / "phonopy_disp.yaml"


def get_code_section(
    config: VelphConfig, code: CodeName, toml_filename: str | pathlib.Path
) -> CodeSection:
    """Return [<code>] of velph.toml, with the defaults of velph init.

    A field that [<code>] does not give, such as amplitude, has the value that
    velph init writes by default.  click.ClickException is raised when
    velph.toml does not have [<code>] or when [<code>] does not give the
    supercell.

    """
    section = getattr(config, code)
    if section is None:
        raise click.ClickException(
            f'[{code}] is not found in "{toml_filename}".\n'
            f'Add [{code}] to "{toml_filename}", or run "velph init" again.'
        )
    merged = merge(getattr(get_default_config(), code), section)
    assert isinstance(merged, CodeSection)
    if merged.supercell is None:
        raise click.ClickException(
            f'[{code}] of "{toml_filename}" does not give the supercell.\n'
            f'Give "supercell_dimension" or "supercell_matrix" in [{code}].'
        )
    return merged


def get_supercell_matrix(section: CodeSection) -> NDArray[np.int64]:
    """Return the supercell matrix of a code section as a 3x3 matrix."""
    if isinstance(section.supercell, SupercellDimension):
        return np.diag(section.supercell.supercell_dimension)
    assert isinstance(section.supercell, SupercellMatrix)
    return np.array(section.supercell.supercell_matrix)


def get_cells(
    config: VelphConfig,
) -> tuple[PhonopyAtoms, PhonopyAtoms, NDArray[np.double]]:
    """Return the unit cell, the primitive cell and the primitive matrix.

    The primitive matrix transforms the basis vectors of the unit cell into
    those of the primitive cell.

    """
    unitcell = get_cell(config, "unitcell")
    primitive = get_cell(config, "primitive")
    primitive_matrix = np.dot(np.linalg.inv(unitcell.cell.T), primitive.cell.T)
    return unitcell, primitive, primitive_matrix


def set_nac_params(
    ph: Phelel | Phonopy | Phono3py,
    config: VelphConfig,
    is_symmetry: bool,
    current_directory: pathlib.Path,
) -> None:
    """Set NAC parameters read from nac/vasprun.xml, if the directory exists.

    The cell of the NAC calculation is given by "cell" of [vasp.nac].  NAC is
    not supported with the merge scheme of site mixture.

    """
    nac_directory = current_directory / "nac"
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
    unitcell, primitive, _ = get_cells(config)
    cell_choice = get_input_set(config, "nac").cell
    nac_cell = primitive if cell_choice == "primitive" else unitcell
    nac_params = get_nac_params(nac_cell, vasprun_path, primitive, is_symmetry)
    if nac_params is not None:
        ph.nac_params = nac_params


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


def read_phonopy_phonon(
    config: VelphConfig,
    current_directory: pathlib.Path,
    toml_filename: str | pathlib.Path,
) -> Phonopy:
    """Return Phonopy of phonopy/phonopy_disp.yaml in current_directory.

    Its supercell matrix and dataset are the phonon supercell matrix and the
    phonon dataset of phelel and phono3py.  click.ClickException is raised when
    the file is not found, or when its unit cell or primitive matrix differs
    from that of velph.toml.

    """
    filename = current_directory / PHONOPY_DISP_YAML
    if not filename.exists():
        raise click.ClickException(
            f'"{filename}" is not found.\nRun "velph phonopy init" first.'
        )
    ph = load_phonopy_yaml(filename)
    unitcell, _, primitive_matrix = get_cells(config)
    ph_unitcell = ph.unitcell if ph.unmerged_unitcell is None else ph.unmerged_unitcell
    if not (
        isclose(ph_unitcell, unitcell)
        and np.allclose(ph.primitive_matrix, primitive_matrix)
    ):
        raise click.ClickException(
            f'The unit cell or the primitive cell of "{filename}" differs from that '
            f'of "{toml_filename}".\nRun "velph phonopy init" again.'
        )
    return ph


def plan_supercells(
    config: VelphConfig, code: CodeName, ph: Phelel | Phonopy | Phono3py
) -> list[VaspJob]:
    """Return the VASP jobs of the supercells of a code.

    The jobs are the perfect supercell in <code>/disp-000 and the supercells
    with displacements in <code>/disp-001, ...  With the merge scheme of site
    mixture, the atoms of the supercells are those of the unmerged supercell.
    The directory name, such as "001", is appended to job_name of the
    scheduler settings.

    """
    assert ph.supercells_with_displacements is not None
    if isinstance(ph, (Phelel, Phonopy)) and ph.unmerged_supercell is not None:
        supercell = ph.unmerged_supercell
    else:
        supercell = ph.supercell
    incar = get_input_set(config, code).incar or {}
    kpoints = get_kpoints(config, code)
    copied_files = find_potcar()
    nd = get_num_digits(ph.supercells_with_displacements)
    jobs = []
    for i, cell in enumerate([supercell] + ph.supercells_with_displacements):
        id_number = f"{i:0{nd}d}"
        jobs.append(
            VaspJob(
                directory=pathlib.Path(f"{code}/disp-{id_number}"),
                poscar=cell,
                incar=add_cell_tags(incar, cell),
                kpoints=kpoints,
                job_script=job_script(config, code, job_id=id_number),
                copied_files=copied_files,
            )
        )
    return jobs
