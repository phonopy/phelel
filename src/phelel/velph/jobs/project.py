"""velph.toml read for the velph commands that use it.

read_velph_toml reads velph.toml into Project.  Project holds VelphConfig and
the values that the commands take from it: the unit cell and the primitive
cell, the input sets, the sections [phelel], [phonopy] and [phono3py] with the
defaults of velph init, and the meshes and band paths of the k-point files of
each input set.  The loader of velph.toml (config/io.py) checks what velph.toml
has to give, such as the cells and the k-point blocks of each input set, and
the methods of Project rely on these checks.

"""

from __future__ import annotations

import dataclasses
import os
import pathlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms, parse_cell_dict

from phelel.velph.config.defaults import get_default_config
from phelel.velph.config.input_sets import INPUT_SETS, KpointsFileName
from phelel.velph.config.legacy import load_velph_toml_any_version
from phelel.velph.config.merge import merge
from phelel.velph.config.schema import (
    CodeName,
    CodeSection,
    InputSetName,
    InputSetSection,
    KpointsMesh,
    LinePath,
    SupercellDimension,
    SupercellMatrix,
    VelphConfig,
)
from phelel.velph.errors import VelphConfigError, VelphError
from phelel.velph.utils.kpath import get_seekpath_line_path


def read_velph_toml(
    toml_filename: str | os.PathLike, set_name: InputSetName | None = None
) -> Project:
    """Read velph.toml for a velph command.

    A velph.toml of the new format or of velph 0.14 is read.  VelphError is
    raised when the file does not exist.  VelphConfigError is raised when it
    has an error, or, with set_name, when it does not have [vasp.<set_name>].

    """
    if not pathlib.Path(toml_filename).exists():
        raise VelphError(f'"{toml_filename}" not found.')
    config = load_velph_toml_any_version(toml_filename)
    project = Project.from_config(config, toml_filename)
    if set_name is not None:
        project.input_set(set_name)
    return project


@dataclasses.dataclass(frozen=True)
class Project:
    """velph.toml read for a velph command.

    Attributes
    ----------
    config : VelphConfig
        Contents of velph.toml, checked by the loader of velph.toml.
    toml_filename : pathlib.Path
        Name of velph.toml, used in error messages.
    unitcell : PhonopyAtoms
        [unitcell] of velph.toml.  It is made once, when Project is made, and
        is shared by the jobs made from Project, so it is not to be modified.
    primitive : PhonopyAtoms
        [primitive_cell] of velph.toml.  It is made and shared in the same way
        as unitcell.

    """

    config: VelphConfig
    toml_filename: pathlib.Path
    unitcell: PhonopyAtoms
    primitive: PhonopyAtoms

    @classmethod
    def from_config(
        cls, config: VelphConfig, toml_filename: str | os.PathLike = "velph.toml"
    ) -> Project:
        """Return Project of VelphConfig read from velph.toml.

        config has [unitcell] and [primitive_cell], as the loader of velph.toml
        checks.

        """
        return cls(
            config=config,
            toml_filename=pathlib.Path(toml_filename),
            unitcell=_parse_cell(config.unitcell),
            primitive=_parse_cell(config.primitive_cell),
        )

    @property
    def primitive_matrix(self) -> NDArray[np.double]:
        """Return the matrix that transforms the unit cell into the primitive cell.

        The basis vectors of the primitive cell are those of the unit cell
        multiplied by this matrix, in the convention of phonopy.

        """
        return np.dot(np.linalg.inv(self.unitcell.cell.T), self.primitive.cell.T)

    @property
    def input_sets(self) -> Mapping[InputSetName, InputSetSection]:
        """Return the input sets of [vasp], in the order of velph.toml."""
        if self.config.vasp is None or self.config.vasp.input_sets is None:
            return MappingProxyType({})
        return self.config.vasp.input_sets

    def input_set(self, name: InputSetName) -> InputSetSection:
        """Return the input set [vasp.<name>].

        VelphConfigError is raised when velph.toml does not have it.

        """
        if name not in self.input_sets:
            raise VelphConfigError(
                f'[vasp.{name}] is not found in "{self.toml_filename}".\n'
                f'Add [vasp.{name}] to "{self.toml_filename}", or run "velph init" '
                "again."
            )
        return self.input_sets[name]

    def code_section(self, code: CodeName) -> CodeSection:
        """Return [<code>] of velph.toml, with the defaults of velph init.

        A field that [<code>] does not give, such as amplitude, has the value
        that velph init writes by default.  So the fields amplitude, diagonal,
        plusminus and nosym are not None.  VelphConfigError is raised when
        velph.toml does not have [<code>], or when [<code>] does not give the
        supercell, which the loader of velph.toml also checks.

        """
        section = getattr(self.config, code)
        if section is None:
            raise VelphConfigError(
                f'[{code}] is not found in "{self.toml_filename}".\n'
                f'Add [{code}] to "{self.toml_filename}", or run "velph init" again.'
            )
        merged = merge(getattr(get_default_config(), code), section)
        assert isinstance(merged, CodeSection)
        if merged.supercell is None:
            raise VelphConfigError(
                f'[{code}] of "{self.toml_filename}" does not give the supercell.\n'
                f'Give "supercell_dimension" or "supercell_matrix" in [{code}].'
            )
        return merged

    def supercell_matrix(self, code: CodeName) -> NDArray[np.int64]:
        """Return the supercell matrix of [<code>] as a 3x3 matrix."""
        supercell = self.code_section(code).supercell
        assert isinstance(supercell, (SupercellDimension, SupercellMatrix))
        return get_supercell_matrix(supercell)

    def cell_of(self, name: InputSetName) -> PhonopyAtoms:
        """Return the cell on which the VASP calculation of an input set is run.

        It is the cell given by "cell" of the input set for relax and nac, and
        the primitive cell for the other input sets, as INPUT_SETS gives.  The
        input sets of phelel, phonopy and phono3py are run on supercells, which
        are not in velph.toml, and ValueError is raised for them.

        """
        entry = INPUT_SETS[name]
        if entry.cell == "supercells":
            raise ValueError(f"{name} is run on the supercells of {name}.")
        if entry.cell == "primitive":
            return self.primitive
        cell_choice = self.input_set(name).cell
        assert cell_choice is not None
        return self.unitcell if cell_choice == "unitcell" else self.primitive

    def kpoints(
        self, name: InputSetName
    ) -> dict[KpointsFileName, KpointsMesh | LinePath]:
        """Return the meshes and band paths of the k-point files of an input set.

        The k-point files and the k-point blocks that give them are those of
        INPUT_SETS.  The loader of velph.toml checks that the blocks exist and
        that a block other than a line block gives a mesh.  A line block
        without "path" gets the band path that seekpath gives for the cell of
        the input set (cell_of).  VelphError is raised when seekpath is not
        installed.

        """
        input_set = self.input_set(name)
        kpoints: dict[KpointsFileName, KpointsMesh | LinePath] = {}
        for filename, block in INPUT_SETS[name].kpoints_files.items():
            value = getattr(input_set, block)
            if isinstance(value, LinePath):
                if value.path is None:
                    value = _seekpath_line_path(
                        self.cell_of(name), value.line, f"vasp.{name}.{block}"
                    )
            else:
                assert isinstance(value, KpointsMesh)
            kpoints[filename] = value
        return kpoints


def get_supercell_matrix(
    supercell: SupercellDimension | SupercellMatrix,
) -> NDArray[np.int64]:
    """Return the supercell matrix of a supercell of velph.toml as a 3x3 matrix.

    supercell_dimension gives the diagonal elements of the matrix.

    """
    if isinstance(supercell, SupercellDimension):
        return np.diag(supercell.supercell_dimension)
    return np.array(supercell.supercell_matrix)


def _parse_cell(cell_dict: Mapping[str, Any] | None) -> PhonopyAtoms:
    assert cell_dict is not None
    cell = parse_cell_dict(dict(cell_dict))  # type: ignore[arg-type]
    assert cell is not None
    return cell


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
