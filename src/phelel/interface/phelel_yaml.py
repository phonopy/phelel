"""phelel_yaml reader and writer."""

# Copyright (C) 2021 Atsushi Togo
# All rights reserved.
#
# This file is part of phelel.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import (
    PhonopyYaml,
    PhonopyYamlData,
    PhonopyYamlDumper,
    PhonopyYamlLoader,
    load_yaml,
)
from phonopy.physical_units import CalculatorPhysicalUnits
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive, Supercell, isclose

if TYPE_CHECKING:
    from phelel import Phelel


@dataclasses.dataclass
class PhelelYamlData(PhonopyYamlData):
    """PhelelYaml data structure."""

    command_name: str = "phelel"
    phonon_supercell_matrix: NDArray | None = None
    phonon_dataset: dict | None = None
    phonon_supercell: Supercell | PhonopyAtoms | None = None
    phonon_primitive: Primitive | PhonopyAtoms | None = None


class PhelelYamlLoader(PhonopyYamlLoader):
    """PhelelYaml loader."""

    def __init__(
        self,
        yaml_data: dict,
        configuration: dict | None = None,
        calculator: str | None = None,
        physical_units: CalculatorPhysicalUnits | None = None,
    ):
        """Init method.

        Parameters
        ----------
        yaml_data : dict

        """
        self._yaml = yaml_data
        self._data = PhelelYamlData(
            configuration=configuration,
            calculator=calculator,
            physical_units=physical_units,
        )

    @property
    def data(self) -> PhelelYamlData:
        """Return PhelelYamlData instance."""
        return self._data

    def parse(self):
        """Yaml dict is parsed. See docstring of this class."""
        super().parse()
        self._parse_phonon_dataset()
        return self

    def _parse_all_cells(self):
        """Parse all cells.

        This method override PhonopyYaml._parse_all_cells.

        """
        super()._parse_all_cells()
        if "phonon_primitive_cell" in self._yaml:
            self._data.phonon_primitive = self._parse_cell(
                self._yaml["phonon_primitive_cell"]
            )
        if "phonon_supercell" in self._yaml:
            self._data.phonon_supercell = self._parse_cell(
                self._yaml["phonon_supercell"]
            )
        if "phonon_supercell_matrix" in self._yaml:
            self._data.phonon_supercell_matrix = np.array(
                self._yaml["phonon_supercell_matrix"], dtype="intc", order="C"
            )

    def _parse_phonon_dataset(self):
        """Parse force dataset for phonon."""
        self._data.phonon_dataset = self._get_dataset(
            self._data.phonon_supercell, key_prefix="phonon_"
        )


class PhelelYamlDumper(PhonopyYamlDumper):
    """PhelelYaml dumper."""

    default_settings = {
        "force_sets": False,
        "displacements": True,
        "force_constants": False,
        "born_effective_charge": True,
        "dielectric_constant": True,
    }

    def __init__(self, data: PhelelYamlData, dumper_settings: dict | None = None):
        """Init method."""
        self._data = data
        self._init_dumper_settings(dumper_settings)

    def _cell_info_yaml_lines(self):
        """Get YAML lines for information of cells.

        This method override PhonopyYaml._cell_info_yaml_lines.

        """
        lines = super()._cell_info_yaml_lines()
        lines += self._supercell_matrix_yaml_lines(
            self._data.phonon_supercell_matrix, "phonon_supercell_matrix"
        )
        lines += self._primitive_yaml_lines(
            self._data.phonon_primitive, "phonon_primitive_cell"
        )
        lines += self._phonon_supercell_yaml_lines()
        return lines

    def _phonon_supercell_matrix_yaml_lines(self):
        lines = []
        if self._data.phonon_supercell_matrix is not None:
            lines.append("phonon_supercell_matrix:")
            assert self._data.supercell_matrix is not None
            for v in self._data.supercell_matrix:
                lines.append("- [ %3d, %3d, %3d ]" % tuple(v))
            lines.append("")
        return lines

    def _phonon_supercell_yaml_lines(self):
        lines = []
        if self._data.phonon_supercell is not None:
            s2p_map = getattr(self._data.phonon_primitive, "s2p_map", None)
            lines += self._cell_yaml_lines(
                self._data.phonon_supercell, "phonon_supercell", s2p_map
            )
            lines.append("")
        return lines

    def _nac_yaml_lines(self):
        """Get YAML lines for parameters of non-analytical term correction.

        This method override PhonopyYaml._nac_yaml_lines.

        """
        if self._data.phonon_primitive is not None:
            assert self._data.primitive is not None
            assert isclose(self._data.primitive, self._data.phonon_primitive)
        return super()._nac_yaml_lines()

    def _displacements_yaml_lines(self, with_forces=False):
        """Get YAML lines for phonon_dataset and dataset.

        This method override PhonopyYaml._displacements_yaml_lines.
        PhonopyYaml._displacements_yaml_lines_2types is written
        to be also used by Phono3pyYaml.

        """
        lines = []
        if self._data.phonon_supercell_matrix is not None:
            lines += self._displacements_yaml_lines_2types(
                self._data.phonon_dataset,
                with_forces=with_forces,
                key_prefix="phonon_",
            )
        lines += self._displacements_yaml_lines_2types(
            self._data.dataset, with_forces=with_forces
        )
        return lines


class PhelelYaml(PhonopyYaml):
    """phelel.yaml reader and writer.

    Details are found in the docstring of PhonopyYaml.
    The common usages are as follows:

    1. Set phelel instance.
        phe_yml = PhelelYaml()
        phe_yml.set_phelel_info(phelel_instance)
    2. Read phelel.yaml file.
        phe_yml = PhelelYaml()
        phe_yml.read(filename)
    3. Parse yaml dict of phelel.yaml.
        with open("phelel.yaml", 'r') as f:
            phe_yml.yaml_data = yaml.load(f, Loader=yaml.CLoader)
            phe_yml.parse()
    4. Save stored data in PhelelYaml instance into a text file in yaml.
        with open(filename, 'w') as w:
            w.write(str(phe_yml))

    """

    default_filenames = ("phelel_disp.yaml", "phelel.yaml")
    command_name = "phelel"

    def __init__(
        self,
        configuration: dict | None = None,
        calculator: str | None = None,
        physical_units: CalculatorPhysicalUnits | None = None,
        settings: dict | None = None,
    ):
        """Init method."""
        self._data = PhelelYamlData(
            configuration=configuration,
            calculator=calculator,
            physical_units=physical_units,
        )
        self._dumper_settings = settings

    @property
    def phonon_primitive(self) -> PhonopyAtoms | None:
        """Return phonon primitive cell of phonopy calculation."""
        return self._data.phonon_primitive

    @phonon_primitive.setter
    def phonon_primitive(self, value: PhonopyAtoms):
        """Set phonon primitive cell of phonopy calculation."""
        self._data.phonon_primitive = value

    @property
    def phonon_supercell(self) -> PhonopyAtoms | None:
        """Return phonon supercell of phonopy calculation."""
        return self._data.phonon_supercell

    @phonon_supercell.setter
    def phonon_supercell(self, value: PhonopyAtoms):
        """Set phonon supercell of phonopy calculation."""
        self._data.phonon_supercell = value

    @property
    def phonon_dataset(self) -> dict | None:
        """Return phonon dataset of phonopy calculation."""
        return self._data.phonon_dataset

    @phonon_dataset.setter
    def phonon_dataset(self, value: dict):
        """Set phonon dataset of phonopy calculation."""
        self._data.phonon_dataset = value

    @property
    def phonon_supercell_matrix(self) -> NDArray | None:
        """Return phonon supercell matrix of phonopy calculation."""
        return self._data.phonon_supercell_matrix

    @phonon_supercell_matrix.setter
    def phonon_supercell_matrix(self, value: ArrayLike):
        """Set supercell matrix of phonopy calculation."""
        self._data.phonon_supercell_matrix = np.array(value, dtype="intc", order="C")

    def __str__(self):
        """Return string text of yaml output."""
        pheyml_dumper = PhelelYamlDumper(
            self._data, dumper_settings=self._dumper_settings
        )
        return "\n".join(pheyml_dumper.get_yaml_lines())

    def read(self, filename):
        """Read PhelelYaml file."""
        self._data = read_phelel_yaml(
            filename,
            configuration=self._data.configuration,
            calculator=self._data.calculator,
            physical_units=self._data.physical_units,
        )
        return self

    def set_phelel_info(self, phelel: "Phelel"):
        """Store data in Phelel instance in this instance."""
        super().set_phonon_info(cast(Phonopy, phelel))
        self._data.phonon_supercell_matrix = phelel.phonon_supercell_matrix
        self._data.phonon_dataset = phelel.phonon_dataset
        self._data.phonon_primitive = phelel.phonon_primitive
        self._data.phonon_supercell = phelel.phonon_supercell


def read_phelel_yaml(
    filename, configuration=None, calculator=None, physical_units=None
) -> PhelelYamlData:
    """Read phelel.yaml like file."""
    yaml_data = load_yaml(filename)
    if isinstance(yaml_data, str):
        msg = f'Could not load "{filename}" properly.'
        raise TypeError(msg)
    return load_phelel_yaml(
        yaml_data,
        configuration=configuration,
        calculator=calculator,
        physical_units=physical_units,
    )


def load_phelel_yaml(
    yaml_data, configuration=None, calculator=None, physical_units=None
) -> PhelelYamlData:
    """Return PhelelYamlData instance loading yaml data.

    Parameters
    ----------
    yaml_data : dict

    """
    pheyml_loader = PhelelYamlLoader(
        yaml_data,
        configuration=configuration,
        calculator=calculator,
        physical_units=physical_units,
    )
    pheyml_loader.parse()
    return pheyml_loader.data
