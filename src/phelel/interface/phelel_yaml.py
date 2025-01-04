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

import dataclasses
from typing import TYPE_CHECKING, Optional

import numpy as np
from phonopy.interface.phonopy_yaml import (
    PhonopyYaml,
    PhonopyYamlDumper,
    PhonopyYamlLoader,
    load_yaml,
    phonopy_yaml_property_factory,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive, Supercell, isclose
from phonopy.structure.symmetry import Symmetry

if TYPE_CHECKING:
    from phelel import Phelel


@dataclasses.dataclass
class PhelelYamlData:
    """PhonopyYaml data structure."""

    configuration: Optional[dict] = None
    calculator: Optional[str] = None
    physical_units: Optional[dict] = None
    unitcell: Optional[PhonopyAtoms] = None
    primitive: Optional[Primitive] = None
    supercell: Optional[Supercell] = None
    dataset: Optional[dict] = None
    supercell_matrix: Optional[np.ndarray] = None
    primitive_matrix: Optional[np.ndarray] = None
    nac_params: Optional[dict] = None
    force_constants: Optional[np.ndarray] = None
    symmetry: Optional[Symmetry] = None  # symmetry of supercell
    frequency_unit_conversion_factor: Optional[float] = None
    version: Optional[str] = None
    command_name: str = "phelel"

    phonon_supercell_matrix: Optional[np.ndarray] = None
    phonon_dataset: Optional[dict] = None
    phonon_supercell: Optional[Supercell] = None
    phonon_primitive: Optional[Primitive] = None


class PhelelYamlLoader(PhonopyYamlLoader):
    """PhelelYaml loader."""

    def __init__(
        self, yaml_data, configuration=None, calculator=None, physical_units=None
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

    def __init__(self, data: PhelelYamlData, dumper_settings=None):
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

    configuration = phonopy_yaml_property_factory("configuration")
    calculator = phonopy_yaml_property_factory("calculator")
    physical_units = phonopy_yaml_property_factory("physical_units")
    unitcell = phonopy_yaml_property_factory("unitcell")
    primitive = phonopy_yaml_property_factory("primitive")
    supercell = phonopy_yaml_property_factory("supercell")
    dataset = phonopy_yaml_property_factory("dataset")
    supercell_matrix = phonopy_yaml_property_factory("supercell_matrix")
    primitive_matrix = phonopy_yaml_property_factory("primitive_matrix")
    nac_params = phonopy_yaml_property_factory("nac_params")
    force_constants = phonopy_yaml_property_factory("force_constants")
    symmetry = phonopy_yaml_property_factory("symmetry")
    frequency_unit_conversion_factor = phonopy_yaml_property_factory(
        "frequency_unit_conversion_factor"
    )
    version = phonopy_yaml_property_factory("version")

    phonon_supercell_matrix = phonopy_yaml_property_factory("phonon_supercell_matrix")
    phonon_dataset = phonopy_yaml_property_factory("phonon_dataset")
    phonon_supercell = phonopy_yaml_property_factory("phonon_supercell")
    phonon_primitive = phonopy_yaml_property_factory("phonon_primitive")

    def __init__(
        self, configuration=None, calculator=None, physical_units=None, settings=None
    ):
        """Init method."""
        self._data = PhelelYamlData(
            configuration=configuration,
            calculator=calculator,
            physical_units=physical_units,
        )
        self._dumper_settings = settings

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
        super().set_phonon_info(phelel)
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
