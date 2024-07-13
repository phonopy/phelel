"""Test for PhelelYaml class."""

from io import StringIO
from pathlib import Path

import numpy as np
import yaml
from phonopy.structure.cells import isclose

import phelel
from phelel.interface.phelel_yaml import PhelelYaml, load_phelel_yaml

cwd = Path(__file__).parent


def test_PhelelYaml_get_yaml_lines():
    """Test PhelelYaml.get_yaml_lines by C 1x1x1."""
    phe = phelel.load(cwd / ".." / "phelel_disp_NaCl111.yaml")
    phe_yml = PhelelYaml()
    phe_yml.set_phelel_info(phe)
    phe_yml_test = PhelelYaml()
    phe_yml_test._data = load_phelel_yaml(yaml.safe_load(StringIO(str(phe_yml))))
    assert isclose(phe_yml.primitive, phe_yml_test.primitive)
    assert isclose(phe_yml.unitcell, phe_yml_test.unitcell)
    assert isclose(phe_yml.supercell, phe_yml_test.supercell)
    assert isclose(phe_yml.phonon_supercell, phe_yml_test.phonon_supercell)
    assert phe_yml.version == phe_yml_test.version
    np.testing.assert_array_equal(
        phe_yml.supercell_matrix, phe_yml_test.supercell_matrix
    )
    np.testing.assert_array_equal(
        phe_yml.phonon_supercell_matrix, phe_yml_test.phonon_supercell_matrix
    )
    np.testing.assert_allclose(
        phe_yml.primitive_matrix, phe_yml_test.primitive_matrix, atol=1e-8
    )
