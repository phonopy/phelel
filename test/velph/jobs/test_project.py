"""Tests of Project, velph.toml read for the velph commands."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable

import numpy as np
import pytest
import tomli_w

from phelel.velph.config.io import VelphConfigError, parse_velph_toml
from phelel.velph.config.schema import CodeSection, KpointsMesh
from phelel.velph.errors import VelphError
from phelel.velph.jobs.project import Project, read_velph_toml


def test_read_velph_toml(nacl_velph_toml: Callable[[], dict], tmp_path: pathlib.Path):
    """velph.toml is read, and errors are VelphError.

    The errors are a missing file, which is VelphError, and a missing input set,
    an unknown key and a file that is not TOML, which are VelphConfigError.

    """
    toml_path = tmp_path / "velph.toml"
    with pytest.raises(VelphError, match="not found") as e:
        read_velph_toml(toml_path, "nac")
    assert not isinstance(e.value, VelphConfigError)

    data = nacl_velph_toml()
    toml_path.write_text(tomli_w.dumps(data))
    project = read_velph_toml(toml_path, "nac")
    assert "nac" in project.input_sets
    assert project.toml_filename == toml_path

    del data["vasp"]["nac"]
    toml_path.write_text(tomli_w.dumps(data))
    with pytest.raises(VelphConfigError) as e:
        read_velph_toml(toml_path, "nac")
    assert str(e.value) == (
        f'[vasp.nac] is not found in "{toml_path}".\n'
        f'Add [vasp.nac] to "{toml_path}", or run "velph init" again.'
    )

    data["phelel"]["unknown_key"] = 1
    toml_path.write_text(tomli_w.dumps(data))
    with pytest.raises(VelphConfigError, match="unknown_key"):
        read_velph_toml(toml_path, "relax")

    toml_path.write_text("[phelel\n")
    with pytest.raises(VelphConfigError) as e:
        read_velph_toml(toml_path, "relax")
    assert str(e.value).startswith(f'Error in reading "{toml_path}"')


def test_read_velph_toml_v014(
    nacl_velph_toml_v014: Callable[[], dict], tmp_path: pathlib.Path
):
    """velph.toml of velph 0.14 is read by the legacy reader."""
    data = nacl_velph_toml_v014()
    assert "version" in data["phelel"]
    toml_path = tmp_path / "velph.toml"
    toml_path.write_text(tomli_w.dumps(data))
    project = read_velph_toml(toml_path, "el_dos")
    assert "el_dos" in project.input_sets


def test_kpoints(nacl_velph_toml: Callable[[], dict]):
    """The meshes of all k-point files of an input set are returned.

    KPOINTS is given by the block kpoints and KPOINTS_ELPH by kpoints_dense.

    """
    data = nacl_velph_toml()
    selfenergy = data["vasp"]["selfenergy"]
    assert selfenergy["kpoints"]["mesh"] != selfenergy["kpoints_dense"]["mesh"]
    project = Project.from_config(parse_velph_toml(data))
    assert project.kpoints("selfenergy") == {
        "KPOINTS": KpointsMesh(mesh=tuple(selfenergy["kpoints"]["mesh"])),
        "KPOINTS_ELPH": KpointsMesh(mesh=tuple(selfenergy["kpoints_dense"]["mesh"])),
    }


@pytest.mark.parametrize("relax_cell", ["unitcell", "primitive"])
def test_cell_of(relax_cell: str, nacl_velph_toml: Callable[[], dict]):
    """The cell of "cell" is used for relax and nac, the primitive cell for others.

    The cells are those made when velph.toml is read, not made again.  The
    input sets of the codes are run on supercells, which are not in
    velph.toml.

    """
    data = nacl_velph_toml()
    data["vasp"]["relax"]["cell"] = relax_cell
    assert data["vasp"]["nac"]["cell"] == "primitive"
    project = Project.from_config(parse_velph_toml(data))
    assert len(project.unitcell) == 8 and len(project.primitive) == 2
    expected = project.unitcell if relax_cell == "unitcell" else project.primitive
    assert project.cell_of("relax") is expected
    assert project.cell_of("nac") is project.primitive
    for name in ("selfenergy", "el_bands", "el_dos", "ph_bands"):
        assert project.cell_of(name) is project.primitive
    with pytest.raises(ValueError):
        project.cell_of("phelel")


def test_supercell_and_primitive_matrices(nacl_velph_toml: Callable[[], dict]):
    """The matrices give the supercell and the primitive cell of the unit cell.

    The basis vectors of the primitive cell are those of the unit cell
    multiplied by the primitive matrix.

    """
    project = Project.from_config(parse_velph_toml(nacl_velph_toml()))
    np.testing.assert_array_equal(project.supercell_matrix("phelel"), np.diag([2] * 3))
    np.testing.assert_allclose(
        project.primitive.cell, (project.unitcell.cell.T @ project.primitive_matrix).T
    )
    assert not np.allclose(project.primitive_matrix, np.eye(3))


def test_code_section(nacl_velph_toml: Callable[[], dict]):
    """Fields not given take the defaults of velph init, and errors name the file.

    The supercell has to be given.

    """
    config = parse_velph_toml(nacl_velph_toml())
    assert config.phelel is not None
    config = dataclasses.replace(
        config,
        phelel=CodeSection(supercell=config.phelel.supercell),
        phonopy=None,
    )
    project = Project.from_config(config)
    section = project.code_section("phelel")
    assert (section.amplitude, section.diagonal, section.plusminus, section.nosym) == (
        0.03,
        False,
        True,
        False,
    )
    with pytest.raises(VelphConfigError) as e:
        project.code_section("phonopy")
    assert str(e.value) == (
        '[phonopy] is not found in "velph.toml".\n'
        'Add [phonopy] to "velph.toml", or run "velph init" again.'
    )
    project = Project.from_config(
        dataclasses.replace(config, phelel=CodeSection(amplitude=0.02))
    )
    with pytest.raises(VelphConfigError) as e:
        project.code_section("phelel")
    assert str(e.value) == (
        '[phelel] of "velph.toml" does not give the supercell.\n'
        'Give "supercell_dimension" or "supercell_matrix" in [phelel].'
    )
