"""Tests of reading velph.toml written by velph earlier than 0.15."""

from __future__ import annotations

import contextlib
import copy
import io
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
import tomli
import tomli_w
from click.testing import CliRunner
from phonopy.structure.atoms import PhonopyAtoms

from phelel.velph.cli.init.init import _run_init
from phelel.velph.cli.utils import VelphInitOptions
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.legacy import load_legacy_velph_toml, parse_legacy_velph_toml
from phelel.velph.config.schema import (
    CodeSection,
    Dimension,
    LinePath,
    Mesh,
    VelphConfig,
)
from phelel.velph.utils.vasp import VaspKpoints

# KPOINTS files that the generate commands of velph 0.14 write, for each input
# set and k-point block of the new format.  The paths are relative to the
# directory of velph.toml.  The perfect supercell is disp-0 with as many digits
# as the number of supercells needs, so it is found by a glob pattern.
_KPOINTS_FILES = {
    ("phelel", "kpoints"): "phelel/disp-0*/KPOINTS",
    ("phonopy", "kpoints"): "phonopy/disp-0*/KPOINTS",
    ("phono3py", "kpoints"): "phono3py/disp-0*/KPOINTS",
    ("relax", "kpoints"): "relax/iter1/KPOINTS",
    ("nac", "kpoints"): "nac/KPOINTS",
    ("selfenergy", "kpoints"): "selfenergy/KPOINTS",
    ("selfenergy", "kpoints_dense"): "selfenergy/KPOINTS_ELPH",
    ("transport", "kpoints"): "transport/KPOINTS",
    ("transport", "kpoints_dense"): "transport/KPOINTS_ELPH",
    ("ph_selfenergy", "kpoints"): "ph_selfenergy/KPOINTS",
    ("ph_selfenergy", "kpoints_dense"): "ph_selfenergy/KPOINTS_ELPH",
    ("el_bands", "kpoints"): "el_bands/bands/KPOINTS",
    ("el_dos", "kpoints"): "el_bands/dos/KPOINTS",
    ("el_dos", "kpoints_dense"): "el_bands/dos/KPOINTS_OPT",
}


def _present_velph_toml(cell: PhonopyAtoms, **options: Any) -> dict[str, Any]:
    """Return the contents of velph.toml that velph init of velph 0.14 writes."""
    with contextlib.redirect_stdout(io.StringIO()):
        toml_lines = _run_init(cell, VelphInitOptions(**options))
    assert toml_lines is not None
    return tomli.loads("\n".join(toml_lines))


def _input_set_tables(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the input set tables of a velph.toml of velph 0.14.

    The keys are the input set names of the new format.

    """
    tables = {
        set_name: table
        for set_name, table in data["vasp"].items()
        if set_name != "el_bands"
    }
    tables["el_bands"] = data["vasp"]["el_bands"]["bands"]
    tables["el_dos"] = data["vasp"]["el_bands"]["dos"]
    return tables


def test_parse_legacy_velph_toml_converts_sections(nacl_cell: PhonopyAtoms):
    """The sections of velph 0.14 are read in the new format.

    The INCAR tags and k-point meshes of each input set are those of the file,
    [vasp.el_bands.bands] and [vasp.el_bands.dos] are read as [vasp.el_bands]
    and [vasp.el_dos], and data is not modified.

    """
    data = _present_velph_toml(nacl_cell, supercell_dimension=(2, 2, 2))
    original = copy.deepcopy(data)
    assert "fft_mesh" in data["phelel"]
    config = parse_legacy_velph_toml(data)
    assert data == original

    assert config.version == data["phelel"]["version"]
    assert config.phelel == CodeSection(
        supercell=Dimension(supercell_dimension=(2, 2, 2)),
        amplitude=0.03,
        diagonal=False,
        plusminus=True,
    )
    assert config.vasp is not None and config.vasp.input_sets is not None
    assert config.vasp.incar is None
    tables = _input_set_tables(data)
    assert set(config.vasp.input_sets) == set(tables)
    for set_name, table in tables.items():
        input_set = config.vasp.input_sets[set_name]
        assert input_set.incar == table["incar"], set_name
        for block in ("kpoints", "kpoints_dense"):
            if block in table:
                mesh = getattr(input_set, block)
                assert isinstance(mesh, Mesh), (set_name, block)
                assert list(mesh.mesh) == table[block]["mesh"], (set_name, block)
    assert config.vasp.input_sets["relax"].cell == "unitcell"
    assert config.vasp.input_sets["nac"].cell == "primitive"
    assert config.vasp.input_sets["el_bands"].kpoints_opt == LinePath(line=51)
    assert config.vasp.input_sets["ph_bands"].qpoints == LinePath(line=51)


def test_parse_legacy_velph_toml_without_supercell(nacl_cell: PhonopyAtoms):
    """[phelel] that has only version is not read as a section."""
    data = _present_velph_toml(nacl_cell)
    assert data["phelel"] == {"version": data["phelel"]["version"]}
    config = parse_legacy_velph_toml(data)
    assert config.version == data["phelel"]["version"]
    assert config.phelel is None


def test_parse_legacy_velph_toml_removed_keys(nacl_cell: PhonopyAtoms):
    """Keys that the new format does not have are not read.

    These are [vasp.incar], phonon_supercell_dimension, [vasp.phelel.phonon]
    and [vasp.phono3py.phonon].  [vasp.relax] and [vasp.nac] without "cell" get
    the cells that the generate commands of velph 0.14 use, and "cell" is read
    in lower case.

    """
    data = _present_velph_toml(nacl_cell, supercell_dimension=(2, 2, 2))
    data["vasp"]["incar"] = {"encut": 300}
    for code in ("phelel", "phono3py"):
        data[code]["phonon_supercell_dimension"] = [3, 3, 3]
        data["vasp"][code]["phonon"] = copy.deepcopy(data["vasp"][code])
    del data["vasp"]["relax"]["cell"]
    data["vasp"]["nac"]["cell"] = "Unitcell"
    config = parse_legacy_velph_toml(data)
    assert config.vasp is not None and config.vasp.input_sets is not None
    assert config.vasp.incar is None
    assert config.vasp.input_sets["relax"].cell == "unitcell"
    assert config.vasp.input_sets["nac"].cell == "unitcell"


def _velph_toml_with_kspacing(data: dict[str, Any]) -> dict[str, Any]:
    """Return data with kspacing in the k-point blocks that velph 0.14 converts.

    "velph ph_bands generate" of velph 0.14 does not convert kspacing, so the
    mesh of [vasp.ph_bands.kpoints] is kept.

    """
    data = copy.deepcopy(data)
    for set_name, table in _input_set_tables(data).items():
        if set_name == "ph_bands":
            continue
        table["kpoints"] = {"kspacing": 0.15}
        if "kpoints_dense" in table:
            table["kpoints_dense"] = {"kspacing": 0.08}
    return data


# Commands of velph 0.14 that write the KPOINTS files of each input set.
# "velph el_bands generate" writes those of el_bands and el_dos.
_GENERATE_COMMANDS = {
    "phelel": (["phelel", "init"], ["phelel", "generate"]),
    "phonopy": (["phonopy", "init"], ["phonopy", "generate"]),
    "phono3py": (["phono3py", "init", "--rd", "1"], ["phono3py", "generate"]),
    "relax": (["relax", "generate"],),
    "nac": (["nac", "generate"],),
    "selfenergy": (["selfenergy", "generate", "--dry-run"],),
    "transport": (["transport", "generate", "--dry-run"],),
    "ph_selfenergy": (["ph_selfenergy", "generate", "--dry-run"],),
    "el_bands": (["el_bands", "generate"],),
    "el_dos": (["el_bands", "generate"],),
}


def _run_generate_commands(set_names: tuple[str, ...]) -> None:
    """Run the commands of velph 0.14 that write the input sets.

    The commands are run in the current directory.

    """
    commands: list[list[str]] = []
    for set_name in set_names:
        for args in _GENERATE_COMMANDS[set_name]:
            if args not in commands:
                commands.append(args)
    runner = CliRunner()
    for args in commands:
        result = runner.invoke(cmd_root, args)
        assert result.exit_code == 0, (args, result.output)
        assert result.exception is None, (args, result.exception)


def _kpoints_text(mesh: Mesh, path: pathlib.Path) -> str:
    """Return KPOINTS written from mesh in the same way as the generate commands."""
    kpoints: dict[str, Any] = {"mesh": mesh.mesh}
    if mesh.shift is not None:
        kpoints["shift"] = mesh.shift
    VaspKpoints.write_mesh_mode(path, kpoints)
    return path.read_text()


def _check_meshes_equal_generate(
    config: VelphConfig, set_names: tuple[str, ...], tmp_path: pathlib.Path
) -> None:
    assert config.vasp is not None and config.vasp.input_sets is not None
    for (set_name, block), pattern in _KPOINTS_FILES.items():
        if set_name not in set_names:
            continue
        mesh = getattr(config.vasp.input_sets[set_name], block)
        assert isinstance(mesh, Mesh), (set_name, block)
        written = sorted(tmp_path.glob(pattern))
        assert written, f'"{pattern}" was not written.'
        expected = _kpoints_text(mesh, tmp_path / "expected_KPOINTS")
        assert written[0].read_text() == expected, (set_name, block)


def test_kspacing_mesh_equals_generate(
    nacl_cell: PhonopyAtoms,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The mesh converted from kspacing is the one that generate writes.

    The supercells are made from [unitcell] and the supercell matrices, and
    the generate commands read them from phelel_disp.yaml, phonopy_disp.yaml
    and phono3py_disp.yaml.  The meshes on the unit cell, the primitive cell
    and the supercells of NaCl differ from each other, and the mesh on the
    primitive cell is a generalized regular grid.

    """
    set_names = tuple(_GENERATE_COMMANDS)
    data = _velph_toml_with_kspacing(
        _present_velph_toml(nacl_cell, supercell_dimension=(2, 2, 2))
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(data))
    _run_generate_commands(set_names)
    config = load_legacy_velph_toml(tmp_path / "velph.toml")
    _check_meshes_equal_generate(config, set_names, tmp_path)

    assert config.vasp is not None and config.vasp.input_sets is not None
    meshes = {
        set_name: config.vasp.input_sets[set_name].kpoints
        for set_name in ("phelel", "relax", "nac")
    }
    assert meshes["phelel"] == Mesh(mesh=(4, 4, 4))
    assert meshes["relax"] == Mesh(mesh=(8, 8, 8))
    assert meshes["nac"] == Mesh(mesh=((-8, 8, 8), (8, -8, 8), (8, 8, -8)))


@pytest.mark.parametrize("split_site_mixture", [False, True])
def test_kspacing_mesh_equals_generate_site_mixture(
    site_mixture_velph_toml: Callable[[bool], dict],
    split_site_mixture: bool,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The mesh converted from kspacing with site mixture is that of generate.

    With the merge scheme, the supercell of phelel and phonopy is made from the
    unit cell whose co-located atoms are merged into sites.  phono3py, el_bands
    and el_dos are not checked.  phono3py does not support site mixture.  "velph
    el_bands generate" of velph 0.14 writes el_bands and el_dos, and it stops
    with "Problem creating primitive cell" for this cell.

    For the cells tried, the mesh on the supercell made with the merge scheme
    is the same as the mesh on the supercell made without it.  So this test
    does not find a loader that forgets to merge the atoms.

    """
    set_names = tuple(
        set_name
        for set_name in _GENERATE_COMMANDS
        if set_name not in ("phono3py", "el_bands", "el_dos")
    )
    data = _velph_toml_with_kspacing(site_mixture_velph_toml(split_site_mixture))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(data))
    _run_generate_commands(set_names)
    config = load_legacy_velph_toml(tmp_path / "velph.toml")
    _check_meshes_equal_generate(config, set_names, tmp_path)
