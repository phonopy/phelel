"""Tests of phelel command line script."""

from __future__ import annotations

import os
import pathlib
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, fields

import h5py
import pytest

import phelel.cui.phelel_script as phelel_script
from phelel import load
from phelel.cui.phelel_argparse import get_parser
from phelel.cui.phelel_script import main
from phelel.interface.phelel_yaml import PhelelYaml

cwd = pathlib.Path(__file__).parent
cwd_called = pathlib.Path.cwd()


@dataclass
class MockArgs:
    """Mock args of ArgumentParser."""

    filename: Sequence[str] | None = None
    conf_filename: str | None = None
    log_level: int | None = None
    cell_filename: str | None = None
    supercell_dimension: str | None = None
    create_derivatives: Sequence[str] | None = None
    is_displacement: bool = False
    is_plusminus_displacements: bool = False
    fft_mesh_numbers: str | None = None

    def __iter__(self):
        """Make self iterable to support in."""
        return (getattr(self, field.name) for field in fields(self))

    def __contains__(self, item):
        """Implement in operator."""
        return item in (field.name for field in fields(self))


@pytest.mark.parametrize(
    "filename,with_BORN_file",
    [
        (pathlib.Path("..") / "phelel_disp_C111.yaml", False),
        (pathlib.Path("..") / "phelel_disp_C111.yaml", True),
        (pathlib.Path("phelel_disp_C111_nac.yaml"), False),
    ],
)
def test_phelel_script(filename: pathlib.Path, with_BORN_file: bool):
    """Test phelel command."""
    born_str = """4.399652
5.7 0 0 0 5.7 0 0 0 5.7
0 0 0 0 0 0 0 0 0"""

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        if with_BORN_file:
            born_file = pathlib.Path("BORN")
            with born_file.open("w", encoding="utf-8") as f:
                f.write(born_str)
        else:
            # Test not to read BORN directory.
            born_dir = pathlib.Path("BORN")
            os.mkdir(born_dir)

        try:
            # Check sys.exit(0)
            cell_filename = str(cwd / filename)
            argparse_control = _get_argparse_control("run", cell_filename=cell_filename)
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if with_BORN_file:
                born_file.unlink()
            else:
                born_dir.rmdir()

            # Clean files created by phelel script.
            for created_filename in ("phelel.yaml",):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("with_BORN_file", [True, False])
def test_phelel_script_create_derivatives(with_BORN_file: bool):
    """Test phelel command with --cd.

    With ``phelel_disp_C111.yaml`` as input, the derivatives are calculated
    and ``phelel_params.hdf5`` is created.

    """
    born_str = """4.399652
5.7 0 0 0 5.7 0 0 0 5.7
0 0 0 0 0 0 0 0 0"""

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        if with_BORN_file:
            born_file = pathlib.Path("BORN")
            with born_file.open("w", encoding="utf-8") as f:
                f.write(born_str)
        else:
            # Test not to read BORN directory.
            born_dir = pathlib.Path("BORN")
            os.mkdir(born_dir)

        try:
            # Check sys.exit(0)
            dirname = cwd / ".." / "interface" / "vasp"
            dispdirs = [str(dirname / "C111_disp-000"), str(dirname / "C111_disp-001")]
            argparse_control = _get_argparse_control(
                "run",
                cell_filename=str(dirname / "phelel_disp_C111.yaml"),
                create_derivatives=dispdirs,
                fft_mesh_numbers="1 1 1",
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            _check_create_params_file(with_BORN_file)

            for created_filename in ("phelel_params.hdf5",):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            if with_BORN_file:
                born_file.unlink()
            else:
                born_dir.rmdir()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phelel_script_create_derivatives_nac_from_phelel_disp_yaml(
    tmp_path: pathlib.Path,
):
    """NAC parameters are read from phelel_disp.yaml without BORN."""
    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        dirname = cwd / ".." / "interface" / "vasp"
        argparse_control = _get_argparse_control(
            "run",
            cell_filename=str(cwd / "phelel_disp_C111_nac.yaml"),
            create_derivatives=[
                str(dirname / "C111_disp-000"),
                str(dirname / "C111_disp-001"),
            ],
            fft_mesh_numbers="1 1 1",
        )
        with pytest.raises(SystemExit) as excinfo:
            main(**argparse_control)
        assert excinfo.value.code == 0
        with h5py.File("phelel_params.hdf5") as f:
            assert "born_effective_charges" in f
            assert "dielectric_constant" in f
    finally:
        os.chdir(original_cwd)


@pytest.mark.parametrize("is_plusminus_displacements", [True, False])
def test_phelel_script_create_displacements(is_plusminus_displacements: bool):
    """Test phelel-init command for creating displacements."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            dirname = cwd / ".." / "interface" / "vasp"
            cell_filename = str(dirname / "POSCAR-unitcell_C111")
            supercell_dimension = "1 1 1"

            argparse_control = _get_argparse_control(
                "init",
                cell_filename=cell_filename,
                supercell_dimenstion=supercell_dimension,
                is_displacement=True,
                is_plusminus_displacements=is_plusminus_displacements,
            )

            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if is_plusminus_displacements:
                created_filenames = (
                    "phelel_disp.yaml",
                    "SPOSCAR",
                    "POSCAR-001",
                    "POSCAR-002",
                )
                for created_filename in ("POSCAR-003", "POSCAR_PH-003"):
                    file_path = pathlib.Path(created_filename)
                    assert not file_path.exists()
            else:
                created_filenames = (
                    "phelel_disp.yaml",
                    "SPOSCAR",
                    "POSCAR-001",
                )
                for created_filename in ("POSCAR-002",):
                    file_path = pathlib.Path(created_filename)
                    assert not file_path.exists()

            for created_filename in created_filenames:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("mode", ["init", "run"])
def test_phelel_script_parser(mode: str, monkeypatch: pytest.MonkeyPatch):
    """Each command builds the parser of its mode."""

    class _ParserCreated(Exception):
        pass

    calls = []

    def _get_parser(mode):
        calls.append(mode)
        raise _ParserCreated

    monkeypatch.setattr(phelel_script, "get_parser", _get_parser)
    with pytest.raises(_ParserCreated):
        main(mode=mode)
    assert calls == [mode]


@pytest.mark.parametrize(
    "options",
    [
        ["-d"],
        ["--disp"],
        ["-c", "POSCAR"],
        ["--dim", "2", "2", "2"],
        ["--dim-phonon", "2", "2", "2"],
        ["--amplitude", "0.03"],
        ["--pm"],
        ["--nodiag"],
    ],
)
def test_phelel_parser_rejects_init_options(options: list[str], capsys):
    """The phelel command tells that an option of phelel-init moved there."""
    parser = get_parser("run")
    parser.prog = "phelel"
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(options)
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert f"'{options[0]}' is an option of 'phelel-init'" in err
    assert "Replace 'phelel' with 'phelel-init' in your command." in err


@pytest.mark.parametrize(
    "mode,kwargs,message",
    [
        ("init", {}, "'phelel-init' generates supercells with displacements"),
        (
            "init",
            {"is_displacement": True, "create_derivatives": ["disp-000"]},
            "'--cd' calculates the derivatives",
        ),
        ("run", {"is_displacement": True}, "'-d' generates supercells"),
        ("run", {"create_derivatives": ["disp-000"]}, "'--cd' needs '--fft-mesh'"),
    ],
)
def test_phelel_script_operation_of_wrong_command(
    mode: str, kwargs: dict, message: str, capsys
):
    """An operation that does not belong to the command is an error."""
    argparse_control = _get_argparse_control(mode, **kwargs)
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 1
    assert message in capsys.readouterr().out


def test_phelel_parser_config_option():
    """The phelel command takes a configuration file by --config."""
    args = get_parser("run").parse_args(["--config", "phelel.conf"])
    assert args.conf_filename == "phelel.conf"
    assert args.filename == []


def test_phelel_script_reads_config_file(tmp_path: pathlib.Path, capsys):
    """FFT_MESH in the file given by --config is used by --cd."""
    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        pathlib.Path("phelel.conf").write_text("FFT_MESH = 1 1 1\n")
        dirname = cwd / ".." / "interface" / "vasp"
        argparse_control = _get_argparse_control(
            "run",
            filename=[str(dirname / "phelel_disp_C111.yaml")],
            conf_filename="phelel.conf",
            create_derivatives=[
                str(dirname / "C111_disp-000"),
                str(dirname / "C111_disp-001"),
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            main(**argparse_control)
        assert excinfo.value.code == 0
        assert (tmp_path / "phelel_params.hdf5").exists()
        out = capsys.readouterr().out
        assert '"phelel.conf" was read as phelel configuration file.' in out
    finally:
        os.chdir(original_cwd)


def test_phelel_script_rejects_config_file_as_argument(tmp_path: pathlib.Path, capsys):
    """A configuration file given as the argument of phelel is an error.

    phelel_disp.yaml in the current directory must not be read in place of it.

    """
    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        (tmp_path / "phelel_disp.yaml").write_text(
            (cwd / ".." / "phelel_disp_C111.yaml").read_text()
        )
        pathlib.Path("phelel.conf").write_text("FFT_MESH = 1 1 1\n")
        argparse_control = _get_argparse_control("run", filename=["phelel.conf"])
        with pytest.raises(SystemExit) as excinfo:
            main(**argparse_control)
        assert excinfo.value.code == 1
        assert not (tmp_path / "phelel.yaml").exists()
        out = capsys.readouterr().out
        assert '"phelel.conf" could not be read as a phelel.yaml-like file.' in out
        assert 'If "phelel.conf" is a configuration file, give it by --config.' in out
    finally:
        os.chdir(original_cwd)


@pytest.mark.parametrize(
    "yaml_filename,ref_phonon_smat,message",
    [
        (
            pathlib.Path("..")
            / "velph"
            / "cli"
            / "phelel"
            / "differentiate"
            / "phelel_disp_C111-222.yaml",
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "But that in",
        ),
        (
            pathlib.Path("..") / "phelel_disp_C111.yaml",
            None,
            "has no phonon supercell matrix, so",
        ),
    ],
)
def test_phelel_script_dim_phonon_of_yaml_is_used(
    yaml_filename: pathlib.Path,
    ref_phonon_smat: list | None,
    message: str,
    tmp_path: pathlib.Path,
    capsys,
):
    """DIM_PHONON in --config does not replace that of phelel_disp.yaml."""
    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        pathlib.Path("phelel.conf").write_text("DIM_PHONON = 3 3 3\n")
        argparse_control = _get_argparse_control(
            "run", filename=[str(cwd / yaml_filename)], conf_filename="phelel.conf"
        )
        with pytest.raises(SystemExit) as excinfo:
            main(**argparse_control)
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "Phonon supercell matrix given by DIM_PHONON is" in out
        assert message in out
        phe_yml = PhelelYaml().read(tmp_path / "phelel.yaml")
        if ref_phonon_smat is None:
            assert phe_yml.phonon_supercell_matrix is None
        else:
            assert phe_yml.phonon_supercell_matrix.tolist() == ref_phonon_smat
    finally:
        os.chdir(original_cwd)


def test_phelel_script_config_file_not_found(tmp_path: pathlib.Path, capsys):
    """A file given by --config that does not exist is an error."""
    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        argparse_control = _get_argparse_control("run", conf_filename="phelel.conf")
        with pytest.raises(SystemExit) as excinfo:
            main(**argparse_control)
        assert excinfo.value.code == 1
        assert '"phelel.conf" was not found.' in capsys.readouterr().out
    finally:
        os.chdir(original_cwd)


def test_phelel_load_is_deprecated(tmp_path: pathlib.Path, capsys):
    """phelel-load works as phelel and warns that it is deprecated."""
    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        argparse_control = _get_argparse_control(
            "run", cell_filename=str(cwd / ".." / "phelel_disp_C111.yaml")
        )
        with pytest.raises(SystemExit) as excinfo:
            main(**argparse_control, deprecated_command="phelel-load")
        assert excinfo.value.code == 0
        assert (tmp_path / "phelel.yaml").exists()
        out = capsys.readouterr().out
        assert "WARNING: 'phelel-load' is deprecated. Use 'phelel' instead." in out
    finally:
        os.chdir(original_cwd)


def test_phelel_loader_is_nac_false(tmp_path: pathlib.Path):
    """With is_nac=False, BORN in the current directory is not read."""
    born_str = """4.399652
5.7 0 0 0 5.7 0 0 0 5.7
0 0 0 0 0 0 0 0 0"""

    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        pathlib.Path("BORN").write_text(born_str, encoding="utf-8")
        dirname = cwd / ".." / "interface" / "vasp"
        cell_filename = str(dirname / "phelel_disp_C111.yaml")
        dispdirs = [str(dirname / "C111_disp-000"), str(dirname / "C111_disp-001")]
        phe = load(cell_filename, dir_names=dispdirs, is_nac=False)
        assert phe.nac_params is None
    finally:
        os.chdir(original_cwd)


@pytest.mark.parametrize("with_BORN_file", [True, False])
def test_phelel_loader_with_reading_BORN(with_BORN_file: bool):
    """Test phelel loader with reading BORN file."""
    born_str = """4.399652
5.7 0 0 0 5.7 0 0 0 5.7
0 0 0 0 0 0 0 0 0"""

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        if with_BORN_file:
            born_file = pathlib.Path("BORN")
            with born_file.open("w", encoding="utf-8") as f:
                f.write(born_str)

        dirname = cwd / ".." / "interface" / "vasp"
        cell_filename = str(dirname / "phelel_disp_C111.yaml")
        dispdirs = [str(dirname / "C111_disp-000"), str(dirname / "C111_disp-001")]
        phe = load(cell_filename, dir_names=dispdirs, fft_mesh=[1, 1, 1], log_level=2)

        if with_BORN_file:
            assert phe.nac_params is not None
            assert "born" in phe.nac_params
        else:
            assert phe.nac_params is None

        phe.save_hdf5(filename="phelel_params.hdf5")
        _check_create_params_file(with_BORN_file)

        file_path = pathlib.Path("phelel_params.hdf5")
        assert file_path.exists()
        file_path.unlink()

        if with_BORN_file:
            born_file.unlink()

        _check_no_files()

        os.chdir(original_cwd)


def _check_create_params_file(with_BORN_file: bool):
    """Check created phelel_params.hdf5 file."""
    if with_BORN_file:
        # extra keys: 'born_effective_charges' and 'dielectric_constant'
        ref_keys = set(
            [
                "Dij",
                "FFT_mesh",
                "atom_indices_in_derivatives",
                "born_effective_charges",
                "dDijdu",
                "dVdu",
                "dielectric_constant",
                "direct_rotations",
                "displacements_atom_indices",
                "displacements_vectors",
                "dqijdu",
                "force_constants",
                "grid_point",
                "lattice_point",
                "p2s_map",
                "primitive_lattice",
                "primitive_masses",
                "primitive_matrix",
                "primitive_numbers",
                "primitive_positions",
                "qij",
                "s2p_map",
                "shortest_vector_multiplicities",
                "shortest_vectors",
                "spacegroup_number",
                "supercell_lattice",
                "supercell_masses",
                "supercell_matrix",
                "supercell_numbers",
                "supercell_positions",
                "transformation_matrix",
                "unitcell_lattice",
                "unitcell_masses",
                "unitcell_numbers",
                "unitcell_positions",
            ]
        )
    else:
        ref_keys = set(
            [
                "Dij",
                "FFT_mesh",
                "atom_indices_in_derivatives",
                "dDijdu",
                "dVdu",
                "direct_rotations",
                "displacements_atom_indices",
                "displacements_vectors",
                "dqijdu",
                "force_constants",
                "grid_point",
                "lattice_point",
                "p2s_map",
                "primitive_lattice",
                "primitive_masses",
                "primitive_matrix",
                "primitive_numbers",
                "primitive_positions",
                "qij",
                "s2p_map",
                "shortest_vector_multiplicities",
                "shortest_vectors",
                "spacegroup_number",
                "supercell_lattice",
                "supercell_masses",
                "supercell_matrix",
                "supercell_numbers",
                "supercell_positions",
                "transformation_matrix",
                "unitcell_lattice",
                "unitcell_masses",
                "unitcell_numbers",
                "unitcell_positions",
            ]
        )

    with h5py.File("phelel_params.hdf5") as f:
        assert set(f) == ref_keys


def _get_argparse_control(
    mode: str,
    cell_filename: str | None = None,
    supercell_dimenstion: str | None = None,
    create_derivatives: Sequence[str] | None = None,
    is_displacement: bool = False,
    is_plusminus_displacements: bool = False,
    fft_mesh_numbers: str | None = None,
    filename: Sequence[str] = (),
    conf_filename: str | None = None,
):
    # Mock of ArgumentParser.args.
    mockargs = MockArgs(
        filename=list(filename),
        conf_filename=conf_filename,
        log_level=1,
        cell_filename=str(cell_filename),
        supercell_dimension=supercell_dimenstion,
        create_derivatives=create_derivatives,
        is_displacement=is_displacement,
        is_plusminus_displacements=is_plusminus_displacements,
        fft_mesh_numbers=fft_mesh_numbers,
    )
    return {"args": mockargs, "mode": mode}


def _ls():
    current_dir = pathlib.Path(".")
    for file in current_dir.iterdir():
        print(file.name)


def _check_no_files():
    assert not list(pathlib.Path(".").iterdir())
