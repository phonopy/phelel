"""Tests CLIs."""

import pathlib
import shutil
from collections.abc import Callable, Sequence

import click
import h5py
import numpy as np
import pytest
import tomli_w
from click.testing import CliRunner
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

import phelel
import phelel.velph.cli.phelel.differentiate as differentiate
from phelel import Phelel
from phelel.velph.cli.phelel.differentiate import get_fft_mesh, run_derivatives
from phelel.velph.cli.phelel.init import run_init
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.io import parse_velph_toml
from phelel.velph.config.schema import InputSetSection, VaspSection, VelphConfig
from phelel.velph.jobs.project import Project
from phelel.velph.jobs.project_directory import ProjectDirectory
from phelel.velph.utils.fft_mesh import get_vasp_fft_mesh

cwd = pathlib.Path(__file__).parent


def test_run_derivatives(tmp_path: pathlib.Path):
    """Test of run_derivatives.

    This test just checks the creation of hdf5 file and go through this command.

    """
    phe = phelel.load(cwd / "C111" / "phelel_disp_C111.yaml", fft_mesh=[9, 9, 9])
    run_derivatives(phe, project_dir=ProjectDirectory(cwd / "C111"))

    hdf5_filename = tmp_path / "phelel_params.hdf5"
    phe.save_hdf5(filename=hdf5_filename)
    with h5py.File(hdf5_filename) as f:
        assert set(f) == set(
            (
                "Dij",
                "FFT_mesh",
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
            )
        )


def test_run_derivatives_with_wrong_supercell_matrix():
    """Test of run_derivatives.

    This test just checks the creation of hdf5 file and go through this command.
    Supercell matrix is inconsistent. Therefore it should raise an error.

    """
    phe = phelel.load(cwd / "phelel_disp_C222.yaml", fft_mesh=[9, 9, 9])
    with pytest.raises(ValueError):
        run_derivatives(phe, project_dir=ProjectDirectory(cwd / "C111"))


def test_run_derivatives_without_phonopy_disp_yaml(tmp_path: pathlib.Path):
    """With the phonon supercell, phonopy_disp.yaml of velph phonopy is needed."""
    phe = phelel.load(cwd / "phelel_disp_C111-222.yaml", fft_mesh=[9, 9, 9])
    with pytest.raises(click.ClickException) as e:
        run_derivatives(phe, project_dir=ProjectDirectory(cwd / "C111"))
    assert 'phonopy_disp.yaml" does not exist' in e.value.message


def _init_phonon_from_phonopy(
    phonopy_lines: Sequence[str],
    ti_velph_toml: Callable[[Sequence[str]], dict],
    code_init_project: Callable[[dict], Project],
    write_phonopy_disp_yaml: Callable[[Project, pathlib.Path], Phonopy],
    tmp_path: pathlib.Path,
) -> tuple[Phelel, list[pathlib.Path], list[pathlib.Path]]:
    """Return Phelel with the phonon supercells of phonopy and the directories.

    The directories, phelel/disp-* with an empty vaspout.h5 and phonopy/disp-*,
    are made in tmp_path.

    """
    project = code_init_project(ti_velph_toml(phonopy_lines))
    write_phonopy_disp_yaml(project, tmp_path)
    phe = run_init(
        project, project_dir=ProjectDirectory(tmp_path), phonon_from_phonopy=True
    )
    assert phe.supercells_with_displacements is not None
    assert phe.phonon_supercells_with_displacements is not None
    dir_names = []
    for code, cells in (
        ("phelel", phe.supercells_with_displacements),
        ("phonopy", phe.phonon_supercells_with_displacements),
    ):
        dir_names.append(ProjectDirectory(tmp_path).disp_directories(code, len(cells)))
        for dir_name in dir_names[-1]:
            dir_name.mkdir(parents=True)
            (dir_name / "vaspout.h5").touch()
    return phe, dir_names[0], dir_names[1]


def test_run_derivatives_phonon_from_phonopy(
    ti_velph_toml: Callable[[Sequence[str]], dict],
    code_init_project: Callable[[dict], Project],
    write_phonopy_disp_yaml: Callable[[Project, pathlib.Path], Phonopy],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The forces of the phonon supercells are those of phonopy/disp-*."""
    phe, phelel_dir_names, phonopy_dir_names = _init_phonon_from_phonopy(
        ["amplitude = 0.01"],
        ti_velph_toml,
        code_init_project,
        write_phonopy_disp_yaml,
        tmp_path,
    )
    calls = []

    def _create_derivatives(phe, dir_names, subtract_rfs=False, log_level=0):
        calls.append(dir_names)

    monkeypatch.setattr(differentiate, "create_derivatives", _create_derivatives)
    run_derivatives(phe, project_dir=ProjectDirectory(tmp_path))
    assert calls == [phelel_dir_names + phonopy_dir_names]


@pytest.mark.parametrize(
    "phonopy_lines", [["amplitude = 0.02"], ["number_of_snapshots = 2"]]
)
def test_run_derivatives_phonopy_disp_yaml_changed(
    phonopy_lines: list[str],
    ti_velph_toml: Callable[[Sequence[str]], dict],
    code_init_project: Callable[[dict], Project],
    write_phonopy_disp_yaml: Callable[[Project, pathlib.Path], Phonopy],
    tmp_path: pathlib.Path,
):
    """phonopy_disp.yaml written again after velph phelel init is an error.

    The supercells of phonopy/disp-* are then not the phonon supercells of
    phelel, with the same number of them (amplitude) or not.

    """
    phe, _, _ = _init_phonon_from_phonopy(
        ["amplitude = 0.01"],
        ti_velph_toml,
        code_init_project,
        write_phonopy_disp_yaml,
        tmp_path,
    )
    write_phonopy_disp_yaml(code_init_project(ti_velph_toml(phonopy_lines)), tmp_path)
    with pytest.raises(click.ClickException) as e:
        run_derivatives(phe, project_dir=ProjectDirectory(tmp_path))
    assert "differ from those of" in e.value.message


def test_run_derivatives_without_phonopy_directory(
    ti_velph_toml: Callable[[Sequence[str]], dict],
    code_init_project: Callable[[dict], Project],
    write_phonopy_disp_yaml: Callable[[Project, pathlib.Path], Phonopy],
    tmp_path: pathlib.Path,
):
    """A missing directory of the phonon supercells is named."""
    phe, _, phonopy_dir_names = _init_phonon_from_phonopy(
        ["amplitude = 0.01"],
        ti_velph_toml,
        code_init_project,
        write_phonopy_disp_yaml,
        tmp_path,
    )
    shutil.rmtree(phonopy_dir_names[-1])
    with pytest.raises(click.ClickException) as e:
        run_derivatives(phe, project_dir=ProjectDirectory(tmp_path))
    assert e.value.message == f'"{phonopy_dir_names[-1]}" does not exist.'


def test_run_derivatives_without_force_files(
    ti_velph_toml: Callable[[Sequence[str]], dict],
    code_init_project: Callable[[dict], Project],
    write_phonopy_disp_yaml: Callable[[Project, pathlib.Path], Phonopy],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """FileNotFoundError of a missing force file is click.ClickException."""
    phe, _, _ = _init_phonon_from_phonopy(
        ["amplitude = 0.01"],
        ti_velph_toml,
        code_init_project,
        write_phonopy_disp_yaml,
        tmp_path,
    )

    def _create_derivatives(phe, dir_names, subtract_rfs=False, log_level=0):
        raise FileNotFoundError('Neither "vaspout.h5" nor "vasprun.xml" is found.')

    monkeypatch.setattr(differentiate, "create_derivatives", _create_derivatives)
    with pytest.raises(click.ClickException, match="Neither"):
        run_derivatives(phe, project_dir=ProjectDirectory(tmp_path))


def test_run_derivatives_site_mixture_merge(
    site_mixture_velph_toml: Callable[[bool], dict],
    tmp_path: pathlib.Path,
):
    """Derivatives are not calculated with the merge scheme of site mixture."""
    phe = run_init(
        Project.from_config(parse_velph_toml(site_mixture_velph_toml(False))),
        project_dir=ProjectDirectory(tmp_path),
    )
    with pytest.raises(click.ClickException) as e:
        run_derivatives(phe, project_dir=ProjectDirectory(tmp_path))
    assert "not supported with the merge scheme" in e.value.message


def _elph_project(incars: dict[str, dict]) -> Project:
    """Return Project with the electron-phonon input sets of incars.

    The cells are not used by get_fft_mesh, and one atom is given.

    """
    config = VelphConfig(
        vasp=VaspSection(
            input_sets={
                name: InputSetSection(incar=incar) for name, incar in incars.items()
            }
        )
    )
    cell = PhonopyAtoms(cell=_LATTICE, symbols=["C"], scaled_positions=[[0, 0, 0]])
    return Project(config, pathlib.Path("velph.toml"), unitcell=cell, primitive=cell)


_LATTICE = np.eye(3) * 5.0


def test_get_fft_mesh_from_input_sets():
    """The mesh is that of ENCUT and PREC of the input sets.

    A missing PREC is Normal, the default of VASP.  The meshes of Normal and
    Accurate differ for this lattice, so the test fails if Accurate is assumed.

    """
    mesh = get_fft_mesh(_elph_project({"selfenergy": {"encut": 500}}), _LATTICE)
    np.testing.assert_array_equal(mesh, get_vasp_fft_mesh(500, _LATTICE, "normal"))
    assert not np.array_equal(mesh, get_vasp_fft_mesh(500, _LATTICE, "accurate"))
    project = _elph_project(
        {
            "selfenergy": {"encut": 500, "prec": "accurate"},
            "transport": {"encut": 500, "prec": "high"},
        }
    )
    np.testing.assert_array_equal(
        get_fft_mesh(project, _LATTICE),
        get_vasp_fft_mesh(500, _LATTICE, "accurate"),
    )


def test_get_fft_mesh_different_meshes():
    """Different meshes of the input sets are an error.

    Different ENCUTs that give the same mesh are not an error: 400 and 500 eV
    with Accurate give 36 points for this lattice, and 300 eV gives 28.  --encut
    replaces ENCUT of the input sets, but not PREC.

    """
    project = _elph_project(
        {
            "selfenergy": {"encut": 500, "prec": "accurate"},
            "transport": {"encut": 400, "prec": "accurate"},
        }
    )
    np.testing.assert_array_equal(get_fft_mesh(project, _LATTICE), [36, 36, 36])
    project = _elph_project(
        {
            "selfenergy": {"encut": 500, "prec": "accurate"},
            "transport": {"encut": 300, "prec": "accurate"},
        }
    )
    with pytest.raises(click.ClickException) as e:
        get_fft_mesh(project, _LATTICE)
    assert e.value.message.startswith(
        'The FFT meshes of the electron-phonon input sets of "velph.toml" differ:\n'
        "  [vasp.selfenergy]: encut = 500, prec = accurate, mesh = "
    )
    assert e.value.message.endswith("or give ENCUT by --encut.")
    np.testing.assert_array_equal(
        get_fft_mesh(project, _LATTICE, encut=600),
        get_vasp_fft_mesh(600, _LATTICE, "accurate"),
    )
    project = _elph_project(
        {"selfenergy": {"encut": 500}, "transport": {"encut": 500, "prec": "accurate"}}
    )
    with pytest.raises(click.ClickException) as e:
        get_fft_mesh(project, _LATTICE, encut=500)
    assert e.value.message.endswith(
        'Make "prec" of these input sets give the same mesh.'
    )


def test_get_fft_mesh_without_encut():
    """ENCUT has to be given by the input sets or by --encut."""
    with pytest.raises(click.ClickException) as e:
        get_fft_mesh(_elph_project({"selfenergy": {"prec": "accurate"}}), _LATTICE)
    assert e.value.message == (
        '[vasp.selfenergy.incar] of "velph.toml" does not give "encut".\n'
        'Give "encut" there, or give ENCUT by --encut.'
    )
    with pytest.raises(click.ClickException) as e:
        get_fft_mesh(_elph_project({}), _LATTICE)
    assert e.value.message == (
        '"velph.toml" has no electron-phonon input set, [vasp.selfenergy], '
        "[vasp.transport] or [vasp.ph_selfenergy], from which ENCUT is taken.\n"
        "Give ENCUT by --encut."
    )
    np.testing.assert_array_equal(
        get_fft_mesh(_elph_project({}), _LATTICE, encut=500),
        get_vasp_fft_mesh(500, _LATTICE, None),
    )


def test_differentiate_writes_fft_mesh(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Velph phelel differentiate writes the mesh of [vasp.selfenergy] in hdf5.

    The INCARs of the electron-phonon input sets written by velph init give the
    same ENCUT and PREC.

    """
    data = nacl_velph_toml()
    selfenergy_incar = data["vasp"]["selfenergy"]["incar"]
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(data))
    shutil.copytree(cwd / "C111" / "phelel", tmp_path / "phelel")
    shutil.copy(
        cwd / "C111" / "phelel_disp_C111.yaml", tmp_path / "phelel" / "phelel_disp.yaml"
    )
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cmd_root, ["phelel", "differentiate"])
    assert result.exit_code == 0, result.output
    phe = phelel.load(tmp_path / "phelel" / "phelel_disp.yaml")
    expected = get_vasp_fft_mesh(
        selfenergy_incar["encut"], phe.primitive.cell, selfenergy_incar.get("prec")
    )
    with h5py.File(tmp_path / "phelel" / "phelel_params.hdf5") as f:
        np.testing.assert_array_equal(f["FFT_mesh"][:], expected)


def test_differentiate_with_nufft(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """With --nufft, "NUFFT: <value>." is printed and dVdu changes slightly.

    Without --nufft, nothing about NUFFT is printed, so that the outputs shown in
    the documentation stay the same.  dVdu with "native-box" differs from that
    with finufft by the error of finufft.  dVdu with "native" differs also by the
    choice of g between the points of the FFT mesh of the supercell.  The
    differences show that the option reaches the calculation.

    """
    data = nacl_velph_toml()
    dVdu = {}
    for nufft in (None, "native-box", "native"):
        work_dir = tmp_path / (nufft or "default")
        work_dir.mkdir()
        (work_dir / "velph.toml").write_text(tomli_w.dumps(data))
        shutil.copytree(cwd / "C111" / "phelel", work_dir / "phelel")
        shutil.copy(
            cwd / "C111" / "phelel_disp_C111.yaml",
            work_dir / "phelel" / "phelel_disp.yaml",
        )
        monkeypatch.chdir(work_dir)
        option = [] if nufft is None else ["--nufft", nufft]
        result = CliRunner().invoke(cmd_root, ["phelel", "differentiate", *option])
        assert result.exit_code == 0, result.output
        if nufft is None:
            assert "NUFFT" not in result.output
        else:
            assert f"NUFFT: {nufft}.\n" in result.output
        with h5py.File(work_dir / "phelel" / "phelel_params.hdf5") as f:
            dVdu[nufft] = f["dVdu"][:]
    scale = np.abs(dVdu[None]).max()
    assert not np.array_equal(dVdu["native-box"], dVdu[None])
    np.testing.assert_allclose(
        dVdu["native-box"], dVdu[None], rtol=0, atol=1e-6 * scale
    )
    assert np.abs(dVdu["native"] - dVdu["native-box"]).max() > 1e-6 * scale
    np.testing.assert_allclose(dVdu["native"], dVdu[None], rtol=0, atol=1e-4 * scale)


def test_differentiate_without_supercell_directory(
    nacl_velph_toml: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A missing phelel/disp-* gives exit code 1, and no hdf5 file is written."""
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(nacl_velph_toml()))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cmd_root, ["phelel", "init"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(cmd_root, ["phelel", "differentiate"])
    assert result.exit_code == 1, result.output
    assert 'Error: "phelel/disp-000" does not exist.' in result.stderr
    assert not (tmp_path / "phelel" / "phelel_params.hdf5").exists()


def test_run_derivatives_missing_files(tmp_path: pathlib.Path):
    """Without vaspout.h5, a directory needs vasprun.xml and four other files.

    A file is found also with a suffix of compression.  The message names the
    directory and the files that are not found.

    """
    phe = phelel.load(cwd / "C111" / "phelel_disp_C111.yaml", fft_mesh=[9, 9, 9])
    disp_000 = tmp_path / "phelel" / "disp-000"
    disp_000.mkdir(parents=True)
    for filename in ("vasprun.xml", "inwap.yaml.xz", "LOCAL-POTENTIAL.bin"):
        (disp_000 / filename).touch()
    with pytest.raises(click.ClickException) as e:
        run_derivatives(phe, project_dir=ProjectDirectory(tmp_path))
    assert e.value.message == (
        f'"{disp_000}" does not have the files for the derivatives.\n'
        "It needs vaspout.h5, or vasprun.xml, inwap.yaml, LOCAL-POTENTIAL.bin,\n"
        "PAW-STRENGTH.bin and PAW-OVERLAP.bin.\n"
        "Not found: vaspout.h5, PAW-STRENGTH.bin, PAW-OVERLAP.bin."
    )
