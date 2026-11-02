"""Tests of reading velph.toml written by velph earlier than 0.15."""

from __future__ import annotations

import copy
import dataclasses
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
import tomli_w
from click.testing import CliRunner

from phelel import __version__
from phelel.velph.cli.velph_cmd_root import cmd_root
from phelel.velph.config.io import load_velph_toml
from phelel.velph.config.legacy import (
    convert_legacy_velph_toml,
    load_velph_toml_any_version,
    parse_legacy_velph_toml,
)
from phelel.velph.config.schema import (
    CodeSection,
    KpointsMesh,
    LinePath,
    SupercellDimension,
    VelphConfig,
)
from phelel.velph.errors import VelphConfigError
from phelel.velph.jobs.project import read_velph_toml
from phelel.velph.jobs.vasp_job import effective_scheduler, make_job_script
from phelel.velph.utils.vasp_input import write_kpoints_mesh

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
    ("el_bands", "kpoints"): "el_bands/KPOINTS",
    ("el_dos", "kpoints"): "el_dos/KPOINTS",
    ("el_dos", "kpoints_dense"): "el_dos/KPOINTS_OPT",
}


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


def test_parse_legacy_velph_toml_converts_sections(
    nacl_velph_toml_v014: Callable[[], dict],
):
    """The sections of velph 0.14 are read in the new format.

    The INCAR tags and k-point meshes of each input set are those of the file,
    [vasp.el_bands.bands] and [vasp.el_bands.dos] are read as [vasp.el_bands]
    and [vasp.el_dos], and data is not modified.

    """
    data = nacl_velph_toml_v014()
    original = copy.deepcopy(data)
    assert "fft_mesh" in data["phelel"]
    config = parse_legacy_velph_toml(data)
    assert data == original

    assert config.version == data["phelel"]["version"]
    assert config.phelel == CodeSection(
        supercell=SupercellDimension(supercell_dimension=(2, 2, 2)),
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
                assert isinstance(mesh, KpointsMesh), (set_name, block)
                assert list(mesh.mesh) == table[block]["mesh"], (set_name, block)
    assert config.vasp.input_sets["relax"].cell == "unitcell"
    assert config.vasp.input_sets["nac"].cell == "primitive"
    assert config.vasp.input_sets["el_bands"].kpoints_opt == LinePath(line=51)
    assert config.vasp.input_sets["ph_bands"].qpoints == LinePath(line=51)


def test_parse_legacy_velph_toml_without_supercell(
    velph_toml_v014: Callable[[str], dict],
):
    """[phelel] that has only version is not read as a section."""
    data = velph_toml_v014("nacl.toml")
    assert data["phelel"] == {"version": data["phelel"]["version"]}
    config = parse_legacy_velph_toml(data)
    assert config.version == data["phelel"]["version"]
    assert config.phelel is None


def test_parse_legacy_velph_toml_removed_keys(
    nacl_velph_toml_v014: Callable[[], dict],
):
    """Keys that the new format does not have are not read.

    These are [vasp.incar], phonon_supercell_dimension, [vasp.phelel.phonon]
    and [vasp.phono3py.phonon].  [vasp.relax] and [vasp.nac] without "cell" get
    the cells that the generate commands of velph 0.14 use, and "cell" is read
    in lower case.

    """
    data = nacl_velph_toml_v014()
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


def _with_el_bands_of_velph_0_10(
    data: dict[str, Any], kpoints_dense: bool = True
) -> dict[str, Any]:
    """Return data with [vasp.el_bands] in the form of velph 0.10 and earlier.

    velph init of these versions writes one [vasp.el_bands] with lorbit and
    nedos in incar, kpoints_opt and kpoints_dense, in place of
    [vasp.el_bands.bands] and [vasp.el_bands.dos].  The tables are made from
    those of [vasp.el_bands.bands] and [vasp.el_bands.dos] of data, and
    [vasp.el_bands.scheduler] is added.

    """
    data = copy.deepcopy(data)
    bands = data["vasp"]["el_bands"]["bands"]
    dos = data["vasp"]["el_bands"]["dos"]
    el_bands = {
        "incar": {"lorbit": 11, "nedos": 5001, **bands["incar"]},
        "kpoints": bands["kpoints"],
        "kpoints_opt": bands["kpoints_opt"],
        "scheduler": {"pe": "mpi 8"},
    }
    if kpoints_dense:
        el_bands["kpoints_dense"] = dos["kpoints_dense"]
    data["vasp"]["el_bands"] = el_bands
    return data


@pytest.mark.parametrize("kpoints_dense", [True, False])
def test_parse_legacy_velph_toml_el_bands_of_velph_0_10(
    nacl_velph_toml_v014: Callable[[], dict], kpoints_dense: bool
):
    """[vasp.el_bands] of velph 0.10 is read as [vasp.el_bands] and [vasp.el_dos].

    These are the input sets that "velph el_bands generate" of velph 0.10
    writes in el_bands/bands and el_bands/dos.  [vasp.el_bands] has the INCAR
    tags without lorbit and nedos and kpoints_opt, and [vasp.el_dos] has all
    the INCAR tags and kpoints_dense.  Both have kpoints and the scheduler.
    Without kpoints_dense, velph 0.10 writes no DOS, and there is no
    [vasp.el_dos].  data is not modified.

    """
    data = _with_el_bands_of_velph_0_10(nacl_velph_toml_v014(), kpoints_dense)
    original = copy.deepcopy(data)
    config = parse_legacy_velph_toml(data)
    assert data == original

    el_bands = data["vasp"]["el_bands"]
    assert config.vasp is not None and config.vasp.input_sets is not None
    input_sets = config.vasp.input_sets
    bands = input_sets["el_bands"]
    assert bands.incar == {
        tag: value
        for tag, value in el_bands["incar"].items()
        if tag not in ("lorbit", "nedos")
    }
    assert bands.kpoints == KpointsMesh(mesh=tuple(el_bands["kpoints"]["mesh"]))
    assert bands.kpoints_opt == LinePath(line=51)
    assert bands.kpoints_dense is None
    assert bands.scheduler == {"pe": "mpi 8"}
    if not kpoints_dense:
        assert "el_dos" not in input_sets
        return
    dos = input_sets["el_dos"]
    assert dos.incar == el_bands["incar"]
    assert dos.kpoints == bands.kpoints
    assert dos.kpoints_dense == KpointsMesh(
        mesh=tuple(el_bands["kpoints_dense"]["mesh"])
    )
    assert dos.kpoints_opt is None
    assert dos.scheduler == {"pe": "mpi 8"}


def test_convert_legacy_velph_toml_el_bands_of_velph_0_10(
    nacl_velph_toml_v014: Callable[[], dict],
    tmp_path: pathlib.Path,
):
    """The command "velph convert" prints how [vasp.el_bands] of 0.10 is converted.

    kspacing of [vasp.el_bands.kpoints] is replaced by mesh in both
    [vasp.el_bands] and [vasp.el_dos].

    """
    data = _with_el_bands_of_velph_0_10(nacl_velph_toml_v014())
    data["vasp"]["el_bands"]["kpoints"] = {"kspacing": 0.15}
    old_filename = tmp_path / "velph_v010.toml"
    old_filename.write_text(tomli_w.dumps(data))
    new_filename = tmp_path / "velph.toml"

    result = CliRunner().invoke(
        cmd_root, ["convert", str(old_filename), str(new_filename)]
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "fft_mesh of [phelel] was removed.",
        "[vasp.el_bands] was split into [vasp.el_bands] with kpoints_opt and "
        "[vasp.el_dos] with kpoints_dense.",
        "lorbit and nedos were removed from [vasp.el_bands.incar].",
        "kspacing of [vasp.el_bands.kpoints] was replaced by mesh.",
        f'"{new_filename}" was written from "{old_filename}".',
    ]
    config = load_velph_toml(new_filename)
    assert config == _with_current_version(parse_legacy_velph_toml(data))
    assert config.vasp is not None and config.vasp.input_sets is not None
    kpoints = config.vasp.input_sets["el_bands"].kpoints
    assert isinstance(kpoints, KpointsMesh)
    assert config.vasp.input_sets["el_dos"].kpoints == kpoints


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
    "el_dos": (["el_dos", "generate"],),
}


def _run_generate_commands(set_names: tuple[str, ...]) -> None:
    """Run the commands that write the input sets.

    The commands are run in the current directory.  The generate commands read
    velph.toml of velph 0.14 through the legacy loader, so only the commands
    that are not rewritten yet are those of velph 0.14.

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


def _kpoints_text(mesh: KpointsMesh, path: pathlib.Path) -> str:
    """Return KPOINTS written from mesh in the same way as the generate commands."""
    write_kpoints_mesh(path, mesh)
    return path.read_text()


def _check_meshes_equal_generate(
    config: VelphConfig, set_names: tuple[str, ...], tmp_path: pathlib.Path
) -> None:
    assert config.vasp is not None and config.vasp.input_sets is not None
    for (set_name, block), pattern in _KPOINTS_FILES.items():
        if set_name not in set_names:
            continue
        mesh = getattr(config.vasp.input_sets[set_name], block)
        assert isinstance(mesh, KpointsMesh), (set_name, block)
        written = sorted(tmp_path.glob(pattern))
        assert written, f'"{pattern}" was not written.'
        expected = _kpoints_text(mesh, tmp_path / "expected_KPOINTS")
        assert written[0].read_text() == expected, (set_name, block)


def test_kspacing_mesh_equals_generate(
    nacl_velph_toml_v014: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The mesh converted from kspacing is the one that generate writes.

    The supercells are made from [unitcell] and the supercell matrices, and
    the generate commands read them from phelel_disp.yaml, phonopy_disp.yaml
    and phono3py_disp.yaml.  The meshes on the unit cell, the primitive cell
    and the supercells of NaCl differ from each other, and the mesh on the
    primitive cell is a generalized regular grid.

    "velph el_bands generate" needs seekpath, because [vasp.el_bands.kpoints_opt]
    of this velph.toml does not give "path".

    """
    pytest.importorskip("seekpath")
    set_names = tuple(_GENERATE_COMMANDS)
    data = _velph_toml_with_kspacing(nacl_velph_toml_v014())
    monkeypatch.chdir(tmp_path)
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(data))
    _run_generate_commands(set_names)
    config = load_velph_toml_any_version(tmp_path / "velph.toml")
    _check_meshes_equal_generate(config, set_names, tmp_path)

    assert config.vasp is not None and config.vasp.input_sets is not None
    meshes = {
        set_name: config.vasp.input_sets[set_name].kpoints
        for set_name in ("phelel", "relax", "nac")
    }
    assert meshes["phelel"] == KpointsMesh(mesh=(4, 4, 4))
    assert meshes["relax"] == KpointsMesh(mesh=(8, 8, 8))
    assert meshes["nac"] == KpointsMesh(mesh=((-8, 8, 8), (8, -8, 8), (8, 8, -8)))


@pytest.mark.parametrize("split_site_mixture", [False, True])
def test_kspacing_mesh_equals_generate_site_mixture(
    velph_toml_v014: Callable[[str], dict],
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
    scheme = "split" if split_site_mixture else "merge"
    data = _velph_toml_with_kspacing(velph_toml_v014(f"gesnte_{scheme}_222.toml"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(data))
    _run_generate_commands(set_names)
    config = load_velph_toml_any_version(tmp_path / "velph.toml")
    _check_meshes_equal_generate(config, set_names, tmp_path)


_VELPH_TOML_V014_DIR = pathlib.Path(__file__).parent.parent / "data" / "velph_toml_v014"


def _with_current_version(config: VelphConfig) -> VelphConfig:
    """Return config with [velph] version that velph convert writes.

    The version in the file of velph 0.14 is checked to differ from that of
    this velph, so that a converter that copies the old version is found.

    """
    assert config.version is not None
    assert config.version != __version__
    return dataclasses.replace(config, version=__version__)


@pytest.mark.parametrize(
    "filename,messages",
    [
        ("nacl.toml", []),
        ("nacl_222.toml", ["fft_mesh of [phelel] was removed."]),
        ("gesnte_merge_222.toml", ["fft_mesh of [phelel] was removed."]),
        ("gesnte_split_222.toml", ["fft_mesh of [phelel] was removed."]),
    ],
)
def test_convert_legacy_velph_toml(
    velph_toml_v014: Callable[[str], dict],
    filename: str,
    messages: list[str],
    tmp_path: pathlib.Path,
):
    """The converted file is read into the VelphConfig of the original file.

    Only [velph] version differs, which is the version of this velph.

    """
    new_filename = tmp_path / "velph.toml"
    assert (
        convert_legacy_velph_toml(_VELPH_TOML_V014_DIR / filename, new_filename)
        == messages
    )
    assert load_velph_toml(new_filename) == _with_current_version(
        parse_legacy_velph_toml(velph_toml_v014(filename))
    )


def test_convert_legacy_velph_toml_kspacing_and_removed_keys(
    nacl_velph_toml_v014: Callable[[], dict],
    tmp_path: pathlib.Path,
):
    """The kspacing is replaced by mesh, and the removed keys are reported.

    The section names in the messages are those of velph 0.14.

    """
    data = _velph_toml_with_kspacing(nacl_velph_toml_v014())
    data["vasp"]["incar"] = {"encut": 300}
    for code in ("phelel", "phono3py"):
        data[code]["phonon_supercell_dimension"] = [3, 3, 3]
        data["vasp"][code]["phonon"] = {"incar": {"encut": 300}}
    old_filename = tmp_path / "velph_v014.toml"
    old_filename.write_text(tomli_w.dumps(data))
    new_filename = tmp_path / "velph.toml"

    result = CliRunner().invoke(
        cmd_root, ["convert", str(old_filename), str(new_filename)]
    )
    assert result.exit_code == 0, result.output
    assert "kspacing" not in new_filename.read_text()
    config = load_velph_toml(new_filename)
    assert config == _with_current_version(parse_legacy_velph_toml(data))
    assert config.vasp is not None and config.vasp.input_sets is not None
    assert config.vasp.input_sets["nac"].kpoints == KpointsMesh(
        mesh=((-8, 8, 8), (8, -8, 8), (8, 8, -8))
    )
    replaced = [
        f"kspacing of [vasp.{block}] was replaced by mesh."
        for block in (
            "phelel.kpoints",
            "phonopy.kpoints",
            "phono3py.kpoints",
            "selfenergy.kpoints",
            "selfenergy.kpoints_dense",
            "transport.kpoints",
            "transport.kpoints_dense",
            "ph_selfenergy.kpoints",
            "ph_selfenergy.kpoints_dense",
            "relax.kpoints",
            "nac.kpoints",
            "el_bands.bands.kpoints",
            "el_bands.dos.kpoints",
            "el_bands.dos.kpoints_dense",
        )
    ]
    assert result.stdout.splitlines() == [
        "[vasp.incar] was removed.",
        "fft_mesh of [phelel] was removed.",
        "phonon_supercell_dimension of [phelel] was removed.",
        "phonon_supercell_dimension of [phono3py] was removed.",
        "[vasp.phelel.phonon] was removed.",
        "[vasp.phono3py.phonon] was removed.",
        *replaced,
        f'"{new_filename}" was written from "{old_filename}".',
    ]


@pytest.mark.parametrize("case", ["new_exists", "new_format", "invalid"])
def test_convert_legacy_velph_toml_error(
    nacl_velph_toml_v014: Callable[[], dict],
    case: str,
    tmp_path: pathlib.Path,
):
    """Nothing is written when the file cannot be converted."""
    data = nacl_velph_toml_v014()
    new_filename = tmp_path / "velph.toml"
    if case == "new_exists":
        new_filename.write_text("x = 1\n")
        expected = 'velph.toml" exists'
    elif case == "new_format":
        data["velph"] = {"version": data["phelel"].pop("version")}
        expected = "it is in the new format"
    else:
        data["phelel"]["supercell_dimension"] = [2, 2]
        expected = "[phelel]"
    old_filename = tmp_path / "velph_v014.toml"
    old_filename.write_text(tomli_w.dumps(data))

    result = CliRunner().invoke(
        cmd_root, ["convert", str(old_filename), str(new_filename)]
    )
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.startswith("Error: ")
    assert expected in result.stderr
    if case == "new_exists":
        assert new_filename.read_text() == "x = 1\n"
    else:
        assert not new_filename.exists()


def test_load_velph_toml_without_version(
    nacl_velph_toml_v014: Callable[[], dict], tmp_path: pathlib.Path
):
    """A velph.toml without [velph] version and [phelel] version is an error.

    velph tells the format of a velph.toml by these keys, and the message names
    both of them.

    """
    data = nacl_velph_toml_v014()
    del data["phelel"]["version"]
    filename = tmp_path / "velph.toml"
    filename.write_text(tomli_w.dumps(data))
    with pytest.raises(VelphConfigError) as e:
        load_velph_toml_any_version(filename)
    assert str(e.value) == (
        f'"{filename}" has no version.\n'
        "velph chooses how to read velph.toml by its version.\n"
        "velph 0.15 and later write [velph] version.\n"
        "velph 0.14 and earlier write [phelel] version.\n"
        'Run "velph init" to make a new velph.toml.'
    )


def _job_script_of_velph_0_14(scheduler: dict[str, Any], job_id: str | None) -> str:
    """Return the job script that velph 0.14 writes for the scheduler settings.

    This is the code of velph/utils/scheduler.py and write_launch_script of
    velph/cli/utils.py of velph 0.14, written here as the reference.

    """
    if scheduler.get("scheduler_name") == "sge":
        header = """#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m n
#$ -N {job_name}
#$ -V
#$ -o _scheduler-stdout.txt
#$ -e _scheduler-stderr.txt
"""
        params = []
        if "pe" in scheduler:
            params.append("#$ -pe {pe}")
        if "walltime" in scheduler:
            params.append("#$ -l h_rt={walltime}")
        params_header = "\n".join(params) + "\n" if params else ""
        commands = """
{prepend_text}

{mpirun_command} {vasp_binary}

{append_text}
"""
        template = header + params_header + commands
    elif scheduler.get("scheduler_name") == "slurm":
        template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --time=96:00:00      # Time limit hrs:min:sec
#SBATCH --output=ci_%j.log   # Standard output and error log

{prepend_text}

{mpirun_command} {vasp_binary}

{append_text}
"""
    else:
        template = scheduler["scheduler_template"]
    scheduler = copy.deepcopy(scheduler)
    if "job_name" in scheduler and job_id is not None:
        scheduler["job_name"] += f"-{job_id}"
    return template.format(**scheduler)


def _with_scheduler_name(data: dict[str, Any], case: str) -> dict[str, Any]:
    """Return data with scheduler_name of velph 0.14 for the case."""
    data = copy.deepcopy(data)
    scheduler = data["scheduler"]
    if case == "sge":
        # velph 0.14 does not use scheduler_template with scheduler_name.
        scheduler.update(scheduler_name="sge", pe="mpi 16")
    elif case == "sge_pe_in_input_set":
        scheduler["scheduler_name"] = "sge"
        del scheduler["walltime"]
        data["vasp"]["phelel"]["scheduler"] = {"pe": "mpi 64"}
    elif case == "slurm":
        scheduler.update(scheduler_name="slurm", partition="all", ntasks=16)
    elif case == "sge_in_input_set":
        data["vasp"]["nac"]["scheduler"] = {"scheduler_name": "sge", "pe": "mpi 4"}
    else:
        scheduler["scheduler_name"] = "custom"
    return data


@pytest.mark.parametrize(
    "case", ["sge", "sge_pe_in_input_set", "slurm", "sge_in_input_set", "custom"]
)
def test_convert_legacy_velph_toml_scheduler_name(
    nacl_velph_toml_v014: Callable[[], dict],
    tmp_path: pathlib.Path,
    case: str,
):
    """scheduler_name is converted so that the job scripts are those of velph 0.14.

    The job script of every input set, also with a job_id such as that of
    relax, is compared with that written by the code of velph 0.14.  The cases
    are SGE with pe and walltime, SGE with pe only in [vasp.phelel.scheduler]
    and without walltime, Slurm, scheduler_name only in [vasp.nac.scheduler],
    and a name other than "sge" and "slurm", with which velph 0.14 uses
    scheduler_template.

    """
    data = _with_scheduler_name(nacl_velph_toml_v014(), case)
    old_filename = tmp_path / "velph_v014.toml"
    old_filename.write_text(tomli_w.dumps(data))
    new_filename = tmp_path / "velph.toml"
    messages = convert_legacy_velph_toml(old_filename, new_filename)
    assert "scheduler_name" not in new_filename.read_text()

    project = read_velph_toml(new_filename)
    tables = _input_set_tables(data)
    assert project.config.vasp is not None
    assert project.config.vasp.input_sets is not None
    assert set(project.config.vasp.input_sets) == set(tables)
    for set_name, table in tables.items():
        scheduler = {**data["scheduler"], **table.get("scheduler", {})}
        for job_id in (None, "iter1"):
            assert make_job_script(
                effective_scheduler(project, set_name, job_id=job_id)
            ) == _job_script_of_velph_0_14(scheduler, job_id), (set_name, job_id)

    if case == "sge_in_input_set":
        removed = ['scheduler_name = "sge" of [vasp.nac.scheduler] was removed.']
    else:
        name = data["scheduler"]["scheduler_name"]
        removed = [f'scheduler_name = "{name}" of [scheduler] was removed.']
    written = (
        []
        if case == "custom"
        else [
            "scheduler_template was written to give the job scripts that velph "
            "0.14 writes for scheduler_name."
        ]
    )
    assert messages == ["fft_mesh of [phelel] was removed.", *removed, *written]
