"""Tests of VASP jobs: effective settings, planning helpers and writing."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import numpy as np
import pytest
import tomli_w
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture

from phelel.velph.cli.utils import echo_vasp_vca_hint
from phelel.velph.config.io import VelphConfigError, parse_velph_toml
from phelel.velph.config.schema import (
    InputSetSection,
    KpointsMesh,
    VaspSection,
    VelphConfig,
)
from phelel.velph.errors import VelphError
from phelel.velph.jobs.vasp_job import (
    VaspJob,
    add_cell_tags,
    effective_scheduler,
    get_kpoints,
    make_job_script,
    read_velph_toml,
    write_poscar,
    write_vasp_job,
)


def _config_with_nac(
    nac: InputSetSection, scheduler: dict | None = None
) -> VelphConfig:
    return VelphConfig(vasp=VaspSection(input_sets={"nac": nac}), scheduler=scheduler)


def test_effective_scheduler():
    """Settings of the input set have priority, and job_id is appended."""
    config = _config_with_nac(
        InputSetSection(scheduler={"walltime": "1:00:00"}),
        scheduler={"job_name": "vasp", "walltime": "96:00:00", "nodes": 1},
    )
    assert effective_scheduler(config, "nac") == {
        "job_name": "vasp",
        "walltime": "1:00:00",
        "nodes": 1,
    }
    assert effective_scheduler(config, "nac", job_id="iter2") == {
        "job_name": "vasp-iter2",
        "walltime": "1:00:00",
        "nodes": 1,
    }


def test_effective_scheduler_without_scheduler_section():
    """Without [scheduler], no scheduler settings are given.

    The settings of the input set are not used alone, as in velph 0.14.

    """
    config = _config_with_nac(InputSetSection(scheduler={"nodes": 2}))
    assert effective_scheduler(config, "nac") is None


def test_add_cell_tags(nacl_cell: PhonopyAtoms):
    """MAGMOM and VCA are added from the cell."""
    incar = {"encut": 500}
    assert add_cell_tags(incar, nacl_cell) == incar

    magnetic_cell = nacl_cell.copy()
    magnetic_cell.magnetic_moments = [1.0] * 4 + [-1.0] * 4
    assert add_cell_tags(incar, magnetic_cell) == {
        "encut": 500,
        "magmom": [1.0] * 4 + [-1.0] * 4,
    }

    cell = PhonopyAtoms(
        symbols=["Ge", "Sn", "Te"],
        cell=np.eye(3) * 4.0,
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
    )
    mixture_cell = apply_site_mixture(cell, [0.5, 0.5, 1.0], symprec=1e-5)
    assert add_cell_tags(incar, mixture_cell) == {
        "encut": 500,
        "vca": [0.5, 0.5, 1.0],
    }
    assert incar == {"encut": 500}


def test_write_vasp_job(nacl_cell: PhonopyAtoms, tmp_path: pathlib.Path):
    """POSCAR, INCAR, the k-point files, copied files and _job.sh are written."""
    potcar = tmp_path / "POTCAR"
    potcar.write_text("potcar\n")
    job = VaspJob(
        directory=tmp_path / "nac",
        poscar=nacl_cell,
        incar={"encut": 500, "lepsilon": True},
        kpoints={
            "KPOINTS": KpointsMesh(mesh=(4, 4, 4)),
            "KPOINTS_ELPH": KpointsMesh(
                mesh=((-8, 8, 8), (8, -8, 8), (8, 8, -8)), shift=(0.5, 0.5, 0.5)
            ),
        },
        job_script="#!/bin/bash\n#name=vasp-nac\n",
        copied_files={"POTCAR": potcar},
    )
    write_vasp_job(job)
    directory = tmp_path / "nac"
    assert sorted(p.name for p in directory.iterdir()) == [
        "INCAR",
        "KPOINTS",
        "KPOINTS_ELPH",
        "POSCAR",
        "POTCAR",
        "_job.sh",
    ]
    assert (directory / "INCAR").read_text().split("\n")[:2] == [
        "ENCUT = 500",
        "LEPSILON = .TRUE.",
    ]
    assert (directory / "KPOINTS").read_text().split("\n")[2:4] == [
        "Gamma",
        " ".join(f"{4:13d}" for _ in range(3)),
    ]
    elph_lines = (directory / "KPOINTS_ELPH").read_text().split("\n")
    assert elph_lines[2] == "Reciprocal"
    assert [float(x) for x in elph_lines[3].split()] == [0, 0.0625, 0.0625]
    assert [float(x) for x in elph_lines[6].split()] == [0.5, 0.5, 0.5]
    assert (directory / "POSCAR").read_text().split("\n")[5].split() == ["Na", "Cl"]
    assert (directory / "POTCAR").read_text() == "potcar\n"
    assert (directory / "_job.sh").read_text() == "#!/bin/bash\n#name=vasp-nac\n"


def test_write_vasp_job_copied_poscar(nacl_cell: PhonopyAtoms, tmp_path: pathlib.Path):
    """A path as poscar is copied as POSCAR, and _job.sh needs job_script."""
    contcar = tmp_path / "CONTCAR"
    contcar.write_text("contcar\n")
    job = VaspJob(
        directory=tmp_path / "relax" / "iter2",
        poscar=contcar,
        incar={"nsw": 10},
        kpoints={"KPOINTS": KpointsMesh(mesh=(4, 4, 4))},
    )
    write_vasp_job(job)
    directory = tmp_path / "relax" / "iter2"
    assert (directory / "POSCAR").read_text() == "contcar\n"
    assert sorted(p.name for p in directory.iterdir()) == ["INCAR", "KPOINTS", "POSCAR"]


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
    config = read_velph_toml(toml_path, "nac")
    assert config.vasp is not None and config.vasp.input_sets is not None
    assert "nac" in config.vasp.input_sets

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
    config = read_velph_toml(toml_path, "el_dos")
    assert config.vasp is not None and config.vasp.input_sets is not None
    assert "el_dos" in config.vasp.input_sets


def test_get_kpoints(nacl_velph_toml: Callable[[], dict]):
    """The meshes of all k-point files of an input set are returned.

    KPOINTS is given by the block kpoints and KPOINTS_ELPH by kpoints_dense.

    """
    data = nacl_velph_toml()
    selfenergy = data["vasp"]["selfenergy"]
    assert selfenergy["kpoints"]["mesh"] != selfenergy["kpoints_dense"]["mesh"]
    config = parse_velph_toml(data)
    assert get_kpoints(config, "selfenergy") == {
        "KPOINTS": KpointsMesh(mesh=tuple(selfenergy["kpoints"]["mesh"])),
        "KPOINTS_ELPH": KpointsMesh(mesh=tuple(selfenergy["kpoints_dense"]["mesh"])),
    }


def test_make_job_script():
    """The keys in braces in scheduler_template are replaced by the values."""
    scheduler = {
        "scheduler_template": "#!/bin/bash\n#$ -N {job_name}\n#$ -pe {pe} {cores}\n",
        "job_name": "PbTe-001",
        "pe": "mpi*",
        "cores": 48,
        "walltime": "12:00:00",
    }
    assert make_job_script(scheduler) == "#!/bin/bash\n#$ -N PbTe-001\n#$ -pe mpi* 48\n"


@pytest.mark.parametrize(
    "scheduler_dict,missing",
    [
        ({"job_name": "x"}, "scheduler_template"),
        ({"scheduler_template": "#!/bin/bash\n# {pe}\n", "job_name": "x"}, "pe"),
    ],
)
def test_make_job_script_missing_keys(scheduler_dict: dict, missing: str):
    """Test that a missing scheduler setting is an error naming the key."""
    with pytest.raises(VelphConfigError, match=missing):
        make_job_script(scheduler_dict)


@pytest.mark.parametrize("template", ["echo }\n", "echo {\n", "echo {0}\n"])
def test_make_job_script_unreadable_template(template: str):
    """A template that str.format cannot read is an error on braces."""
    with pytest.raises(VelphConfigError) as e:
        make_job_script({"scheduler_template": template})
    assert str(e.value).startswith("The job script template cannot be read:")
    assert str(e.value).endswith(
        'To write "{" or "}" in _job.sh, write "{{" or "}}" in scheduler_template.'
    )


def _get_GeSnTe_weighted_cell() -> PhonopyAtoms:
    """Return CsCl-like cell with Ge and Sn (0.5 each) co-located and Te."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.0,
        symbols=["Ge", "Sn", "Te"],
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
    )
    return apply_site_mixture(cell, [0.5, 0.5, 1.0])


def test_write_poscar_of_site_mixture(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    """A cell with weighted species is written for VASP VCA with the hint."""
    cell = _get_GeSnTe_weighted_cell()
    write_poscar(tmp_path, cell)
    lines = (tmp_path / "POSCAR").read_text().splitlines()
    assert lines[5].split() == ["Ge", "Sn", "Te"]
    assert lines[6].split() == ["1", "1", "1"]
    echo_vasp_vca_hint(cell)
    assert "INCAR:               VCA = 0.5 0.5 1" in capsys.readouterr().out


def test_write_poscar_of_ordinary_cell(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    """An ordinary cell is written without the hint of VCA."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.0,
        symbols=["Na", "Cl"],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    write_poscar(tmp_path, cell)
    assert (tmp_path / "POSCAR").read_text().splitlines()[5].split() == ["Na", "Cl"]
    echo_vasp_vca_hint(cell)
    assert capsys.readouterr().out == ""
