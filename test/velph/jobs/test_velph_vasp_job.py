"""Tests of VASP jobs: effective settings, planning helpers and writing."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import click
import numpy as np
import pytest
import tomli_w
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture

from phelel.velph.config.legacy import parse_legacy_velph_toml
from phelel.velph.config.schema import (
    InputSetSection,
    KpointsMesh,
    VaspSection,
    VelphConfig,
)
from phelel.velph.jobs.vasp_job import (
    VaspJob,
    add_cell_tags,
    effective_incar,
    effective_scheduler,
    get_kpoints,
    read_velph_toml,
    write_vasp_job,
)


def _config_with_nac(
    base_incar: dict | None,
    nac: InputSetSection,
    scheduler: dict | None = None,
) -> VelphConfig:
    return VelphConfig(
        vasp=VaspSection(incar=base_incar, input_sets={"nac": nac}),
        scheduler=scheduler,
    )


def test_effective_incar():
    """Tags of [vasp.incar] are inherited, except those in incar_unset.

    A tag given in both [vasp.incar] and the input set has the value of the
    input set.

    """
    config = _config_with_nac(
        {"encut": 500, "npar": 4, "ismear": 0},
        InputSetSection(incar={"lepsilon": True, "ismear": -5}, incar_unset=("npar",)),
    )
    assert effective_incar(config, "nac") == {
        "encut": 500,
        "ismear": -5,
        "lepsilon": True,
    }


def test_effective_incar_without_base():
    """Without [vasp.incar], the INCAR tags are those of the input set."""
    config = _config_with_nac(None, InputSetSection(incar={"lepsilon": True}))
    assert effective_incar(config, "nac") == {"lepsilon": True}


def test_effective_scheduler():
    """Settings of the input set have priority, and job_id is appended."""
    config = _config_with_nac(
        None,
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
    config = _config_with_nac(None, InputSetSection(scheduler={"nodes": 2}))
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
        scheduler={
            "job_name": "vasp-nac",
            "scheduler_template": "#!/bin/bash\n#name={job_name}\n",
        },
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
    assert "#name=vasp-nac" in (directory / "_job.sh").read_text()


def test_write_vasp_job_copied_poscar(nacl_cell: PhonopyAtoms, tmp_path: pathlib.Path):
    """A path as poscar is copied as POSCAR, and _job.sh needs scheduler."""
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


def test_read_velph_toml(
    nacl_velph_toml_v014: Callable[[], dict], tmp_path: pathlib.Path
):
    """velph.toml of velph 0.14 is read, and errors are click.ClickException.

    The errors are a missing file, a missing input set, an unknown key and a
    file that is not TOML.

    """
    toml_path = tmp_path / "velph.toml"
    with pytest.raises(click.ClickException, match="not found"):
        read_velph_toml(toml_path, "nac")

    data = nacl_velph_toml_v014()
    toml_path.write_text(tomli_w.dumps(data))
    config = read_velph_toml(toml_path, "nac")
    assert config.vasp is not None and config.vasp.input_sets is not None
    assert "nac" in config.vasp.input_sets

    del data["vasp"]["nac"]
    toml_path.write_text(tomli_w.dumps(data))
    with pytest.raises(click.ClickException) as e:
        read_velph_toml(toml_path, "nac")
    assert e.value.message == (
        f'[vasp.nac] is not found in "{toml_path}".\n'
        f'Add [vasp.nac] to "{toml_path}", or run "velph init" again.'
    )

    data["phelel"]["unknown_key"] = 1
    toml_path.write_text(tomli_w.dumps(data))
    with pytest.raises(click.ClickException, match="unknown_key"):
        read_velph_toml(toml_path, "relax")

    toml_path.write_text("[phelel\n")
    with pytest.raises(click.ClickException) as e:
        read_velph_toml(toml_path, "relax")
    assert e.value.message.startswith(f'Error in reading "{toml_path}"')


def test_get_kpoints(nacl_velph_toml_v014: Callable[[], dict]):
    """The meshes of all k-point files of an input set are returned.

    KPOINTS is given by the block kpoints and KPOINTS_ELPH by kpoints_dense.

    """
    data = nacl_velph_toml_v014()
    selfenergy = data["vasp"]["selfenergy"]
    assert selfenergy["kpoints"]["mesh"] != selfenergy["kpoints_dense"]["mesh"]
    config = parse_legacy_velph_toml(data)
    assert get_kpoints(config, "selfenergy") == {
        "KPOINTS": KpointsMesh(mesh=tuple(selfenergy["kpoints"]["mesh"])),
        "KPOINTS_ELPH": KpointsMesh(mesh=tuple(selfenergy["kpoints_dense"]["mesh"])),
    }
