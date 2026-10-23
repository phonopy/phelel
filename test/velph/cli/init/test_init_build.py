"""Tests of the settings of velph.toml made by velph init."""

from __future__ import annotations

import pathlib
import tomllib
from typing import Any

import click
import pytest
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_primitive

from phelel import __version__
from phelel.velph.cli.init.build import build_velph_config
from phelel.velph.config.io import load_velph_toml, parse_template
from phelel.velph.config.schema import (
    CodeSection,
    InitSection,
    KpointsMesh,
    LinePath,
    MaxNumAtoms,
    SupercellDimension,
    SupercellMatrix,
    VelphConfig,
)
from phelel.velph.config.toml_writer import write_velph_toml

_DIM222 = SupercellDimension(supercell_dimension=(2, 2, 2))


def _build(
    cell: PhonopyAtoms, template_lines: list[str] | None = None, **options: Any
) -> VelphConfig:
    """Return build_velph_config with a template given by lines and options."""
    template = None
    if template_lines is not None:
        template = parse_template(tomllib.loads("\n".join(template_lines)))
    return build_velph_config(cell, template, InitSection(**options))


def _input_set_names(config: VelphConfig) -> list[str]:
    assert config.vasp is not None and config.vasp.input_sets is not None
    return list(config.vasp.input_sets)


def _kpoints(config: VelphConfig, set_name: str, block: str = "kpoints") -> Any:
    assert config.vasp is not None and config.vasp.input_sets is not None
    return getattr(config.vasp.input_sets[set_name], block)


def test_build_shows_supercell(nacl_cell: PhonopyAtoms, capsys: pytest.CaptureFixture):
    """The number of atoms and the supercell matrix of each code are shown.

    The unit cell has 8 atoms, and the determinant of the matrix is 16.

    """
    _build(nacl_cell, supercell=SupercellMatrix(((0, 2, 2), (2, 0, 2), (2, 2, 0))))
    out = capsys.readouterr().out
    for code in ("phelel", "phonopy", "phono3py"):
        assert (
            f"Supercell of [{code}]: 128 atoms, supercell matrix "
            "[[0, 2, 2], [2, 0, 2], [2, 2, 0]]."
        ) in out


def test_build_without_supercell(
    nacl_cell: PhonopyAtoms, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    """Without a supercell, only the input sets without a code are written.

    The codes are the default ones, so the codes without a supercell are not
    written, with a message.  What is built is written and read back as
    velph.toml.

    """
    config = _build(nacl_cell)
    out = capsys.readouterr().out
    for code in ("phelel", "phonopy", "phono3py"):
        assert getattr(config, code) is None
        assert f"[{code}] is not written: no supercell is given for it." in out
    assert _input_set_names(config) == ["relax", "nac", "el_bands", "el_dos"]
    assert config.version == __version__
    assert config.init is None
    assert config.vasp is not None and config.vasp.incar is None
    write_velph_toml(config, tmp_path / "velph.toml")
    assert load_velph_toml(tmp_path / "velph.toml") == config


def test_build_with_supercell(nacl_cell: PhonopyAtoms, tmp_path: pathlib.Path):
    """With a supercell, every code and every input set is written."""
    config = _build(nacl_cell, supercell=_DIM222)
    for code, amplitude in (("phelel", 0.03), ("phonopy", 0.01), ("phono3py", 0.03)):
        assert getattr(config, code) == CodeSection(
            supercell=_DIM222,
            amplitude=amplitude,
            diagonal=False,
            plusminus=True,
            nosym=False,
        )
    assert _input_set_names(config) == [
        "relax",
        "nac",
        "phelel",
        "phonopy",
        "phono3py",
        "selfenergy",
        "transport",
        "ph_selfenergy",
        "el_bands",
        "el_dos",
        "ph_bands",
    ]
    write_velph_toml(config, tmp_path / "velph.toml")
    assert load_velph_toml(tmp_path / "velph.toml") == config


def test_build_codes(nacl_cell: PhonopyAtoms, capsys: pytest.CaptureFixture):
    """Only the codes of codes and their input sets are written.

    A section of the template for a code not in codes is ignored with a
    message.

    """
    config = _build(
        nacl_cell,
        ["[phelel]", "supercell_dimension = [3, 3, 3]"],
        codes=("phonopy",),
        supercell=_DIM222,
    )
    assert config.phelel is None and config.phono3py is None
    assert config.phonopy is not None and config.phonopy.supercell == _DIM222
    assert _input_set_names(config) == ["relax", "nac", "phonopy", "el_bands", "el_dos"]
    assert '[phelel] of the template is ignored: "phelel" is not in codes.' in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize("where", ["template", "options"])
def test_build_codes_without_supercell(nacl_cell: PhonopyAtoms, where: str):
    """A code given in codes without a supercell is an error."""
    with pytest.raises(click.ClickException, match='"phono3py" is in codes'):
        if where == "template":
            _build(
                nacl_cell,
                ["[init]", 'codes = ["phonopy", "phono3py"]', "[phonopy]"]
                + ["supercell_dimension = [2, 2, 2]"],
            )
        else:
            _build(
                nacl_cell,
                ["[phonopy]", "supercell_dimension = [2, 2, 2]"],
                codes=("phonopy", "phono3py"),
            )


def test_build_supercell_of_one_code(
    nacl_cell: PhonopyAtoms, capsys: pytest.CaptureFixture
):
    """A supercell given only in a section of the template is that of the code."""
    config = _build(nacl_cell, ["[phonopy]", "supercell_dimension = [3, 3, 3]"])
    assert config.phelel is None and config.phono3py is None
    assert config.phonopy is not None
    assert config.phonopy.supercell == SupercellDimension(supercell_dimension=(3, 3, 3))
    assert "[phelel] is not written" in capsys.readouterr().out


@pytest.mark.parametrize(
    "layers,amplitude",
    [
        (("section",), 0.04),
        (("section", "init"), 0.05),
        (("section", "init", "options"), 0.06),
        (("section", "options"), 0.06),
        (("init",), 0.05),
    ],
)
def test_build_layers(nacl_cell: PhonopyAtoms, layers: tuple[str, ...], amplitude):
    """An upper layer wins: defaults < sections < [init] < command line.

    The supercell, amplitude, nosym and cell of relax are given in the layers.

    """
    template_lines = []
    if "section" in layers:
        template_lines += [
            "[phelel]",
            "supercell_dimension = [1, 1, 1]",
            "amplitude = 0.04",
            "nosym = true",
            "[vasp.relax]",
            'cell = "primitive"',
        ]
    if "init" in layers:
        template_lines += [
            "[init]",
            "supercell_dimension = [2, 2, 2]",
            "amplitude = 0.05",
            "nosym = false",
            'cell_for_relax = "unitcell"',
        ]
    options: dict[str, Any] = {}
    if "options" in layers:
        options = {
            "supercell": SupercellDimension((1, 1, 2)),
            "amplitude": 0.06,
            "nosym": True,
            "cell_for_relax": "primitive",
        }
    config = _build(nacl_cell, template_lines, **options)
    assert config.phelel is not None
    assert config.phelel.amplitude == amplitude
    assert config.vasp is not None and config.vasp.input_sets is not None
    if "options" in layers:
        expected = ((1, 1, 2), True, "primitive")
    elif "init" in layers:
        expected = ((2, 2, 2), False, "unitcell")
    else:
        expected = ((1, 1, 1), True, "primitive")
    assert config.phelel.supercell == SupercellDimension(expected[0])
    assert config.phelel.nosym is expected[1]
    assert config.vasp.input_sets["relax"].cell == expected[2]


def test_build_init_options_act_on_every_code(nacl_cell: PhonopyAtoms):
    """Displacement options of [init] and the command line act on every code."""
    config = _build(
        nacl_cell,
        ["[init]", "diagonal = true", 'plusminus = "auto"'],
        supercell=_DIM222,
        nosym=True,
    )
    for code in ("phelel", "phonopy", "phono3py"):
        section = getattr(config, code)
        assert section.diagonal is True
        assert section.plusminus == "auto"
        assert section.nosym is True


def test_build_number_of_snapshots(nacl_cell: PhonopyAtoms):
    """number_of_snapshots of the template is written."""
    config = _build(
        nacl_cell, ["[phonopy]", "number_of_snapshots = 10"], supercell=_DIM222
    )
    assert config.phonopy is not None and config.phonopy.number_of_snapshots == 10


def _site_mixture_cell() -> PhonopyAtoms:
    """Return a CsCl-like cell with a co-located Ge/Sn site and a Te site."""
    return PhonopyAtoms(
        symbols=["Ge", "Sn", "Te"],
        cell=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
    )


@pytest.mark.parametrize(
    "template_lines,options,schemes",
    [
        ([], {}, ("merge", "merge")),
        (["[phelel]", 'site_mixture_scheme = "split"'], {}, ("split", "merge")),
        (["[init]", "split_site_mixture = true"], {}, ("split", "split")),
        (
            ["[init]", "split_site_mixture = true"],
            {"split_site_mixture": False},
            ("merge", "merge"),
        ),
    ],
)
def test_build_site_mixture_scheme(
    template_lines: list[str], options: dict[str, Any], schemes: tuple[str, str]
):
    """site_mixture_scheme of [phelel] and [phonopy] follows the layers.

    phono3py, which does not support site mixture, has none.

    """
    config = _build(
        _site_mixture_cell(),
        template_lines,
        site_mixture="0.5 0.5 1.0",
        supercell=_DIM222,
        **options,
    )
    assert config.phelel is not None and config.phonopy is not None
    assert (config.phelel.site_mixture_scheme, config.phonopy.site_mixture_scheme) == (
        schemes
    )
    assert config.phono3py is not None
    assert config.phono3py.site_mixture_scheme is None


def test_build_site_mixture_scheme_without_weights(nacl_cell: PhonopyAtoms):
    """site_mixture_scheme is not written for a cell without weighted species."""
    config = _build(
        nacl_cell, ["[phelel]", 'site_mixture_scheme = "split"'], supercell=_DIM222
    )
    assert config.phelel is not None and config.phelel.site_mixture_scheme is None


def test_build_kpoints_kspacing(nacl_cell: PhonopyAtoms):
    """The meshes of kspacing and kspacing_dense are on the cell of each input set.

    The meshes follow N = ceiling(|b| 2 pi / kspacing): with kspacing 0.2,
    5.52 -> 6 on the conventional cell of NaCl (a = 5.69 Angstrom), 9.56 -> 10
    on the primitive cell and 2.76 -> 3 on the 2x2x2 supercell, and 19.1 -> 20
    on the primitive cell with kspacing_dense 0.1.  ph_bands has the mesh of
    the defaults and the line blocks are those of the defaults.

    """
    config = _build(nacl_cell, supercell=_DIM222, kspacing=0.2, kspacing_dense=0.1)
    expected = {
        ("relax", "kpoints"): (6, 6, 6),
        ("nac", "kpoints"): (10, 10, 10),
        ("phelel", "kpoints"): (3, 3, 3),
        ("phonopy", "kpoints"): (3, 3, 3),
        ("phono3py", "kpoints"): (3, 3, 3),
        ("selfenergy", "kpoints"): (10, 10, 10),
        ("selfenergy", "kpoints_dense"): (20, 20, 20),
        ("transport", "kpoints_dense"): (20, 20, 20),
        ("ph_selfenergy", "kpoints_dense"): (20, 20, 20),
        ("el_bands", "kpoints"): (10, 10, 10),
        ("el_dos", "kpoints"): (10, 10, 10),
        ("el_dos", "kpoints_dense"): (20, 20, 20),
        ("ph_bands", "kpoints"): (1, 1, 1),
    }
    for (set_name, block), mesh in expected.items():
        assert _kpoints(config, set_name, block) == KpointsMesh(mesh=mesh), (
            set_name,
            block,
        )
    assert _kpoints(config, "el_bands", "kpoints_opt") == LinePath(line=51)
    assert _kpoints(config, "ph_bands", "qpoints") == LinePath(line=51)


@pytest.mark.parametrize(
    "block_lines,mesh,shift",
    [
        (["mesh = [4, 4, 4]"], (4, 4, 4), None),
        (["kspacing = 0.4"], (3, 3, 3), None),
        (["kspacing = 0.4", "shift = [0.5, 0.5, 0.5]"], (3, 3, 3), (0.5, 0.5, 0.5)),
    ],
)
def test_build_kpoints_of_template(
    nacl_cell: PhonopyAtoms,
    block_lines: list[str],
    mesh: tuple[int, int, int],
    shift: tuple[float, float, float] | None,
):
    """A block of the template wins over kspacing of [init].

    kspacing 0.4 gives 2.76 -> 3 on the conventional cell of NaCl.

    """
    config = _build(nacl_cell, ["[vasp.relax.kpoints]"] + block_lines, kspacing=0.2)
    assert _kpoints(config, "relax") == KpointsMesh(mesh=mesh, shift=shift)


def test_build_kpoints_of_ph_bands(nacl_cell: PhonopyAtoms):
    """The kspacing of [vasp.ph_bands.kpoints] is resolved on the primitive cell."""
    config = _build(
        nacl_cell, ["[vasp.ph_bands.kpoints]", "kspacing = 0.2"], supercell=_DIM222
    )
    assert _kpoints(config, "ph_bands") == KpointsMesh(mesh=(10, 10, 10))


@pytest.mark.filterwarnings("error")
def test_build_use_grg(nacl_cell: PhonopyAtoms):
    """use_grg acts on the primitive cell only, not on the unit cell and supercells.

    No warning of phonopy is given for the cells with several lattice points.

    """
    config = _build(nacl_cell, supercell=_DIM222, kspacing=0.2, use_grg=True)
    assert _kpoints(config, "nac") == KpointsMesh(
        mesh=((-6, 6, 6), (6, -6, 6), (6, 6, -6))
    )
    assert _kpoints(config, "relax") == KpointsMesh(mesh=(6, 6, 6))
    assert _kpoints(config, "phelel") == KpointsMesh(mesh=(3, 3, 3))


def test_build_use_grg_supercell_of_primitive_cell(nacl_cell: PhonopyAtoms):
    """use_grg does not act on a supercell also when the supercell is primitive.

    The input cell is the primitive cell of NaCl, and the supercell is 1x1x1.

    """
    primitive = get_primitive(nacl_cell, "F")
    config = _build(
        primitive,
        supercell=SupercellDimension((1, 1, 1)),
        kspacing=0.2,
        use_grg=True,
    )
    assert _kpoints(config, "phelel") == KpointsMesh(mesh=(10, 10, 10))
    assert _kpoints(config, "nac") == KpointsMesh(
        mesh=((-6, 6, 6), (6, -6, 6), (6, 6, -6))
    )


def test_build_grg_of_body_centred_lattice(
    tio2_prim_cell: PhonopyAtoms, capsys: pytest.CaptureFixture
):
    """The mesh is a GR-grid when a regular mesh breaks the symmetry, with a message.

    The input cell is the primitive cell of anatase, which is the unit cell and
    the primitive cell, and use_grg is not given.

    """
    config = _build(tio2_prim_cell, kspacing=0.5)
    grg = KpointsMesh(mesh=((0, 4, 4), (4, 0, 4), (2, 2, 0)))
    assert _kpoints(config, "relax") == grg
    assert _kpoints(config, "nac") == grg
    # The messages are wrapped, and the blocks of a cell are in one message.
    out = " ".join(capsys.readouterr().out.split())
    assert (
        "The k-point mesh of [vasp.relax.kpoints] is a generalized regular grid, "
        "because a regular mesh breaks the symmetry of the unit cell."
    ) in out
    assert (
        "The k-point meshes of [vasp.nac.kpoints], [vasp.el_bands.kpoints], "
        "[vasp.el_dos.kpoints], [vasp.el_dos.kpoints_dense] are generalized "
        "regular grids, because a regular mesh breaks the symmetry of the "
        "primitive cell."
    ) in out
    assert out.count("breaks the symmetry of the primitive cell.") == 1


def test_build_broken_grid_symmetry(tio2_prim_cell: PhonopyAtoms):
    """No mesh of the 2x2x2 supercell of the primitive cell of BCT is an error."""
    with pytest.raises(
        click.ClickException,
        match=r"\[vasp\.phelel\.kpoints\].*supercell of \[phelel\]",
    ):
        _build(tio2_prim_cell, supercell=_DIM222, kspacing=0.5)


def test_build_symmetrize_cell_mesh_of_unit_cell(bi2te3_prim_cell: PhonopyAtoms):
    """The mesh of the unit cell uses the symmetry of the unit cell.

    With symmetrize_cell, the unit cell of Bi2Te3 is the hexagonal cell (a =
    4.44, c = 31.4 Angstrom), whose mesh with kspacing 0.1 is (17, 17, 2).

    """
    config = _build(bi2te3_prim_cell, symmetrize_cell=True)
    assert _kpoints(config, "relax") == KpointsMesh(mesh=(17, 17, 2))


def test_build_scheduler(nacl_cell: PhonopyAtoms, capsys: pytest.CaptureFixture[str]):
    """[scheduler] is written only when the template has it, as it is.

    Without it, a message says that no job script is written.

    """
    message = "[scheduler] is not written: the template has no [scheduler]"
    for template_lines in (None, ["[vasp.incar]", "encut = 400"]):
        assert _build(nacl_cell, template_lines).scheduler is None
        assert message in capsys.readouterr().out
    config = _build(nacl_cell, ["[scheduler]", 'job_name = "x"'])
    assert config.scheduler == {"job_name": "x"}
    assert message not in capsys.readouterr().out


def test_build_incar(nacl_cell: PhonopyAtoms):
    """[vasp.incar] of the template reaches the INCAR of every input set."""
    config = _build(
        nacl_cell, ["[vasp.incar]", "encut = 300", "[vasp.relax.incar]", "nsw = 3"]
    )
    assert config.vasp is not None and config.vasp.input_sets is not None
    for set_name, input_set in config.vasp.input_sets.items():
        assert input_set.incar is not None
        assert input_set.incar["encut"] == 300, set_name
    assert config.vasp.input_sets["relax"].incar is not None
    assert config.vasp.input_sets["relax"].incar["nsw"] == 3


def test_build_max_num_atoms_without_symmetrize_cell(nacl_cell: PhonopyAtoms):
    """max_num_atoms without symmetrize_cell is an error."""
    with pytest.raises(click.ClickException, match="symmetrize_cell"):
        _build(nacl_cell, supercell=MaxNumAtoms(max_num_atoms=64))
