"""Tests of reading and writing velph.toml and velph init templates."""

from __future__ import annotations

import dataclasses
import io
import pathlib
import tomllib
from typing import Any, cast

import numpy as np
import pytest
from phono3py.interface.calculator import (
    get_default_displacement_distance as get_phono3py_displacement_distance,
)
from phonopy.interface.calculator import (
    get_default_displacement_distance as get_phonopy_displacement_distance,
)
from phonopy.structure.atoms import PhonopyAtoms, get_cell_dict, parse_cell_dict
from phonopy.structure.cells import apply_site_mixture, build_mixture_cell

from phelel.velph.config.defaults import get_default_config
from phelel.velph.config.input_sets import INPUT_SETS, input_set_fields
from phelel.velph.config.io import (
    load_template,
    load_velph_toml,
    parse_template,
    parse_velph_toml,
)
from phelel.velph.config.merge import merge
from phelel.velph.config.resolve import resolve_incar
from phelel.velph.config.schema import (
    CODES,
    CodeSection,
    InputSetSection,
    KpointsMesh,
    KpointsSpacing,
    LinePath,
    MaxNumAtoms,
    SupercellDimension,
    SupercellMatrix,
    SymmetrySection,
    VaspSection,
    VelphConfig,
)
from phelel.velph.config.toml_writer import dumps_config, write_velph_toml
from phelel.velph.errors import VelphConfigError


def _resolved_config(cell: PhonopyAtoms) -> VelphConfig:
    """Return a complete velph.toml config.

    It is made of the defaults, of [scheduler] that a template would give, and
    of values that velph init would compute, with the INCAR tags of each input
    set resolved.

    """
    path = (((0.0, 0.0, 0.0), (0.5, 0.0, 0.5)), ((0.5, 0.0, 0.5), (0.5, 0.25, 0.75)))
    label = (("GAMMA", "X"), ("X", "W"))
    input_sets: dict[str, InputSetSection] = {}
    for set_name in INPUT_SETS:
        fields = input_set_fields(set_name)
        values: dict[str, Any] = {"kpoints": KpointsMesh(mesh=(4, 4, 4))}
        if "cell" in fields:
            values["cell"] = "primitive"
        if "kpoints_dense" in fields:
            values["kpoints_dense"] = KpointsMesh(mesh=(8, 8, 8), shift=(0.5, 0.5, 0.5))
        if "kpoints_opt" in fields:
            values["kpoints_opt"] = LinePath(line=51, path=path, label=label)
        if "qpoints" in fields:
            values["qpoints"] = LinePath(line=101)
        input_sets[set_name] = InputSetSection(**values)
    input_sets["phelel"] = InputSetSection(
        kpoints=KpointsMesh(mesh=((-2, 2, 2), (2, -2, 2), (2, 2, -2))),
        scheduler={"job_name": "phelel-job"},
    )
    resolved = VelphConfig(
        version="0.15.0",
        phelel=CodeSection(supercell=SupercellDimension(supercell_dimension=(2, 2, 2))),
        phonopy=CodeSection(
            supercell=SupercellMatrix(
                supercell_matrix=((0, 2, 2), (2, 0, 2), (2, 2, 0))
            ),
            number_of_snapshots=10,
        ),
        phono3py=CodeSection(
            supercell=SupercellDimension(supercell_dimension=(1, 1, 1))
        ),
        vasp=VaspSection(input_sets=input_sets),
        scheduler={
            "job_name": "vasp-elph",
            "scheduler_template": "#!/bin/bash\n#SBATCH --job-name={job_name}\n",
        },
        symmetry=SymmetrySection(
            tolerance=1e-5,
            spacegroup_type="Fm-3m",
            primitive_matrix=((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),
        ),
        unitcell=get_cell_dict(cell),
        primitive_cell=get_cell_dict(cell),
    )
    config = merge(get_default_config(), resolved)
    assert config is not None
    return dataclasses.replace(resolve_incar(config), init=None)


def _velph_toml_text(cell: PhonopyAtoms) -> str:
    return dumps_config(_resolved_config(cell))


def _parse_text(text: str) -> VelphConfig:
    return parse_velph_toml(tomllib.loads(text))


def _template(text: str) -> VelphConfig:
    return parse_template(tomllib.loads(text))


def test_defaults_amplitude():
    """Default amplitudes are those of phonopy and phono3py for VASP.

    phelel uses the default of phono3py.

    """
    defaults = get_default_config()
    expected = {
        "phelel": get_phono3py_displacement_distance("vasp"),
        "phonopy": get_phonopy_displacement_distance("vasp"),
        "phono3py": get_phono3py_displacement_distance("vasp"),
    }
    for code in CODES:
        assert getattr(defaults, code).amplitude == pytest.approx(expected[code])


def test_defaults_init_acts_on_no_section():
    """[init] of the defaults holds only values used by velph init itself."""
    init = get_default_config().init
    assert init is not None
    for name in (
        "supercell",
        "amplitude",
        "diagonal",
        "plusminus",
        "nosym",
        "cell_for_relax",
        "cell_for_nac",
    ):
        assert getattr(init, name) is None


def test_round_trip(nacl_cell: PhonopyAtoms, tmp_path: pathlib.Path):
    """What is written is read back to the same config."""
    config = _resolved_config(nacl_cell)
    text = dumps_config(config)
    assert _parse_text(text) == config
    assert dumps_config(_parse_text(text)) == text

    filename = tmp_path / "velph.toml"
    write_velph_toml(config, filename)
    assert load_velph_toml(filename) == config


def test_round_trip_template():
    """A template with max_num_atoms in it is read back to the same config."""
    config = get_default_config()
    config = dataclasses.replace(
        config,
        phelel=dataclasses.replace(
            config.phelel, supercell=MaxNumAtoms(max_num_atoms=120)
        ),
    )
    assert _template(dumps_config(config)) == config


def _cells_for_round_trip() -> dict[str, PhonopyAtoms]:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    two_sites = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    co_located = PhonopyAtoms(
        cell=lattice,
        symbols=["Ge", "Sn", "Si"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    return {
        "plain": PhonopyAtoms(
            cell=lattice, symbols=["Na", "Cl"], scaled_positions=two_sites
        ),
        "magmom_scalar": PhonopyAtoms(
            cell=lattice,
            symbols=["Fe", "Fe"],
            scaled_positions=two_sites,
            magnetic_moments=[1.5, -1.5],
        ),
        "magmom_vector": PhonopyAtoms(
            cell=lattice,
            symbols=["Fe", "Fe"],
            scaled_positions=two_sites,
            magnetic_moments=[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        ),
        "extended_symbol": PhonopyAtoms(
            cell=lattice, symbols=["Fe1", "Fe2"], scaled_positions=two_sites
        ),
        "merged_mixture": build_mixture_cell(co_located, [0.5, 0.5, 1.0]),
        "weighted_species": apply_site_mixture(co_located, [0.5, 0.5, 1.0]),
    }


@pytest.mark.parametrize("name", list(_cells_for_round_trip()))
def test_round_trip_cell(name: str):
    """A cell written in velph.toml is read back by parse_cell_dict of phonopy.

    Merged mixed-species sites, extended symbols, magnetic moments and
    weighted species are written as phonopy writes them in yaml.

    """
    cell = _cells_for_round_trip()[name]
    config = _parse_text(_velph_toml_text(cell))
    assert config.unitcell is not None
    cell2 = parse_cell_dict(cast(Any, config.unitcell))
    assert cell2 is not None
    assert cell2.symbols == cell.symbols
    np.testing.assert_allclose(cell2.cell, cell.cell)
    np.testing.assert_allclose(cell2.scaled_positions, cell.scaled_positions)
    assert cell.masses is not None
    assert cell2.masses is not None
    np.testing.assert_allclose(cell2.masses, cell.masses)
    assert cell2.has_mixtures == cell.has_mixtures
    assert cell2.has_weighted_species == cell.has_weighted_species
    if cell.mixture_weights is None:
        assert cell2.mixture_weights is None
    else:
        assert cell2.mixture_weights is not None
        np.testing.assert_allclose(cell2.mixture_weights, cell.mixture_weights)
    if cell.magnetic_moments is None:
        assert cell2.magnetic_moments is None
    else:
        assert cell2.magnetic_moments is not None
        np.testing.assert_allclose(cell2.magnetic_moments, cell.magnetic_moments)


def test_cell_old_extended_symbol():
    """Symbol = "Fe1" written by velph earlier than 0.15 is still read."""
    toml_dict = tomllib.loads(
        "\n".join(
            [
                "[unitcell]",
                "lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]",
                "[[unitcell.points]]",
                'symbol = "Fe1"',
                "coordinates = [0.0, 0.0, 0.0]",
                "[[unitcell.points]]",
                'symbol = "Fe2"',
                "coordinates = [0.5, 0.5, 0.5]",
            ]
        )
    )
    cell = parse_cell_dict(toml_dict["unitcell"])
    assert cell is not None
    assert cell.symbols == ["Fe1", "Fe2"]


def test_dumps_format(nacl_cell: PhonopyAtoms):
    """Matrices are written with one row per line, and points with numbers."""
    text = _velph_toml_text(nacl_cell)
    assert "supercell_matrix = [\n  [0, 2, 2],\n  [2, 0, 2],\n  [2, 2, 0],\n]" in text
    assert "supercell_dimension = [2, 2, 2]" in text
    assert 'scheduler_template = """' in text
    assert "[[unitcell.points]]  # 1" in text
    assert "[[unitcell.points]]  # 2" in text
    assert max(len(line) for line in text.split("\n") if "temps" not in line) <= 88


def test_dumps_blocks(nacl_cell: PhonopyAtoms):
    """A blank line is written only before a block.

    [vasp.relax] with its incar and kpoints is one block.  [vasp.phelel.incar]
    and [vasp.phelel.kpoints] are one block, although [vasp.phelel] has no
    header.  [unitcell] with its points is one block.

    """
    text = _velph_toml_text(nacl_cell)
    for header in ("[vasp.relax]", "[vasp.phelel.incar]", "[unitcell]"):
        assert f"\n\n{header}\n" in text
    for header in (
        "[vasp.relax.incar]",
        "[vasp.phelel.kpoints]",
        "[[unitcell.points]]",
    ):
        assert f"\n{header}" in text
        assert f"\n\n{header}" not in text


def test_dumps_negative_zero(nacl_cell: PhonopyAtoms):
    """-0.0 in the basis vectors is written as 0.0."""
    cell = nacl_cell.copy()
    cell.cell = [[5.69, -0.0, -0.0], [0.0, 5.69, -0.0], [0.0, 0.0, 5.69]]
    text = _velph_toml_text(cell)
    assert "[5.69, 0.0, 0.0]," in text
    assert "-0.0" not in text


def test_dumps_long_element(nacl_cell: PhonopyAtoms):
    """A list with an element longer than the line width is written per line.

    Each element is on its own line, and no line has only spaces.

    """
    config = _resolved_config(nacl_cell)
    assert config.scheduler is not None
    names = ["x" * 85, "y"]
    config = dataclasses.replace(config, scheduler={**config.scheduler, "names": names})
    text = dumps_config(config)
    assert f'names = [\n  "{"x" * 85}",\n  "y",\n]' in text
    assert all(line.strip() or not line for line in text.split("\n"))
    assert _parse_text(text).scheduler["names"] == names  # type: ignore[index]


def test_dumps_quoted_keys(nacl_cell: PhonopyAtoms):
    """A key that is not a bare key of TOML is written in quotes."""
    config = _resolved_config(nacl_cell)
    assert config.scheduler is not None
    keys = {"a b": 1, "a.b": 2, "a = b": 3, 'x"y': 4, "job-name": 5}
    config = dataclasses.replace(config, scheduler={**config.scheduler, **keys})
    text = dumps_config(config)
    assert '"a = b" = 3' in text
    assert "job-name = 5" in text
    scheduler = _parse_text(text).scheduler
    assert scheduler is not None
    assert {key: scheduler[key] for key in keys} == keys


def test_write_velph_toml_checks(tmp_path: pathlib.Path):
    """A config that is not a complete velph.toml is not written."""
    filename = tmp_path / "velph.toml"
    with pytest.raises(VelphConfigError):
        write_velph_toml(get_default_config(), filename)
    assert not filename.exists()


@pytest.mark.parametrize(
    "text,message",
    [
        ("[foo]\na = 1", "Unknown section [foo]"),
        (
            "[phelel]\nsupercell_dimension = [2, 2, 2]\nbar = 1",
            '[phelel] Unknown key "bar"',
        ),
        (
            "[phelel]\nnumber_of_snapshots = 2",
            '[phelel] Unknown key "number_of_snapshots"',
        ),
        (
            '[phono3py]\nsite_mixture_scheme = "merge"',
            '[phono3py] Unknown key "site_mixture_scheme"',
        ),
        ("[vasp.foo.incar]\nencut = 500", '[vasp] Unknown key "foo"'),
        ("[vasp.relax.qpoints]\nline = 51", '[vasp.relax] Unknown key "qpoints"'),
        (
            "[vasp.relax.kpoints]\nmesh = [1, 1, 1]\nD_diag = [1, 1, 1]",
            '[vasp.relax.kpoints] Unknown key "D_diag"',
        ),
        ("[vasp.relax.incar.sub]\na = 1", '"sub" has a table as value'),
        ("[symmetry]\ntolerance = 1e-5", "Unknown section [symmetry]"),
        ('[velph]\nversion = "0.15.0"', "Unknown section [velph]"),
    ],
)
def test_template_unknown_keys(text: str, message: str):
    """Unknown keys and sections are an error in a template."""
    with pytest.raises(VelphConfigError) as e:
        _template(text)
    assert message in str(e.value)


def test_velph_toml_unknown_section(nacl_cell: PhonopyAtoms):
    """[init] and unknown sections are an error in a velph.toml."""
    text = _velph_toml_text(nacl_cell)
    for extra in ("[init]\ntolerance = 1e-5", "[foo]\na = 1"):
        with pytest.raises(VelphConfigError) as e:
            _parse_text(text + "\n" + extra)
        assert "Unknown section" in str(e.value)


def test_template_intent_forms():
    """A template accepts max_num_atoms and kspacing."""
    config = _template(
        "[phelel]\nmax_num_atoms = 120\n"
        "[vasp.relax.kpoints]\nkspacing = 0.2\nshift = [0.5, 0.5, 0.5]\n"
    )
    assert config.phelel == CodeSection(supercell=MaxNumAtoms(max_num_atoms=120))
    assert config.vasp is not None and config.vasp.input_sets is not None
    assert config.vasp.input_sets["relax"].kpoints == KpointsSpacing(
        kspacing=0.2, shift=(0.5, 0.5, 0.5)
    )


@pytest.mark.parametrize(
    "old,new",
    [
        ("supercell_dimension = [2, 2, 2]", "max_num_atoms = 120"),
        (
            "[vasp.relax.kpoints]\nmesh = [4, 4, 4]",
            "[vasp.relax.kpoints]\nkspacing = 0.2",
        ),
    ],
)
def test_velph_toml_rejects_intent_forms(nacl_cell: PhonopyAtoms, old: str, new: str):
    """max_num_atoms and kspacing are not allowed in a velph.toml."""
    text = _velph_toml_text(nacl_cell)
    assert old in text
    with pytest.raises(VelphConfigError) as e:
        _parse_text(text.replace(old, new, 1))
    assert "cannot be given in velph.toml" in str(e.value)


def test_supercell_keys():
    """Two supercell keys in one table are an error.

    A list of nine integers is read as a 3x3 matrix.

    """
    with pytest.raises(VelphConfigError) as e:
        _template("[phelel]\nmax_num_atoms = 120\nsupercell_dimension = [2, 2, 2]")
    assert "Give only one of" in str(e.value)
    config = _template("[init]\nsupercell_matrix = [0, 2, 2, 2, 0, 2, 2, 2, 0]")
    assert config.init is not None
    assert config.init.supercell == SupercellMatrix(
        supercell_matrix=((0, 2, 2), (2, 0, 2), (2, 2, 0))
    )


def test_kpoints_block():
    """A k-point block has to give exactly one of mesh and kspacing."""
    for block, message in (
        ("mesh = [4, 4, 4]\nkspacing = 0.2", 'Give either "mesh" or "kspacing"'),
        ("shift = [0.5, 0.5, 0.5]", 'Give "mesh" or "kspacing"'),
        ("mesh = [4, 4]", '"mesh" has to be three integers'),
        ("mesh = [4.0, 4, 4]", '"mesh" has to be an integer'),
    ):
        with pytest.raises(VelphConfigError) as e:
            _template(f"[vasp.relax.kpoints]\n{block}")
        assert message in str(e.value)


def test_incar():
    """INCAR tag names are read in lower case, and some tags are errors."""
    config = _template("[vasp.incar]\nENCUT = 600\n[vasp.relax]\nincar_unset = ['NSW']")
    assert config.vasp is not None
    assert config.vasp.incar == {"encut": 600}
    assert config.vasp.input_sets is not None
    assert config.vasp.input_sets["relax"].incar_unset == ("nsw",)
    for incar, message in (
        ("ENCUT = 600\nencut = 500", 'INCAR tag "encut" is given twice'),
        ("KSPACING = 0.2", 'INCAR tag "KSPACING" cannot be used'),
        ("elph_kspacing = 0.2", 'INCAR tag "elph_kspacing" cannot be used'),
    ):
        with pytest.raises(VelphConfigError) as e:
            _template(f"[vasp.incar]\n{incar}")
        assert message in str(e.value)


def test_incar_table_value():
    """In a template, an INCAR tag with a table as value is an error.

    The message points to incar_unset.  An empty table, the form of velph
    earlier than 0.15, is reported by the digest of old forms instead.

    """
    with pytest.raises(VelphConfigError) as e:
        _template("[vasp.incar]\nncore = {value = 1}")
    assert '"ncore" has a table as value.  A section does not inherit' in str(e.value)


def test_incar_empty_table_in_velph_toml(nacl_cell: PhonopyAtoms):
    """In a velph.toml, an INCAR tag set to {} is an error that asks to remove it."""
    text = _velph_toml_text(nacl_cell).replace(
        "[vasp.relax.incar]\n", "[vasp.relax.incar]\nncore = {}\n"
    )
    assert "ncore = {}" in text
    with pytest.raises(VelphConfigError) as e:
        _parse_text(text)
    assert str(e.value) == (
        '"velph.toml": [vasp.relax.incar] "ncore" has a table as value.  '
        'Remove "ncore".'
    )


def test_velph_toml_rejects_shared_incar(nacl_cell: PhonopyAtoms):
    """[vasp.incar] and incar_unset are not allowed in a velph.toml."""
    text = _velph_toml_text(nacl_cell)
    assert "[vasp.incar]" not in text and "incar_unset" not in text
    with pytest.raises(VelphConfigError) as e:
        _parse_text(text + "\n[vasp.incar]\nencut = 500\n")
    assert str(e.value) == (
        '"velph.toml": [vasp.incar] This table cannot be given in '
        "velph.toml.  Each input set of "
        "velph.toml holds all of its INCAR tags in [vasp.<input set>.incar].  "
        "Write the tags there, or give [vasp.incar] in a velph init template and "
        'run "velph init" again.'
    )
    text = text.replace("[vasp.nac]\n", '[vasp.nac]\nincar_unset = ["npar"]\n')
    assert "incar_unset" in text
    with pytest.raises(VelphConfigError) as e:
        _parse_text(text)
    assert str(e.value) == (
        '"velph.toml": [vasp.nac] "incar_unset" cannot be given in '
        "velph.toml.  It removes tags "
        "of [vasp.incar], which only a velph init template has.  Remove the tags "
        'from [vasp.nac.incar], or give "incar_unset" in a velph init template and '
        'run "velph init" again.'
    )


def test_choices():
    """Choices are compared in lower case.

    A value that is not one of the choices is an error.

    """
    config = _template('[vasp.nac]\ncell = "Primitive"\n[phelel]\nplusminus = "AUTO"')
    assert config.vasp is not None and config.vasp.input_sets is not None
    assert config.vasp.input_sets["nac"].cell == "primitive"
    assert config.phelel is not None and config.phelel.plusminus == "auto"
    for text, message in (
        ('[vasp.nac]\ncell = "primitive_cell"', '"cell" has to be one of'),
        ('[init]\nprimitive_cell_choice = "std"', '"primitive_cell_choice" has to be'),
        ('[phelel]\nplusminus = "yes"', '"plusminus" has to be true, false or "auto"'),
        ("[phelel]\namplitude = true", '"amplitude" has to be a number'),
    ):
        with pytest.raises(VelphConfigError) as e:
            _template(text)
        assert message in str(e.value)


def test_init_codes():
    """The order of codes is kept, and a code given twice is taken once."""
    config = _template('[init]\ncodes = ["phono3py", "phelel", "phono3py"]')
    assert config.init is not None
    assert config.init.codes == ("phono3py", "phelel")
    with pytest.raises(VelphConfigError) as e:
        _template('[init]\ncodes = ["phonon"]')
    assert '"codes" has to be one of' in str(e.value)


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("[unitcell]", "[unitcell_]", "Unknown section [unitcell_]"),
        (
            'version = "0.15.0"',
            'version = "0.15.0"\nfoo = 1',
            '[velph] Unknown key "foo"',
        ),
        (
            "supercell_dimension = [2, 2, 2]\n",
            "",
            '[phelel] "supercell_dimension" or "supercell_matrix" is required.',
        ),
        ('[vasp.relax]\ncell = "primitive"\n', "[vasp.relax]\n", '[vasp.relax] "cell"'),
        (
            "[vasp.el_bands.kpoints_opt]",
            "[vasp.el_bands.qpoints_]",
            '[vasp.el_bands] Unknown key "qpoints_"',
        ),
    ],
)
def test_velph_toml_required(nacl_cell: PhonopyAtoms, old: str, new: str, message: str):
    """Missing required sections, keys and blocks are errors in a velph.toml."""
    text = _velph_toml_text(nacl_cell)
    assert old in text
    with pytest.raises(VelphConfigError) as e:
        _parse_text(text.replace(old, new, 1))
    assert message in str(e.value)


def test_velph_toml_required_blocks(nacl_cell: PhonopyAtoms):
    """Every k-point block of an input set is required in a velph.toml."""
    config = _resolved_config(nacl_cell)
    assert config.vasp is not None and config.vasp.input_sets is not None
    for set_name, block in (
        ("el_bands", "kpoints_opt"),
        ("selfenergy", "kpoints_dense"),
    ):
        input_sets = dict(config.vasp.input_sets)
        input_sets[set_name] = dataclasses.replace(
            input_sets[set_name], **{block: None}
        )
        vasp = dataclasses.replace(config.vasp, input_sets=input_sets)
        text = dumps_config(dataclasses.replace(config, vasp=vasp))
        with pytest.raises(VelphConfigError) as e:
            _parse_text(text)
        assert f"[vasp.{set_name}.{block}] This block is required." in str(e.value)


def test_invalid_cell(nacl_cell: PhonopyAtoms):
    """A cell that phonopy does not read is an error."""
    text = _velph_toml_text(nacl_cell)
    document = tomllib.loads(text)
    document["unitcell"]["lattice"] = [[1.0, 0.0, 0.0]]
    with pytest.raises(VelphConfigError) as e:
        parse_velph_toml(document)
    assert "[unitcell] Invalid crystal structure" in str(e.value)


@pytest.mark.parametrize(
    "path,key,value,message",
    [
        (
            ("scheduler",),
            "scheduler_name",
            "sge",
            '[scheduler] "scheduler_name" is removed.',
        ),
        (
            ("vasp", "nac", "scheduler"),
            "scheduler_name",
            "sge",
            '[vasp.nac.scheduler] "scheduler_name" is removed.',
        ),
        (
            ("scheduler",),
            "custom_template",
            "#!/bin/bash",
            '[scheduler] "custom_template" is removed.',
        ),
        (
            ("scheduler",),
            "scheduler_template",
            1,
            '[scheduler] "scheduler_template" has to be a string',
        ),
    ],
)
def test_scheduler_errors(
    nacl_cell: PhonopyAtoms, path: tuple[str, ...], key: str, value: Any, message: str
):
    """scheduler_name and custom_template are an error in the scheduler settings.

    scheduler_template has to be a string.

    """
    document = tomllib.loads(_velph_toml_text(nacl_cell))
    table = document
    for name in path:
        table = table.setdefault(name, {})
    table[key] = value
    with pytest.raises(VelphConfigError) as e:
        parse_velph_toml(document)
    assert message in str(e.value)


def test_old_velph_toml_digest():
    """An old velph.toml gives a digest of only the changes that concern it."""
    text = (
        "[phelel]\nfft_mesh = [18, 18, 18]\n"
        "[vasp.relax.kpoints]\nkspacing = 0.2\n"
        "[vasp.el_bands.dos.incar]\nnedos = 5001\n"
        '[vasp.relax.scheduler]\nscheduler_name = "sge"\n'
    )
    with pytest.raises(VelphConfigError) as e:
        parse_velph_toml(tomllib.loads(text), name="old/velph.toml")
    message = str(e.value)
    assert message.startswith(
        '"old/velph.toml" was written by velph earlier than 0.15 and cannot be read.'
    )
    # The list is wrapped at 76 columns, so the contents are checked with the
    # line breaks replaced by spaces.
    flat = " ".join(message.split())
    assert "Changes that concern this file:" in flat
    assert "[vasp.el_bands.bands] and [vasp.el_bands.dos] are now" in flat
    assert "[phelel] fft_mesh is removed." in flat
    assert "converts kspacing in a k-point block to mesh" in flat
    assert '"scheduler_name" is removed.' in flat
    assert "incar_unset" not in flat
    assert message.endswith(
        'See "Changes in velph 0.15" in the documentation for all changes.'
    )
    assert max(len(line) for line in message.split("\n")) <= 80


def test_old_velph_toml_without_old_forms():
    """An old velph.toml without the old keys gives no list of changes."""
    with pytest.raises(VelphConfigError) as e:
        parse_velph_toml({"vasp": {"relax": {}}})
    message = str(e.value)
    assert "earlier than 0.15" in message
    assert "Changes that concern this file" not in message


def test_old_template_digest():
    """A template with old keys gives the list of changes that concern it.

    kspacing in a k-point block is still valid in a template, and the list
    does not mention it.

    """
    text = (
        "[init.options]\nkspacing = 0.2\n"
        "[vasp.nac.incar]\nncore = {}\n"
        "[vasp.relax.kpoints]\nkspacing = 0.2\n"
        '[scheduler]\nscheduler_name = "sge"\n'
    )
    with pytest.raises(VelphConfigError) as e:
        parse_template(tomllib.loads(text), name="tmpl.toml")
    message = str(e.value)
    assert message.startswith(
        '"tmpl.toml" is a velph init template of velph earlier than 0.15.'
    )
    assert max(len(line) for line in message.split("\n")) <= 80
    flat = " ".join(message.split())
    assert "[init.options] is now [init]" in flat
    assert (
        'does not inherit the tags of [vasp.incar] listed in its "incar_unset"' in flat
    )
    assert '"scheduler_name" is removed.' in flat
    assert "converts kspacing" not in flat


def test_load_template_file_and_bytesio(tmp_path: pathlib.Path):
    """A template is read from a file name and from a binary file object."""
    text = '[init]\ncodes = ["phelel"]\n[vasp.incar]\nENCUT = 600\n'
    filename = tmp_path / "velph-tmpl.toml"
    filename.write_text(text)
    assert load_template(filename) == load_template(io.BytesIO(text.encode()))


def test_toml_decode_error(tmp_path: pathlib.Path):
    """A file that is not TOML gives an error with its name."""
    filename = tmp_path / "velph-tmpl.toml"
    filename.write_text("[init\n")
    with pytest.raises(VelphConfigError) as e:
        load_template(filename)
    assert f'Error in reading "{filename}"' in str(e.value)
