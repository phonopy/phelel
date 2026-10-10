"""Tests of reading and writing velph.toml and velph init templates."""

from __future__ import annotations

import dataclasses
import io
import pathlib
from typing import Any

import pytest
import tomli
from phonopy.structure.atoms import PhonopyAtoms, get_cell_dict

from phelel.velph.cli.init.init import _merge_incar_commons
from phelel.velph.cli.utils import get_default_amplitude
from phelel.velph.config.defaults import get_default_config
from phelel.velph.config.io import (
    VelphConfigError,
    dumps_config,
    load_template,
    load_velph_toml,
    parse_template,
    parse_velph_toml,
    write_velph_toml,
)
from phelel.velph.config.merge import merge
from phelel.velph.config.schema import (
    CODES,
    INPUT_SET_FIELDS,
    CodeSection,
    Dimension,
    InputSetSection,
    LinePath,
    Matrix,
    MaxNumAtoms,
    Mesh,
    Spacing,
    SymmetrySection,
    VaspSection,
    VelphConfig,
)
from phelel.velph.templates import default_template_dict


def _resolved_config(cell: PhonopyAtoms) -> VelphConfig:
    """Return a complete velph.toml config.

    It is made of the defaults and of values that velph init would compute.

    """
    path = (((0.0, 0.0, 0.0), (0.5, 0.0, 0.5)), ((0.5, 0.0, 0.5), (0.5, 0.25, 0.75)))
    label = (("GAMMA", "X"), ("X", "W"))
    input_sets = {}
    for set_name, fields in INPUT_SET_FIELDS.items():
        values: dict[str, Any] = {"kpoints": Mesh(mesh=(4, 4, 4))}
        if "cell" in fields:
            values["cell"] = "primitive"
        if "kpoints_dense" in fields:
            values["kpoints_dense"] = Mesh(mesh=(8, 8, 8), shift=(0.5, 0.5, 0.5))
        if "kpoints_opt" in fields:
            values["kpoints_opt"] = LinePath(line=51, path=path, label=label)
        if "qpoints" in fields:
            values["qpoints"] = LinePath(line=101)
        input_sets[set_name] = InputSetSection(**values)
    input_sets["phelel"] = InputSetSection(
        kpoints=Mesh(mesh=((-2, 2, 2), (2, -2, 2), (2, 2, -2))),
        scheduler={"job_name": "phelel-job"},
    )
    resolved = VelphConfig(
        version="0.15.0",
        phelel=CodeSection(supercell=Dimension(supercell_dimension=(2, 2, 2))),
        phonopy=CodeSection(
            supercell=Matrix(supercell_matrix=((0, 2, 2), (2, 0, 2), (2, 2, 0))),
            number_of_snapshots=10,
        ),
        phono3py=CodeSection(supercell=Dimension(supercell_dimension=(1, 1, 1))),
        vasp=VaspSection(input_sets=input_sets),
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
    return dataclasses.replace(config, init=None)


def _velph_toml_text(cell: PhonopyAtoms) -> str:
    return dumps_config(_resolved_config(cell))


def _parse_text(text: str) -> VelphConfig:
    return parse_velph_toml(tomli.loads(text))


def _template(text: str) -> VelphConfig:
    return parse_template(tomli.loads(text))


def test_defaults_amplitude():
    """Default amplitudes are those of phonopy and phono3py for VASP."""
    defaults = get_default_config()
    for code in CODES:
        assert getattr(defaults, code).amplitude == pytest.approx(
            get_default_amplitude(code)
        )


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


def test_defaults_match_default_template_dict():
    """The defaults give the INCARs that the present velph init gives.

    The effective INCAR of each input set, the base [vasp.incar] without
    incar_unset and with the tags of the input set, is compared with the INCAR that
    _merge_incar_commons builds from default_template_dict.

    """
    vasp = get_default_config().vasp
    assert vasp is not None and vasp.incar is not None and vasp.input_sets is not None
    old_vasp = default_template_dict["vasp"]
    renamed = {
        "el_bands.bands": "el_bands",
        "el_bands.dos": "el_dos",
    }
    for old_name in old_vasp:
        if old_name == "incar":
            continue
        set_name = renamed.get(old_name, old_name)
        input_set = vasp.input_sets[set_name]
        unset = input_set.incar_unset or ()
        effective = {
            tag: value for tag, value in vasp.incar.items() if tag not in unset
        } | (input_set.incar or {})
        expected = _merge_incar_commons(old_vasp[old_name]["incar"], old_vasp["incar"])
        assert effective == expected, set_name
    assert vasp.input_sets["el_bands"].kpoints_opt == LinePath(line=51)
    assert vasp.input_sets["ph_bands"].qpoints == LinePath(line=51)
    assert vasp.input_sets["ph_bands"].kpoints == Mesh(mesh=(1, 1, 1))
    assert vasp.input_sets["relax"].cell == "unitcell"
    assert vasp.input_sets["nac"].cell == "primitive"
    assert get_default_config().scheduler == default_template_dict["scheduler"]


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


def test_dumps_format(nacl_cell: PhonopyAtoms):
    """Matrices are written with one row per line, and points with numbers."""
    text = _velph_toml_text(nacl_cell)
    assert "supercell_matrix = [\n  [0, 2, 2],\n  [2, 0, 2],\n  [2, 2, 0],\n]" in text
    assert "supercell_dimension = [2, 2, 2]" in text
    assert 'scheduler_template = """' in text
    assert "[[unitcell.points]]  # 1" in text
    assert "[[unitcell.points]]  # 2" in text
    assert max(len(line) for line in text.split("\n") if "temps" not in line) <= 88


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
    assert config.vasp.input_sets["relax"].kpoints == Spacing(
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
    assert config.init.supercell == Matrix(
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


def test_incar_empty_table_in_velph_toml(nacl_cell: PhonopyAtoms):
    """An INCAR tag set to {} is an error that points to incar_unset."""
    text = _velph_toml_text(nacl_cell).replace(
        "[vasp.incar]\n", "[vasp.incar]\nncore = {}\n"
    )
    with pytest.raises(VelphConfigError) as e:
        _parse_text(text)
    assert '"ncore" has a table as value.  A section does not inherit' in str(e.value)


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
    document = tomli.loads(text)
    document["unitcell"]["lattice"] = [[1.0, 0.0, 0.0]]
    with pytest.raises(VelphConfigError) as e:
        parse_velph_toml(document)
    assert "[unitcell] Invalid crystal structure" in str(e.value)


def test_old_velph_toml_digest():
    """An old velph.toml gives a digest of only the changes that concern it."""
    text = (
        "[phelel]\nfft_mesh = [18, 18, 18]\n"
        "[vasp.relax.kpoints]\nkspacing = 0.2\n"
        "[vasp.el_bands.dos.incar]\nnedos = 5001\n"
    )
    with pytest.raises(VelphConfigError) as e:
        parse_velph_toml(tomli.loads(text), name="old/velph.toml")
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
    )
    with pytest.raises(VelphConfigError) as e:
        parse_template(tomli.loads(text), name="tmpl.toml")
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
