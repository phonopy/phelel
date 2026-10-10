"""Tests of the merge of layers of velph settings."""

from __future__ import annotations

import pytest
import tomli

from phelel.velph.config.defaults import get_default_config
from phelel.velph.config.io import parse_template
from phelel.velph.config.merge import merge
from phelel.velph.config.schema import (
    CodeSection,
    Dimension,
    InputSetSection,
    Matrix,
    MaxNumAtoms,
    Mesh,
    Spacing,
    VaspSection,
    VelphConfig,
)


def _template(text: str) -> VelphConfig:
    return parse_template(tomli.loads(text))


def test_merge_none():
    """A layer that is None, or a field that is None, takes the other value."""
    config = VelphConfig(phelel=CodeSection(amplitude=0.02))
    assert merge(None, config) == config
    assert merge(config, None) == config
    assert merge(config, VelphConfig()) == config
    assert merge(None, None) is None


def test_merge_supercell_is_atomic():
    """A supercell of the upper layer replaces the lower one as a whole."""
    lower = CodeSection(
        supercell=Matrix(supercell_matrix=((0, 2, 2), (2, 0, 2), (2, 2, 0))),
        amplitude=0.03,
    )
    upper = CodeSection(supercell=Dimension(supercell_dimension=(2, 2, 2)))
    assert merge(lower, upper) == CodeSection(
        supercell=Dimension(supercell_dimension=(2, 2, 2)), amplitude=0.03
    )
    upper = CodeSection(supercell=MaxNumAtoms(max_num_atoms=120))
    merged = merge(lower, upper)
    assert merged is not None
    assert merged.supercell == MaxNumAtoms(max_num_atoms=120)


def test_merge_kpoints_is_atomic():
    """A k-point block of the upper layer replaces the whole lower block.

    The shift of the lower block is not kept.

    """
    lower = InputSetSection(kpoints=Spacing(kspacing=0.1, shift=(0.5, 0.5, 0.5)))
    upper = InputSetSection(kpoints=Mesh(mesh=(4, 4, 4)))
    assert merge(lower, upper) == InputSetSection(kpoints=Mesh(mesh=(4, 4, 4)))


def test_merge_incar_and_scheduler_key_by_key():
    """INCAR and scheduler tables are merged key by key."""
    lower = VelphConfig(
        vasp=VaspSection(incar={"encut": 500, "prec": "accurate"}),
        scheduler={"job_name": "a", "nodes": 1},
    )
    upper = VelphConfig(
        vasp=VaspSection(incar={"encut": 600}), scheduler={"job_name": "b"}
    )
    merged = merge(lower, upper)
    assert merged is not None and merged.vasp is not None
    assert merged.vasp.incar == {"encut": 600, "prec": "accurate"}
    assert merged.scheduler == {"job_name": "b", "nodes": 1}


def test_merge_input_sets_one_by_one():
    """An input set in both layers is merged field by field.

    An input set in only one layer is kept.

    """
    lower = VaspSection(
        input_sets={"relax": InputSetSection(cell="unitcell", incar={"nsw": 10})},
    )
    upper = VaspSection(
        input_sets={
            "relax": InputSetSection(incar={"isif": 2}, kpoints=Mesh(mesh=(4, 4, 4))),
            "nac": InputSetSection(cell="primitive"),
        },
    )
    merged = merge(lower, upper)
    assert merged is not None and merged.input_sets is not None
    assert merged.input_sets["relax"] == InputSetSection(
        cell="unitcell", incar={"nsw": 10, "isif": 2}, kpoints=Mesh(mesh=(4, 4, 4))
    )
    assert merged.input_sets["nac"] == InputSetSection(cell="primitive")


def test_merge_template_over_defaults():
    """A template merged over the defaults changes the values that it gives.

    The other default values are kept.

    """
    defaults = get_default_config()
    template = _template(
        "[vasp.incar]\nENCUT = 600\n"
        "[vasp.relax]\nincar_unset = ['ediff']\n"
        "[scheduler]\njob_name = 'PbTe'\n"
    )
    merged = merge(defaults, template)
    assert merged is not None and merged.vasp is not None
    assert defaults.vasp is not None and defaults.vasp.incar is not None
    assert merged.vasp.incar == defaults.vasp.incar | {"encut": 600}
    assert merged.vasp.input_sets is not None and defaults.vasp.input_sets is not None
    assert merged.vasp.input_sets["relax"].incar_unset == ("ediff",)
    assert (
        merged.vasp.input_sets["relax"].incar == defaults.vasp.input_sets["relax"].incar
    )
    assert merged.scheduler is not None and defaults.scheduler is not None
    assert merged.scheduler["job_name"] == "PbTe"
    assert (
        merged.scheduler["scheduler_template"]
        == (defaults.scheduler["scheduler_template"])
    )
    assert merged.phelel == defaults.phelel


def test_merge_type_mismatch():
    """Sections of different types are not merged."""
    with pytest.raises(TypeError):
        merge(CodeSection(), InputSetSection())
