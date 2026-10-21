"""Tests of the resolution of settings that velph init makes."""

from __future__ import annotations

from phelel.velph.config.resolve import resolve_incar
from phelel.velph.config.schema import InputSetSection, VaspSection, VelphConfig


def test_resolve_incar():
    """Tags of [vasp.incar] are inherited, except those in incar_unset.

    A tag given in both [vasp.incar] and the input set has the value of the
    input set.  [vasp.incar] and incar_unset are removed.

    """
    config = VelphConfig(
        vasp=VaspSection(
            incar={"encut": 500, "npar": 4, "ismear": 0},
            input_sets={
                "nac": InputSetSection(
                    incar={"lepsilon": True, "ismear": -5}, incar_unset=("npar",)
                ),
                "relax": InputSetSection(cell="unitcell"),
            },
        )
    )
    resolved = resolve_incar(config)
    assert resolved.vasp is not None and resolved.vasp.input_sets is not None
    assert resolved.vasp.incar is None
    assert resolved.vasp.input_sets["nac"] == InputSetSection(
        incar={"encut": 500, "ismear": -5, "lepsilon": True}
    )
    assert resolved.vasp.input_sets["relax"] == InputSetSection(
        cell="unitcell", incar={"encut": 500, "npar": 4, "ismear": 0}
    )


def test_resolve_incar_without_base():
    """Without [vasp.incar], an input set keeps its tags or has no incar."""
    config = VelphConfig(
        vasp=VaspSection(
            input_sets={
                "nac": InputSetSection(incar={"lepsilon": True}),
                "relax": InputSetSection(incar_unset=("npar",)),
            }
        )
    )
    resolved = resolve_incar(config)
    assert resolved.vasp is not None and resolved.vasp.input_sets is not None
    assert resolved.vasp.input_sets["nac"] == InputSetSection(incar={"lepsilon": True})
    assert resolved.vasp.input_sets["relax"] == InputSetSection()
    assert resolve_incar(VelphConfig()) == VelphConfig()
