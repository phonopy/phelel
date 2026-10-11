"""Resolution of the settings that velph init makes before it writes velph.toml.

A velph init template can give INCAR tags shared by the input sets in
[vasp.incar], and remove some of them from an input set with incar_unset.
velph.toml has neither: each input set holds all of its INCAR tags.
resolve_incar makes these INCAR tags.

"""

from __future__ import annotations

import dataclasses

from phelel.velph.config.schema import VelphConfig


def resolve_incar(config: VelphConfig) -> VelphConfig:
    """Return config in which each input set holds all of its INCAR tags.

    The INCAR tags of an input set are the tags of [vasp.incar], except those
    listed in incar_unset of the input set, together with the tags of the input
    set.  A tag given in both [vasp.incar] and the input set has the value of
    the input set.  [vasp.incar] and incar_unset are removed.

    """
    if config.vasp is None:
        return config
    base = config.vasp.incar or {}
    input_sets = {}
    for set_name, input_set in (config.vasp.input_sets or {}).items():
        unset = set(input_set.incar_unset or ())
        incar = {tag: value for tag, value in base.items() if tag not in unset}
        incar.update(input_set.incar or {})
        input_sets[set_name] = dataclasses.replace(
            input_set, incar=incar or None, incar_unset=None
        )
    vasp = dataclasses.replace(config.vasp, incar=None, input_sets=input_sets or None)
    return dataclasses.replace(config, vasp=vasp)
