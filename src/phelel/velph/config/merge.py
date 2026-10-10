"""Merge of two layers of velph settings."""

from __future__ import annotations

import dataclasses
from typing import Any, TypeVar

from phelel.velph.config.schema import KEYS, MERGE_KEY, RECURSIVE, SECTIONS

T = TypeVar("T")


def merge(lower: T | None, upper: T | None) -> T | None:
    """Return the settings of two layers merged, with priority to upper.

    lower and upper are dataclasses of the schema of the same type, such as
    two VelphConfig, for example the defaults and a template.  A field that
    upper does not give (None) takes the value of lower.  A field that both
    give is merged by the rule in its metadata in the schema:

    - a section such as [phelel] is merged field by field;
    - an INCAR or scheduler table is merged key by key, and a key of upper
      replaces the same key of lower;
    - the input sets of [vasp], such as [vasp.relax], are merged one by one;
    - any other field, such as a number, a supercell or a k-point block, is
      replaced by the value of upper.

    """
    if upper is None:
        return lower
    if lower is None:
        return upper
    return _merge_section(lower, upper)


def _merge_section(lower: Any, upper: Any) -> Any:
    if type(lower) is not type(upper):
        raise TypeError(
            f"Cannot merge {type(lower).__name__} with {type(upper).__name__}."
        )
    values = {}
    for field in dataclasses.fields(upper):
        value_lower = getattr(lower, field.name)
        value_upper = getattr(upper, field.name)
        rule = field.metadata.get(MERGE_KEY)
        if value_upper is None:
            values[field.name] = value_lower
        elif value_lower is None:
            values[field.name] = value_upper
        elif rule == RECURSIVE:
            values[field.name] = _merge_section(value_lower, value_upper)
        elif rule == KEYS:
            values[field.name] = {**value_lower, **value_upper}
        elif rule == SECTIONS:
            merged = dict(value_lower)
            for key, section in value_upper.items():
                merged[key] = merge(value_lower.get(key), section)
            values[field.name] = merged
        else:
            values[field.name] = value_upper
    return type(upper)(**values)
