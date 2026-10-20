"""Errors on velph.toml files and templates of velph earlier than 0.15.

check_old_forms raises VelphConfigError on a velph.toml without [velph]
version, which velph earlier than 0.15 wrote, and on a velph init template that
has keys of velph earlier than 0.15, such as [init.options].  The message lists
only the changes that concern the keys found in the file.  The loader of
config/io.py calls it before it reads a file.

A velph.toml of velph 0.14 is read by config/legacy.py, which converts it
before the loader reads it.

"""

from __future__ import annotations

import textwrap
from collections.abc import Iterator, Mapping
from typing import Any

from phelel.velph.config.schema import VelphFileKind
from phelel.velph.errors import VelphConfigError

_DIGEST_EL_BANDS = (
    "[vasp.el_bands.bands] and [vasp.el_bands.dos] are now [vasp.el_bands] and "
    '[vasp.el_dos], with the directories "el_bands" and "el_dos".'
)
_DIGEST_FFT_MESH = (
    '[phelel] fft_mesh is removed.  "velph phelel differentiate" computes the '
    "FFT mesh from encut and prec in the INCARs of the electron-phonon "
    "calculations, or from --encut."
)
_DIGEST_KSPACING = (
    '"velph init" converts kspacing in a k-point block to mesh, and velph.toml '
    "has only mesh."
)
_DIGEST_EMPTY_TABLE = (
    "An INCAR tag set to {} is no longer used.  A section such as [vasp.nac] "
    'does not inherit the tags of [vasp.incar] listed in its "incar_unset".'
)
_DIGEST_INIT_OPTIONS = (
    "[init.options] is now [init].  Its keys are those of the velph init "
    "command line.  supercell_dimension, amplitude and the other supercell and "
    "displacement keys in [init] apply to every code in codes."
)
SCHEDULER_NAME_REMOVED = (
    '"scheduler_name" is removed.  Write the job script in "scheduler_template".'
)
DIGEST_TAIL = 'See "Changes in velph 0.15" in the documentation for all changes.'


def check_old_forms(
    data: Mapping[str, Any], file_kind: VelphFileKind, name: str
) -> None:
    """Raise VelphConfigError when data is a file of velph earlier than 0.15.

    A velph.toml without [velph] version is such a file.  A template is such a
    file when it has keys of velph earlier than 0.15, such as [init.options].
    The message lists the changes that concern the keys found in data, and
    name is the name of the file in the message.

    """
    if file_kind == "velph_toml":
        velph = data.get("velph")
        if isinstance(velph, Mapping) and "version" in velph:
            return
        lines = list(_old_form_lines(data, file_kind))
        head = [
            f'"{name}" was written by velph earlier than 0.15 and cannot be read.',
            'Re-run "velph init" to make a new velph.toml.',
        ]
    else:
        lines = list(_old_form_lines(data, file_kind))
        if not lines:
            return
        head = [f'"{name}" is a velph init template of velph earlier than 0.15.']
    message = head
    if lines:
        message += ["", "Changes that concern this file:", ""]
        for line in lines:
            message += textwrap.wrap(
                line, width=76, initial_indent="  - ", subsequent_indent="    "
            )
    message += ["", DIGEST_TAIL]
    raise VelphConfigError("\n".join(message))


def _old_form_lines(data: Mapping[str, Any], file_kind: VelphFileKind) -> Iterator[str]:
    vasp = data.get("vasp")
    if not isinstance(vasp, Mapping):
        vasp = {}
    el_bands = vasp.get("el_bands")
    if isinstance(el_bands, Mapping) and ({"bands", "dos"} & set(el_bands)):
        yield _DIGEST_EL_BANDS
    phelel = data.get("phelel")
    if isinstance(phelel, Mapping) and "fft_mesh" in phelel:
        yield _DIGEST_FFT_MESH
    input_set_tables = list(_old_input_set_tables(vasp))
    scheduler_tables = [data.get("scheduler")] + [
        input_set_table.get("scheduler") for input_set_table in input_set_tables
    ]
    if any(
        isinstance(scheduler, Mapping) and "scheduler_name" in scheduler
        for scheduler in scheduler_tables
    ):
        yield SCHEDULER_NAME_REMOVED
    if file_kind == "velph_toml":
        if any(
            isinstance(input_set_table.get(block), Mapping)
            and "kspacing" in input_set_table[block]
            for input_set_table in input_set_tables
            for block in ("kpoints", "kpoints_dense")
        ):
            yield _DIGEST_KSPACING
    else:
        incar_tables = [vasp.get("incar")] + [
            input_set_table.get("incar") for input_set_table in input_set_tables
        ]
        if any(
            isinstance(incar, Mapping) and any(value == {} for value in incar.values())
            for incar in incar_tables
        ):
            yield _DIGEST_EMPTY_TABLE
        init = data.get("init")
        if isinstance(init, Mapping) and "options" in init:
            yield _DIGEST_INIT_OPTIONS


def _old_input_set_tables(vasp: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield the tables of the input sets in [vasp].

    The tables in [vasp.el_bands] of a velph.toml of velph earlier than 0.15
    are included.

    """
    for key, table in vasp.items():
        if key == "incar" or not isinstance(table, Mapping):
            continue
        yield table
        if key == "el_bands":
            for sub_table in table.values():
                if isinstance(sub_table, Mapping):
                    yield sub_table
