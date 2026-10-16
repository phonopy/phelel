"""Settings of velph.toml made by velph init.

build_velph_config makes them as follows.

   defaults                       template                       command line
   +--------------+--------+      +--------------+--------+      +-------------+
   | [phelel],    | [init] |      | [phelel],    | [init] |      | InitSection |
   | ...,         |        |      | ...,         |        |      |             |
   | [vasp.*],    |        |      | [vasp.*],    |        |      |             |
   |              |        |      | [scheduler]  |        |      |             |
   +------+-------+---+----+      +------+-------+---+----+      +------+------+
          |           |                  |           |                  |
          |      _init_layer             |      _init_layer        _init_layer
          |           |                  |           |                  |
          v           v                  v           v                  v
         (1)         (2)                (3)         (4)                (5)
          |           |                  |           |                  |
          +-----------+------------------+-----------+------------------+
                                         |
                                         v
             merge: (1) < (2) < (3) < (4) < (5), an upper layer wins
                                         |
                                         v
                     merged settings (VelphConfig with [init])
                                         |
                                         |  resolve
                                         v
           +-------+------------+-------------+----------------+-------+
           | cells | supercells | input sets  | k-point meshes | INCAR |
           |       |            | written     |                |       |
           +-------+------------+-------------+----------------+-------+
                                         |
                                         v
                        settings of velph.toml (no [init])

[init] and the command line take the same keys.  _init_layer translates
them into settings of the sections, such as --dim into the supercell of each
code and --cell-for-relax into "cell" of [vasp.relax], so that the layers
are merged by one merge.  codes is taken from the [init] layers before the
merge, because _init_layer gives the settings to the sections of the codes.

"""

from __future__ import annotations

import dataclasses
from typing import Any, TypeVar

import click
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms, get_cell_dict
from phonopy.structure.cells import get_supercell

from phelel import __version__
from phelel.velph.cli.init.cells import InitCells, make_cells, resolve_supercell
from phelel.velph.config.defaults import get_default_config
from phelel.velph.config.input_sets import INPUT_SETS, InputSetName
from phelel.velph.config.merge import merge
from phelel.velph.config.resolve import resolve_incar
from phelel.velph.config.schema import (
    CODES,
    CodeName,
    CodeSection,
    InitSection,
    InputSetSection,
    KpointsMesh,
    KpointsSpacing,
    SupercellDimension,
    SupercellMatrix,
    VaspSection,
    VelphConfig,
)
from phelel.velph.utils.mesh import kspacing_to_mesh

T = TypeVar("T")

# k-point blocks whose mesh velph init computes, and the [init] key of the
# spacing used when the block is not given.
_KSPACING_KEYS = {"kpoints": "kspacing", "kpoints_dense": "kspacing_dense"}


def build_velph_config(
    input_cell: PhonopyAtoms,
    template: VelphConfig | None = None,
    options: InitSection | None = None,
) -> VelphConfig:
    """Return the settings of velph.toml made by velph init.

    template is the template read in template mode, and options are the
    command-line options.  The returned VelphConfig has no [init] and is
    written by write_velph_toml.

    [scheduler] is that of the template, and a message is shown when the
    template has none.  click.ClickException is raised when a code given
    explicitly in codes, by the template or the command line, has no
    supercell, and by make_cells, resolve_supercell and the k-point meshes.

    """
    defaults = get_default_config()
    template_init = template.init if template is not None else None
    init = merge(merge(defaults.init, template_init), options)
    assert init is not None and init.codes is not None
    codes = init.codes
    explicit_codes = any(
        layer is not None and layer.codes is not None
        for layer in (template_init, options)
    )

    sections = None
    if template is not None:
        sections = dataclasses.replace(template, init=None)
        for code in CODES:
            if code not in codes and getattr(sections, code) is not None:
                click.echo(
                    f'[{code}] of the template is ignored: "{code}" is not in codes.'
                )
                sections = dataclasses.replace(sections, **{code: None})

    config: VelphConfig | None = defaults
    for layer in (
        _init_layer(defaults.init, codes),
        sections,
        _init_layer(template_init, codes),
        _init_layer(options, codes),
    ):
        config = merge(config, layer)
    assert config is not None and config.init is not None
    init = config.init

    cells = make_cells(
        input_cell,
        tolerance=_required(init.tolerance),
        symmetrize_cell=_required(init.symmetrize_cell),
        find_primitive=_required(init.find_primitive),
        primitive_cell_choice=_required(init.primitive_cell_choice),
        magmom=init.magmom,
        site_mixture=init.site_mixture,
    )
    code_sections = {
        code: _resolve_code_section(
            code, getattr(config, code), codes, explicit_codes, cells, init
        )
        for code in CODES
    }
    config = dataclasses.replace(config, **code_sections)  # type: ignore[arg-type]
    if config.scheduler is None:
        click.echo(
            "[scheduler] is not written: the template has no [scheduler], so no job "
            'script "_job.sh" is written.  See "velph.toml-[scheduler] section" in '
            "the documentation to write job scripts."
        )
    config = dataclasses.replace(
        config,
        version=__version__,
        init=None,
        vasp=_resolve_vasp(config, cells, init),
        symmetry=cells.symmetry,
        unitcell=get_cell_dict(cells.unitcell),
        primitive_cell=get_cell_dict(cells.primitive),
    )
    return resolve_incar(config)


def _init_layer(
    init: InitSection | None, codes: tuple[CodeName, ...]
) -> VelphConfig | None:
    """Return the settings that [init] or the command line gives.

    The supercell and the displacement settings are given to the section of
    each code of codes, split_site_mixture is given as site_mixture_scheme to
    [phelel] and [phonopy], and cell_for_relax and cell_for_nac are given to
    [vasp.relax] and [vasp.nac].  init is kept as [init].

    """
    if init is None:
        return None
    site_mixture_scheme = None
    if init.split_site_mixture is not None:
        site_mixture_scheme = "split" if init.split_site_mixture else "merge"
    code_sections = {}
    for code in codes:
        section = CodeSection(
            supercell=init.supercell,
            amplitude=init.amplitude,
            diagonal=init.diagonal,
            plusminus=init.plusminus,
            nosym=init.nosym,
            # phono3py does not support site mixture.
            site_mixture_scheme=(
                site_mixture_scheme if code != "phono3py" else None  # type: ignore[arg-type]
            ),
        )
        if section != CodeSection():
            code_sections[code] = section
    input_sets = {
        set_name: InputSetSection(cell=cell)
        for set_name, cell in (
            ("relax", init.cell_for_relax),
            ("nac", init.cell_for_nac),
        )
        if cell is not None
    }
    vasp = VaspSection(input_sets=input_sets) if input_sets else None
    return VelphConfig(init=init, vasp=vasp, **code_sections)  # type: ignore[arg-type]


def _resolve_code_section(
    code: CodeName,
    section: CodeSection | None,
    codes: tuple[CodeName, ...],
    explicit_codes: bool,
    cells: InitCells,
    init: InitSection,
) -> CodeSection | None:
    """Return the section of code as written in velph.toml, or None.

    A code not in codes is not written.  A code without a supercell is not
    written, with a message, unless codes is given explicitly, when it is an
    error.  site_mixture_scheme is written only for a unit cell with weighted
    species.

    """
    if code not in codes:
        return None
    if section is None or section.supercell is None:
        if explicit_codes:
            raise click.ClickException(
                f'"{code}" is in codes, but no supercell is given for it.  Give '
                "--dim, --supercell-matrix or --max-num-atoms, or "
                f"supercell_dimension, supercell_matrix or max_num_atoms in [{code}] "
                "of the template."
            )
        click.echo(f"[{code}] is not written: no supercell is given for it.")
        return None
    supercell = resolve_supercell(
        code,
        section.supercell,
        cells,
        symmetrize_cell=_required(init.symmetrize_cell),
        tolerance=_required(init.tolerance),
    )
    site_mixture_scheme = section.site_mixture_scheme
    if not cells.unitcell.has_weighted_species:
        site_mixture_scheme = None
    return dataclasses.replace(
        section, supercell=supercell, site_mixture_scheme=site_mixture_scheme
    )


def _resolve_vasp(
    config: VelphConfig, cells: InitCells, init: InitSection
) -> VaspSection | None:
    """Return [vasp] with the input sets written and their k-point meshes.

    An input set is written when the code that it needs is written (INPUT_SETS).
    The input sets are in the order of INPUT_SETS.

    """
    if config.vasp is None:
        return None
    given = config.vasp.input_sets or {}
    input_sets: dict[str, InputSetSection] = {}
    for set_name, entry in INPUT_SETS.items():
        if set_name not in given:
            continue
        if entry.required_code is not None:
            if getattr(config, entry.required_code) is None:
                continue
        input_sets[set_name] = _resolve_kpoints(
            set_name, given[set_name], config, cells, init
        )
    return dataclasses.replace(config.vasp, input_sets=input_sets or None)


def _resolve_kpoints(
    set_name: InputSetName,
    input_set: InputSetSection,
    config: VelphConfig,
    cells: InitCells,
    init: InitSection,
) -> InputSetSection:
    """Return input_set with the meshes of its k-point blocks.

    The mesh of a block is the mesh given in the block, else the mesh of the
    kspacing given in the block, else the mesh of kspacing (kpoints) or
    kspacing_dense (kpoints_dense) of [init].  It is computed on the cell of the
    input set (INPUT_SETS), with use_grg of [init] except on the supercells.

    """
    entry = INPUT_SETS[set_name]
    cell, cell_name = _cell_of_input_set(set_name, input_set, config, cells, init)
    use_grg = entry.cell != "supercells" and _required(init.use_grg)
    meshes: dict[str, Any] = {}
    for block in entry.kpoints_files.values():
        if block not in _KSPACING_KEYS:
            continue
        spec = getattr(input_set, block)
        if isinstance(spec, KpointsMesh):
            continue
        if isinstance(spec, KpointsSpacing):
            kspacing, shift = spec.kspacing, spec.shift
        else:
            kspacing, shift = _required(getattr(init, _KSPACING_KEYS[block])), None
        where = f"[vasp.{set_name}.{block}]"
        try:
            mesh = kspacing_to_mesh(
                kspacing, cell, use_grg=use_grg, tolerance=_required(init.tolerance)
            )
        except RuntimeError as e:
            raise click.ClickException(
                f"No k-point mesh of {where} with kspacing {kspacing} respects the "
                f"symmetry of the {cell_name} ({e})"
            ) from e
        if not use_grg and isinstance(mesh[0], list):
            click.echo(
                f"The k-point mesh of {where} is a generalized regular grid, "
                f"because a regular mesh breaks the symmetry of the {cell_name}."
            )
        meshes[block] = KpointsMesh(mesh=_tuple_mesh(mesh), shift=shift)
    return dataclasses.replace(input_set, **meshes)


def _cell_of_input_set(
    set_name: InputSetName,
    input_set: InputSetSection,
    config: VelphConfig,
    cells: InitCells,
    init: InitSection,
) -> tuple[PhonopyAtoms, str]:
    """Return the cell on which the k-point meshes are computed, and its name."""
    entry = INPUT_SETS[set_name]
    if entry.cell == "supercells":
        section = getattr(config, set_name)
        assert isinstance(section, CodeSection)
        assert isinstance(section.supercell, (SupercellDimension, SupercellMatrix))
        if isinstance(section.supercell, SupercellDimension):
            smat = np.diag(section.supercell.supercell_dimension)
        else:
            smat = np.array(section.supercell.supercell_matrix)
        supercell = get_supercell(
            cells.unitcell, smat, symprec=_required(init.tolerance)
        )
        return supercell, f"supercell of [{set_name}]"
    if entry.cell == "cell" and input_set.cell == "unitcell":
        return cells.unitcell, "unit cell"
    return cells.primitive, "primitive cell"


def _tuple_mesh(mesh: list[int] | list[list[int]]) -> tuple:
    if isinstance(mesh[0], list):
        return tuple(tuple(row) for row in mesh)  # type: ignore[union-attr]
    return tuple(mesh)


def _required(value: T | None) -> T:
    """Return a value of [init] that the defaults give."""
    assert value is not None
    return value
