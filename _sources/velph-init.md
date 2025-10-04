(velph_init)=
# `velph init`

`velph init` with command options will generate modified `velph.toml` from the
template.
```bash
% velph init [OPTIONS] CELL_FILENAME PROJECT_FOLDER
```

`velph init --help` shows short documents of the options. Two arguments have to
be specified, POSCAR-format crystal structure and directory name that is created
where `velph.toml` is stored. The available options are explained below. Some
options can be specified in `[init.options]` section of the velph-toml template
file (see {ref}`velph_init_template`).

## `velph init` options

### `--template-toml`

Using this option, `velph.toml` like file is read as the template instead of the
template hard coded in velph code.

### `--tolerance`

This is used for the symmetry check tolerance in Angstrom. Symmetry is searched
always even without `--symmetrize-cell`.

### `--symmetrize-cell`

By default (no-symmetrize), the input POSCAR-type structure is simply used as
the unit cell. With `--symmetrize-cell`, input POSCAR-type structure is
symmetrized and standardized conventional unit cell and primitive cell
("unitcell" and "primitive_cell", respectively) are written in `velph.toml`.

### `--no-find-primitive`

By this option, the input POSCAR-type structure is used as the primitive cell
even if it is not a primitive cell.

### `--kspacing` and `--kspacing-dense`

Sampling k-point meshes are calculated from these values in the similar way to
VASP `KSPACING` definition by overwriting the template.

### `--max-num-atoms`

Supercell shape is determined so that its number of atoms is equal or less than
this number respecting the crystallographic point group. Use of this option
requires `--symmetrize-cell`.

### `--dim`

Supercell shape is determined by three integer values that extend along a, b,
and c axes of the unit cell, respectively. See also `--supercell-matrix`.

### `--supercell-matrix`

Supercell shape is determined by nine integer values (v1, ..., v9) that
corresponds to a 3x3 matrix [[v1, v2, v3], [v4, v5, v6], [v7, v8, v9]].

### `--cell-for-relax`

This chooses unit cell or primitive cell for structure optimization (`relax`).
The default is `unitcell`. Specify `primitive` to use primitive cell.

### `--cell-for-nac`

This chooses unit cell or primitive cell for NAC calculation (`nac`). The
default is `primitive`. Specify `primitive` to use primitive cell.

### `--primitive-cell-choice`

Primitive cell choice, "standardized" (default) or "reduced".

### `--use-grg`

Use generalized regular grid.

### `--amplitude`

Distance of displacements in Angstrom.

### `--magmom`

String corresponding to INCAR MAGMOM tag value for unit cell, e.g., "24*1" or "0
0 1". This is similar to `MAGMOM` tag in phonopy, see
https://phonopy.github.io/phonopy/setting-tags.html#magmom. In velph, the
asterisk symbol (`*`) is supported.
