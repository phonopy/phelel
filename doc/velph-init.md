(velph_init)=
# `velph init`

`velph init` reads a crystal structure and writes `velph.toml` in a new project
folder.

```bash
% velph init [OPTIONS] CELL_FILENAME PROJECT_FOLDER
```

`velph init --help` shows short documents of the options. Two arguments have to
be specified, a POSCAR-format crystal structure and the name of the folder where
`velph.toml` is written. The available options are explained below. The options
except `--force`, `--template-toml` and `--toml-filename` can also be given in
the `[init]` section of a template (see {ref}`velph_init_template`).

The settings of `velph.toml` come from the default settings of `velph init`, from
the template given by `--template-toml`, and from the command-line options. A
command-line option takes priority over the template, and the template takes
priority over the default settings (see {ref}`velph_init_template_precedence`).

## Project folder

`velph init` creates `PROJECT_FOLDER` and writes `velph.toml` in it.

- When `PROJECT_FOLDER` already exists, `velph init` stops with an error. With
  `--force`, it writes `velph.toml` in the existing folder.
- With `.` as `PROJECT_FOLDER`, `velph init` writes `velph.toml` in the current
  folder.
- When `velph.toml` already exists in the folder, `velph init` stops with an
  error. With `--force`, it overwrites the file.
- When the current folder has `POTCAR`, `velph init` shows the elements and the
  largest `ENMAX` of `POTCAR`, and copies it to `PROJECT_FOLDER` unless the
  folder already has one.

## `velph init` options

### `--template-toml`

Using this option, a template in TOML is read, and its settings are used instead
of the default settings of `velph init`. See {ref}`velph_init_template`.

### `--code`

This chooses the codes whose sections, `[phelel]`, `[phonopy]` and `[phono3py]`,
are written in `velph.toml`. Repeat it for more than one code, for example
`--code phelel --code phonopy`. By default all three codes are chosen. The input
sets of `[vasp]` that a code uses are written only for the chosen codes (see
{ref}`velph_toml_vasp`).

A code needs a supercell, which is given by `--dim`, `--supercell-matrix` or
`--max-num-atoms`, or by the template.

- When the codes are not chosen, a code without a supercell is not written. For
  example, `velph init` without any option writes `velph.toml` with the input
  sets `relax`, `nac`, `el_bands` and `el_dos` only.
- A code chosen with `--code`, or with `codes` in `[init]` of the template, has
  to have a supercell. Otherwise `velph init` stops with an error that says which
  option or key gives the supercell.

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

The k-point meshes are calculated from these values, in 1/Angstrom. `--kspacing`
is used for the `kpoints` blocks and `--kspacing-dense` for the `kpoints_dense`
blocks. The number of divisions along each reciprocal basis vector b_i is
max(1, ceiling(|b_i| 2π / `kspacing`)), the same rule as the VASP INCAR tag
`KSPACING`. The mesh is computed on the cell of each input set: the supercell for
`phelel`, `phonopy` and `phono3py`, and the unit cell or the primitive cell for
the others.

A k-point block of the template that gives `mesh` or `kspacing` takes priority
over these options. For example, with `kspacing = 0.3` in
`[vasp.relax.kpoints]` of the template and `--kspacing 0.2`, the mesh of `relax`
is computed from 0.3 and those of the other input sets from 0.2. The default
settings give `mesh = [1, 1, 1]` in `[vasp.ph_bands.kpoints]`, so these options
do not change the mesh of `ph_bands`.

### `--max-num-atoms`, `--dim`, and `--supercell-matrix`

These three options give the supercell in different ways.

`--max-num-atoms` determines the supercell shape so that its number of atoms is
equal or less than this number respecting the crystallographic point group. Use
of this option requires `--symmetrize-cell`.

`--dim` determines the supercell shape by three integer values that extend
along a, b, and c axes of the unit cell, respectively.

`--supercell-matrix` determines the supercell shape by nine integer values
(v1, ..., v9) that correspond to the 3x3 matrix
[[v1, v2, v3], [v4, v5, v6], [v7, v8, v9]].

Give only one of `--max-num-atoms`, `--dim`, and `--supercell-matrix`. When two
or more of them are given, `velph init` stops with an error. The supercell given
by one of these options is used for all the chosen codes, instead of the
supercell settings in the template (see {ref}`velph_init_template_precedence`).

### `--cell-for-relax`

This chooses unit cell or primitive cell for structure optimization (`relax`).
The default is `unitcell`. Specify `primitive` to use primitive cell.

### `--cell-for-nac`

This chooses unit cell or primitive cell for NAC calculation (`nac`). The
default is `primitive`. Specify `unitcell` to use unit cell.

### `--primitive-cell-choice`

Primitive cell choice, "standardized" (default) or "reduced".

### `--use-grg`

Use generalized regular grid for the k-point meshes computed from `--kspacing`
and `--kspacing-dense`, except those of the supercells. Even without this
option, a generalized regular grid is used when a (conventional) regular mesh
breaks the symmetry of the cell, and `velph init` says so.

### `--amplitude`

Distance of displacements in Angstrom. The value is used for `[phelel]`,
`[phonopy]`, and `[phono3py]`. Without this option, `velph init` writes 0.03 in
`[phelel]` and `[phono3py]` and 0.01 in `[phonopy]`, which are the default
displacement distances of phono3py and phonopy for VASP.

### `--diagonal`

Whether to also generate displacements along diagonal (non-axis) directions. By
default, only axis-aligned displacements are used. This corresponds to the
`diagonal` key of `[phelel]`, `[phonopy]`, and `[phono3py]`.

### `--plusminus` / `--auto`

`--plusminus` (default) adds the opposite (minus) displacement for every
displacement. `--auto` adds the minus displacement only when it is not
symmetrically equivalent to the plus one. This corresponds to the `plusminus`
key of `[phelel]`, `[phonopy]`, and `[phono3py]`.

### `--nosym`

Disable symmetry reduction of the displacements (equivalent to running
`phonopy --nosym`). This corresponds to the `nosym` key of `[phelel]`,
`[phonopy]`, and `[phono3py]`.

### `--magmom`

String corresponding to INCAR MAGMOM tag value for unit cell, e.g., "24*1" or "0
0 1". This is similar to `MAGMOM` tag in phonopy, see
https://phonopy.github.io/phonopy/setting-tags.html#magmom. In velph, the
asterisk symbol (`*`) is supported.

### `--force`

Write `velph.toml` in an existing project folder, and overwrite `velph.toml`
when it exists.

### `--toml-filename`

File name of the `velph.toml`-type file written in the project folder. The
default is `velph.toml`.

```{note}
`--force`, `--template-toml`, and `--toml-filename` are command-line only and
have no corresponding `[init]` key in the template file (see
{ref}`velph_init_template`).
```
