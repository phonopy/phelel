(velph_toml_cell)=
# `velph.toml`-crystal structure sections

The `[unitcell]`, `[primitive_cell]`, and `[symmetry]` sections describe the
crystal structure and its symmetry. They are written by `velph init` from the
input cell file.

```{note}
These sections are generated automatically and are normally **not** edited by
hand. They are the reference structure for the whole project, and the
reproducibility of a calculation depends on them staying unchanged. This page
explains how to read them; edit them only if you understand the consequences.
```

(velph_toml_cell_cells)=
## `[unitcell]` and `[primitive_cell]`

Both sections use the same layout: a `lattice` table followed by one
`[[...points]]` entry per atom.

```toml
[unitcell]
lattice = [
  [     6.000000000000000,     0.000000000000000,     0.000000000000000 ], # a
  [     0.000000000000000,     6.000000000000000,     0.000000000000000 ], # b
  [     0.000000000000000,     0.000000000000000,     6.000000000000000 ], # c
]
[[unitcell.points]]  # 1
symbol = "Na"
coordinates = [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
mass = 22.989769
[[unitcell.points]]  # 2
symbol = "Cl"
coordinates = [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
mass = 35.453000
```

- `lattice` : the three lattice vectors **a**, **b**, **c** as rows, in Angstrom.
- `[[unitcell.points]]` / `[[primitive_cell.points]]` : a TOML array of tables,
  one per atom, in the order the atoms are stored. The trailing `# 1`, `# 2`, ...
  comments are the 1-based atom indices, written for readability only.

Each point entry accepts the following keys:

| Key               | Type                        | Meaning                                                              |
| ----------------- | --------------------------- | ------------------------------------------------------------------- |
| `symbol`          | str                         | Chemical element symbol.                                            |
| `coordinates`     | list[float] (3)             | Fractional coordinates in the cell.                                 |
| `mass`            | float                       | Atomic mass in AMU. See the note below.                             |
| `magnetic_moment` | float or list[float] (optional) | Collinear (scalar) or non-collinear (3-vector) magnetic moment. |
<!-- Hidden until site mixture is public:
| `weight`          | float (optional)            | Site occupation weight for site mixture (partial occupancy).        |
-->

```{note}
`mass` is written explicitly so that `velph.toml` is self-describing: the masses
do not depend on the default atomic-mass table and can be edited (e.g. for
isotopes). These values may differ from the masses used internally by VASP
(`POMASS`). `magnetic_moment` is written only when the structure carries magnetic
moments (e.g. set through the `--magmom` option of `velph init`).
<!-- Hidden until site mixture is public:
`weight` is written only for site-mixture structures.
-->
```

The `[primitive_cell]` section is written with the same layout as `[unitcell]`.
When the primitive cell differs from the unit cell, the transformation between
them is recorded as `primitive_matrix` in the `[symmetry]` section.

(velph_toml_cell_symmetry)=
## `[symmetry]`

This section records the symmetry of the unit cell as found by spglib.

```toml
[symmetry]
spacegroup_type = "Fm-3m"
tolerance = 1e-05
primitive_matrix = [
  [  0.000000000000000,  0.500000000000000,  0.500000000000000 ],
  [  0.500000000000000,  0.000000000000000,  0.500000000000000 ],
  [  0.500000000000000,  0.500000000000000,  0.000000000000000 ],
]
```

| Key               | Type                  | Meaning                                                                        |
| ----------------- | --------------------- | ------------------------------------------------------------------------------ |
| `spacegroup_type` | str                   | International space-group symbol (non-magnetic structures).                     |
| `uni_number`      | int                   | Magnetic space-group "uni number" (written instead of `spacegroup_type` for magnetic structures). |
| `tolerance`       | float                 | Symmetry-search tolerance in Angstrom, set by the `--tolerance` option of `velph init`. |
| `primitive_matrix`| list[list[float]] (3x3) | Transformation from the unit cell to the primitive cell. Written only when the primitive cell is smaller than the unit cell. |

For a non-magnetic structure, `spacegroup_type` holds the international symbol
(e.g. `"P1"`, `"Fm-3m"`). For a magnetic structure, `uni_number` is written
instead. The `primitive_matrix` appears only when the unit cell and primitive
cell have different numbers of atoms.
