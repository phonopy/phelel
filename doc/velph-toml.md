(velph_toml)=
# `velph.toml` file

The `velph.toml` file contains the settings required to perform electron-phonon
interaction calculations using the phelel and VASP codes. Ideally, once created,
it should remain unchanged throughout the duration of a calculation project for
the reproducibility.

`velph init` writes `velph.toml` from its default settings, from a template given
by `--template-toml`, and from its command-line options (see {ref}`velph_init`
and {ref}`velph_init_template`). A `velph.toml` of another project can also be
copied and edited.

The `velph` subcommands read `velph.toml` to generate VASP input files and to run
the other tools. This page explains its contents.

```{note}
A `velph.toml` written by velph 0.14 has no `[velph]` section. The `velph`
subcommands still read such a file.
```

## Section overview

A `velph.toml` file is organized into the following top-level sections.

```
velph.toml
├── [velph]           # version of phelel that wrote the file
├── [phelel]          # supercell and displacements for phelel
├── [phonopy]         # supercell and displacements for phonopy
├── [phono3py]        # supercell and displacements for phono3py
├── [vasp]            # VASP input settings, one input set per kind of calculation
│   ├── [vasp.relax]
│   ├── [vasp.nac]
│   └── ...           # (see the [vasp] page)
├── [scheduler]       # job-script template and its parameters
├── [symmetry]        # space group and primitive-cell transformation
├── [unitcell]        # crystal structure (unit cell)
└── [primitive_cell]  # crystal structure (primitive cell)
```

`[velph]`, `[unitcell]` and `[primitive_cell]` are required. `velph init` writes
`[phelel]`, `[phonopy]` and `[phono3py]` for the codes chosen with its `--code`
option, all three by default. Of the input sets of `[vasp]`, it writes those that
every project uses and those that the chosen codes use (see
{ref}`velph_toml_vasp`).

The detailed contents of the larger sections are described on separate pages:

- {ref}`[vasp] <velph_toml_vasp>` : VASP input settings of each input set.
- {ref}`[scheduler] <velph_toml_scheduler>` : job-script generation.
- {ref}`[unitcell], [primitive_cell], [symmetry] <velph_toml_cell>` : crystal
  structure and symmetry.

(velph_toml_template_only)=
## Settings that only a template has

A template of `velph init` is written in the same format as `velph.toml`, and it
can have the following settings in addition. `velph init` resolves them into the
values that it writes in `velph.toml`, so they cannot be given in `velph.toml`.

- `[init]` : defaults of the `velph init` command-line options (see
  {ref}`velph_init_template_init_options`).
- `max_num_atoms` in `[phelel]`, `[phonopy]` and `[phono3py]` : `velph init`
  writes the supercell that it chooses from this number as
  `supercell_dimension` or `supercell_matrix`.
- `kspacing` in a k-point block : `velph init` writes the mesh computed from it
  as `mesh`.
- `[vasp.incar]` and `incar_unset` : `velph init` writes all the INCAR tags of
  each input set in `[vasp.<input set>.incar]` (see
  {ref}`velph_toml_vasp_incar`).

If `velph.toml` has one of them, the `velph` subcommands stop with an error.

(velph_toml_edit)=
## Sections you edit vs. sections that are generated

- `[phelel]`, `[phonopy]`, `[phono3py]`, the input sets of `[vasp]`, and
  `[scheduler]` are edited by the user. These hold the calculation settings and
  are the parts you tune for a project, most conveniently through the
  {ref}`template <velph_init_template>`.
- `[velph]`, `[unitcell]`, `[primitive_cell]`, and `[symmetry]` are generated
  automatically. These record the version of phelel, the reference crystal
  structure and its symmetry, and are normally left untouched, because
  reproducibility depends on them. See {ref}`velph_toml_cell`.

(velph_toml_override)=
## Settings of an input set and the shared settings

- Each input set holds all of its INCAR tags in `[vasp.<input set>.incar]`, and
  the generate command of the input set writes them to `INCAR` as they are. To
  change a tag, edit it in each input set that uses it. See
  {ref}`velph_toml_vasp_incar`.
- The settings of the top-level `[scheduler]` are overridden by those of
  `[vasp.<input set>.scheduler]` for that input set only. See
  {ref}`velph_toml_scheduler`.

## `[velph]`

```toml
[velph]
version = "0.16.0"
```

`version` is the version of phelel that wrote the file. A `velph.toml` without
it is read as a file written by velph 0.14 (see
{ref}`velph_changes_0_15_project_folder`).

## `[phelel]`

This section contains information specific to the phelel code and must be
included to execute the {ref}`velph_phelel_subcommand` subcommand.

```toml
[phelel]
supercell_dimension = [2, 2, 2]
amplitude = 0.03
diagonal = false
plusminus = true
nosym = false
```

| Key                   | Type                  | Default | Meaning                                                                |
| --------------------- | --------------------- | ------- | ---------------------------------------------------------------------- |
| `supercell_dimension` | list[int] (3)         | —       | Diagonal supercell expansion of the unit cell.                         |
| `supercell_matrix`    | list[list[int]] (3x3) | —       | Full supercell matrix (alternative to `supercell_dimension`).          |
| `amplitude`           | float                 | `0.03`  | Displacement distance in Angstrom.                                     |
| `diagonal`            | bool                  | `false` | Whether to include displacements along non-axis (diagonal) directions. |
| `plusminus`           | bool or `"auto"`      | `true`  | Whether to add the opposite (minus) displacement of each displacement. |
| `nosym`               | bool                  | `false` | Disable symmetry reduction of displacements.                           |
<!-- Hidden until site mixture is public:
| `site_mixture_scheme` | `"merge"` or `"split"` | —      | Scheme of site mixture, written by `velph init` with `--site-mixture`. |
-->

The default column gives the value used when the key is not written.

```{note}
Give either `supercell_dimension` (three integers) or `supercell_matrix` (a 3x3
matrix). One of them is required. These are set with the `velph init` options
`--dim`, `--supercell-matrix` and `--max-num-atoms` (see {ref}`velph_init`).
```

## `[phonopy]` and `[phono3py]`

These sections contain the supercell and displacement settings for harmonic
(phonopy) and anharmonic (phono3py) phonon calculations, and must be included to
execute the {ref}`velph_phonopy_subcommand` and {ref}`velph_phono3py_subcommand`
subcommands, respectively.

```toml
[phonopy]
supercell_dimension = [2, 2, 2]
amplitude = 0.01
diagonal = false
plusminus = true
nosym = false
```

They have the keys of `[phelel]` with the same meanings and defaults, except
that the default `amplitude` of `[phonopy]` is `0.01`. These defaults are the
default displacement distances of phonopy (0.01) and phono3py (0.03) for VASP.
In addition, they have the following key:

| Key                   | Type           | Meaning                                                                 |
| --------------------- | -------------- | ----------------------------------------------------------------------- |
| `number_of_snapshots` | int (optional) | Number of supercells with random directional displacements to generate. |

The phonon part of phelel and the second-order (fc2) part of phono3py can be
computed with the supercell of `[phonopy]`, which can be larger than the
supercell of `[phelel]` or `[phono3py]`. Run `velph phonopy init` first, and then
`velph phelel init` or `velph phono3py init` with `--phonon-from-phonopy`, which
takes the supercells with displacements from `phonopy/phonopy_disp.yaml`.

## `[vasp]`

The `[vasp]` section holds the VASP `INCAR` tags, k-point sampling, and
job-script settings, organized into one input set `[vasp.<input set>]` per kind
of calculation. Because it is the largest section, it is documented on its own
page: {ref}`velph_toml_vasp`.

## `[scheduler]`

Job scripts `_job.sh` for the calculations are generated from this section.
`velph init` writes it only when its template has `[scheduler]`. Without
`[scheduler]`, no `_job.sh` is written. See {ref}`velph_toml_scheduler`.

## `[symmetry]`, `[unitcell]`, `[primitive_cell]`

These auto-generated sections describe the crystal structure and its symmetry.
See {ref}`velph_toml_cell`.
