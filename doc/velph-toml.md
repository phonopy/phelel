(velph_toml)=
# `velph.toml` file

The `velph.toml` file contains the settings required to perform electron-phonon
interaction calculations using the phelel and VASP codes. Ideally, once created,
it should remain unchanged throughout the duration of a calculation project for
the reproducibility.

There are several ways to create a `velph.toml` file:

- Using the default template (see {ref}`velph_init` command)
- Using a user-defined template (`velph init --template-toml`), as explained in
  {ref}`velph_init_template`
- Copying and modifying an existing `velph.toml` file

Regardless of how it is created, the `velph.toml` file is used in conjunction
with the `velph` command to generate VASP input files and operate other tools,
based on the information it contains. This page provides guidance on how to
interpret and effectively use the contents of `velph.toml`.

## Section overview

A `velph.toml` file is organized into the following top-level sections. Only the
sections relevant to a given project are written.

```
velph.toml
├── [phelel]          # supercell and displacements for phelel
├── [phonopy]         # supercell and displacements for phonopy (optional)
├── [phono3py]        # supercell and displacements for phono3py (optional)
├── [vasp]            # VASP INCAR, k-points, and per-step job settings
│   ├── [vasp.incar]            # base INCAR shared by all calc types
│   ├── [vasp.phelel] ...       # one sub-section per calculation type
│   └── ...                     # (see the [vasp] page)
├── [scheduler]       # job-script template and its parameters
├── [symmetry]        # space group and primitive-cell transformation
├── [unitcell]        # crystal structure (unit cell)
└── [primitive_cell]  # crystal structure (primitive cell)
```

The detailed contents of the larger sections are described on separate pages:

- {ref}`[vasp] <velph_toml_vasp>` : VASP input and per-step job settings.
- {ref}`[scheduler] <velph_toml_scheduler>` : job-script generation.
- {ref}`[unitcell], [primitive_cell], [symmetry] <velph_toml_cell>` : crystal
  structure and symmetry.

```{note}
The `[init.options]` section is **not** part of `velph.toml`. It appears only in
a `velph init` template file (`velph-tmpl.toml`) and supplies defaults for the
`velph init` command-line options. See {ref}`velph_init_template_init_options`.
```

(velph_toml_edit)=
## Sections you edit vs. sections that are generated

- **Edited by the user**: `[phelel]`, `[phonopy]`, `[phono3py]`, the `[vasp.*]`
  sub-sections, and `[scheduler]`. These hold the calculation settings and are
  the parts you tune for a project (most conveniently through the
  {ref}`template <velph_init_template>`).
- **Generated automatically**: `[unitcell]`, `[primitive_cell]`, and
  `[symmetry]`. These record the reference crystal structure and its symmetry
  and are normally left untouched, because reproducibility depends on them. See
  {ref}`velph_toml_cell`.

(velph_toml_override)=
## Override and merge rules

Two mechanisms let a base setting be specialized per calculation type:

- **INCAR**: the base `[vasp.incar]` settings are merged into every
  `[vasp.CALC_TYPE.incar]` for any tag not defined there. A base tag can be
  suppressed for a calculation type by setting it to an empty table `{}`. See
  {ref}`velph_toml_vasp_incar`.
- **Scheduler**: the top-level `[scheduler]` settings are overridden by
  `[vasp.CALC_TYPE.scheduler]` for that step only. See
  {ref}`velph_toml_scheduler`.

## `[phelel]`

This section contains information specific to the phelel code and must be
included to execute the {ref}`velph_phelel_subcommand` subcommand.

```toml
[phelel]
version = "0.6.5"
supercell_dimension = [2, 2, 2]
amplitude = 0.03
diagonal = false
plusminus = true
fft_mesh = [30, 30, 30]
```

| Key                          | Type                          | Default | Meaning                                                                 |
| ---------------------------- | ----------------------------- | ------- | ----------------------------------------------------------------------- |
| `version`                    | str                           | —       | Version of phelel that wrote the file (informational).                  |
| `supercell_dimension`        | list[int] (3)                 | —       | Diagonal supercell expansion of the unit cell.                          |
| `supercell_matrix`           | list[list[int]] (3x3)         | —       | Full supercell matrix (alternative to `supercell_dimension`).           |
| `phonon_supercell_dimension` | list[int] (3)                 | —       | Optional larger supercell used for the phonon (force-constant) part.    |
| `phonon_supercell_matrix`    | list[list[int]] (3x3)         | —       | Full matrix form of the phonon supercell.                               |
| `amplitude`                  | float                         | `0.03`  | Displacement distance in Angstrom.                                      |
| `diagonal`                   | bool                          | `false` | Whether to include displacements along non-axis (diagonal) directions.  |
| `plusminus`                  | bool or `"auto"`              | `true`  | Whether to add the opposite (minus) displacement of each displacement.  |
| `nosym`                      | bool                          | `false` | Disable symmetry reduction of displacements.                            |
| `fft_mesh`                   | list[int] (3)                 | —       | FFT mesh of the primitive cell used for the sandwich. Computed from `encut`/`prec` if not given. |

```{note}
Give either `supercell_dimension` (three integers) or `supercell_matrix` (a 3x3
matrix), not both. The same applies to the `phonon_*` pair. These are set with
the `velph init` options `--dim` / `--supercell-matrix` (see {ref}`velph_init`).
```

## `[phonopy]` and `[phono3py]`

These sections contain the supercell and displacement settings for harmonic
(phonopy) and anharmonic (phono3py) phonon calculations, and must be included to
execute the {ref}`velph_phonopy_subcommand` and {ref}`velph_phono3py_subcommand`
subcommands, respectively.

```toml
[phonopy]
supercell_dimension = [2, 2, 2]
amplitude = 0.03
diagonal = false
plusminus = true
```

They share the supercell and displacement keys of `[phelel]`
(`supercell_dimension`/`supercell_matrix`, `amplitude`, `diagonal`,
`plusminus`, `nosym`) with the same meanings and defaults, plus:

| Key                          | Type           | Meaning                                                                   |
| ---------------------------- | -------------- | ------------------------------------------------------------------------- |
| `number_of_snapshots`        | int (optional) | Number of supercells with random directional displacements to generate.   |
| `phonon_supercell_dimension` | list[int] (3)  | (`[phono3py]` only) Larger supercell for the harmonic (fc2) part.         |
| `phonon_supercell_matrix`    | list[list[int]] (3x3) | (`[phono3py]` only) Full matrix form of the fc2 supercell.         |

`version` and `fft_mesh` are specific to `[phelel]` and do not appear here.

## `[vasp]`

The `[vasp]` section holds the VASP `INCAR` tags, k-point sampling, and per-step
job-script settings, organized into one `[vasp.CALC_TYPE]` sub-section per
calculation step. Because it is the largest section, it is documented on its own
page: {ref}`velph_toml_vasp`.

## `[scheduler]`

Job scripts `_job.sh` for the calculation steps are generated from this section.
See {ref}`velph_toml_scheduler`.

## `[symmetry]`, `[unitcell]`, `[primitive_cell]`

These auto-generated sections describe the crystal structure and its symmetry.
See {ref}`velph_toml_cell`.
