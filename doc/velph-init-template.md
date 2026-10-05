(velph_init_template)=
# `velph init` template

The `velph init` command is used to prepare the `velph.toml` file. If no custom
template is specified, a default template is applied. To specify a custom
template, use the following command:

```bash
% velph init CELL_FILENAME PROJECT_FOLDER --template-toml velph-tmpl.toml [OPTIONS]
```

The `velph-tmpl.toml` file (which can have any file name) is similar to the
`velph.toml` file. It can be created by modifying an existing `velph.toml`.
However, a key difference is that `velph-tmpl.toml` may include an
`[init.options]` section that is not present in `velph.toml`. This section
allows you to define default values as substitutes for command-line options used
with `velph init`, as described in the next section.

(velph_init_template_init_options)=
## `[init.options]`

```toml
[init.options]
kspacing = 0.2
kspacing_dense = 0.1
max_num_atoms = 120
```

The `[init.options]` keywords correspond to command-line options for `velph
init`. A value given on the command line overrides the one in `[init.options]`.
To view the available options and their defaults, use:

```bash
% velph init --help
```

The recognized keywords are listed below. Each maps to the `velph init`
command-line option of the same name (with underscores written as hyphens).

| Keyword                 | Type                | Default         | `velph init` option        |
| ----------------------- | ------------------- | --------------- | -------------------------- |
| `amplitude`             | float               | (see below)     | `--amplitude`              |
| `cell_for_nac`          | str                 | `"primitive"`   | `--cell-for-nac`           |
| `cell_for_relax`        | str                 | `"unitcell"`    | `--cell-for-relax`         |
| `diagonal`              | bool                | `false`         | `--diagonal`               |
| `find_primitive`        | bool                | `true`          | `--no-find-primitive`      |
| `kspacing`              | float               | `0.1`           | `--kspacing`               |
| `kspacing_dense`        | float               | `0.05`          | `--kspacing-dense`         |
| `magmom`                | str                 | (none)          | `--magmom`                 |
| `max_num_atoms`         | int                 | (none)          | `--max-num-atoms`          |
| `phelel_nosym`          | bool                | `false`         | `--phelel-nosym`           |
| `plusminus`             | bool or `"auto"`    | `true`          | `--plusminus` / `--auto`   |
| `primitive_cell_choice` | str                 | `"standardized"`| `--primitive-cell-choice`  |
| `supercell_dimension`   | list[int] (3)       | (none)          | `--dim`                    |
| `supercell_matrix`      | list[int] (9)       | (none)          | `--supercell-matrix`       |
| `symmetrize_cell`       | bool                | `false`         | `--symmetrize-cell`        |
| `tolerance`             | float               | `1e-5`          | `--tolerance`              |
| `use_grg`               | bool                | `false`         | `--use-grg`                |
<!-- Hidden until site mixture is public:
| `site_mixture`          | str                 | (none)          | `--site-mixture`           |
| `split_site_mixture`    | bool                | `false`         | `--split-site-mixture`     |
-->

Notes:

- Without `amplitude`, `velph init` writes 0.03 in `[phelel]` and `[phono3py]`
  and 0.01 in `[phonopy]`. These are the default displacement distances of
  phono3py and phonopy for VASP.
- `cell_for_nac` and `cell_for_relax` accept `"primitive"` or `"unitcell"`.
- `primitive_cell_choice` accepts `"standardized"` or `"reduced"`.
- `max_num_atoms` determines the supercell dimension and must be used together
  with `symmetrize_cell`.
- `max_num_atoms`, `supercell_dimension` (three integers), and `supercell_matrix`
  (nine integers) are three ways to give the supercell. Give only one of them in
  `[init.options]`. When two or more of them are given, `velph init` stops with
  an error.
- When the command line gives the supercell, for example with `--dim`, the
  supercell keywords in `[init.options]` are not used. See
  {ref}`velph_init_template_precedence`.
<!-- Hidden until site mixture is public:
- `site_mixture` and `split_site_mixture` are experimental, and `site_mixture`
  cannot be combined with `magmom`.
-->
- The file-handling options of `velph init` (`--force`, `--template-toml`,
  `--toml-filename`) are command-line only and have no `[init.options]`
  keyword.

(velph_init_template_precedence)=
## Order of precedence

The same setting can be given in more than one place. For example, the
displacement amplitude can be given by `--amplitude` on the command line, by
`amplitude` in `[init.options]`, and by `amplitude` in `[phelel]` of the
template. `velph init` takes the value from the first of the following places
that gives it:

1. a command-line option of `velph init`
2. `[init.options]` of the template
3. the sections of the template that have the same layout as in `velph.toml`
4. the default value

This order applies to the settings in the table below.

| Setting   | Command line                                  | `[init.options]`                                      | Template section                                                     |
| --------- | --------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------- |
| amplitude | `--amplitude`                                 | `amplitude`                                           | `amplitude` in `[phelel]`, `[phonopy]`, `[phono3py]`                 |
| diagonal  | `--diagonal`                                  | `diagonal`                                            | `diagonal` in `[phelel]`, `[phonopy]`, `[phono3py]`                  |
| plusminus | `--plusminus` / `--auto`                      | `plusminus`                                           | `plusminus` in `[phelel]`, `[phonopy]`, `[phono3py]`                 |
| supercell | `--max-num-atoms`, `--dim`, `--supercell-matrix` | `max_num_atoms`, `supercell_dimension`, `supercell_matrix` | `supercell_dimension`, `supercell_matrix` in `[phelel]`, `[phonopy]`, `[phono3py]` |
| cell      | `--cell-for-relax`, `--cell-for-nac`          | `cell_for_relax`, `cell_for_nac`                      | `cell` in `[vasp.relax]`, `[vasp.nac]`                               |

A value from the command line or from `[init.options]` is used for all of
`[phelel]`, `[phonopy]`, and `[phono3py]`. A value in a template section is used
only for that section. For example, `amplitude = 0.05` in `[phonopy]` of the
template sets the amplitude of `[phonopy]`, and `[phelel]` gets the default
value 0.03.

A template made from an existing `velph.toml` has `amplitude`, `diagonal`,
`plusminus`, and the supercell in `[phelel]`. These values are used when the
command line and `[init.options]` do not give them. To change one of them for a
new project, give it on the command line, for example `--amplitude 0.02`, and
the template can be used as it is.

The supercell is one setting that can be given in three ways: by the number of
atoms (`max_num_atoms`), by three integers (`supercell_dimension`), or by nine
integers (`supercell_matrix`). In each place, give only one of them. When two or
more of them are given in the same place, `velph init` stops with an error. When
the command line gives one of them, none of the three in `[init.options]` is
used. For example, with `max_num_atoms = 120` in `[init.options]` and
`--dim 3 3 3` on the command line, the supercell is 3x3x3.

INCAR tags follow different rules, which are described in the next section.

(velph_init_template_incar)=
## `[vasp.incar]`

In a template, the `[vasp.incar]` section holds the base INCAR settings that are
common to the whole project, such as the plane-wave cutoff and the
parallelization tags:

```toml
[vasp.incar]
encut = 400
ncore = 4
gga = "PS"
```

At `velph init` these base settings are merged into every
`[vasp.CALC_TYPE.incar]` of the generated `velph.toml`. The calculation-type
defaults and any `[vasp.CALC_TYPE.incar]` settings written in the template take
precedence over them. See {ref}`velph_toml_vasp_incar` for the full merge,
override, and suppression rules, including how the merge happens only at
initialization.
