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
| `amplitude`             | float               | `0.03`          | `--amplitude`              |
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

- `cell_for_nac` and `cell_for_relax` accept `"primitive"` or `"unitcell"`.
- `primitive_cell_choice` accepts `"standardized"` or `"reduced"`.
- `max_num_atoms` determines the supercell dimension and must be used together
  with `symmetrize_cell`.
- Give either `supercell_dimension` (three integers) or `supercell_matrix` (nine
  integers), not both.
<!-- Hidden until site mixture is public:
- `site_mixture` and `split_site_mixture` are experimental, and `site_mixture`
  cannot be combined with `magmom`.
-->
- The file-handling options of `velph init` (`--force`, `--template-toml`,
  `--toml-filename`) are command-line only and have no `[init.options]`
  keyword.

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
